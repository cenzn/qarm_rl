from typing import List
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class CommandBridge(Node):
    """Bridges policy joint actions to arm controller JointTrajectory commands."""

    def __init__(self) -> None:
        super().__init__('command_bridge')

        self.declare_parameter('input_topic', '/policy/joint_actions')
        self.declare_parameter('output_topic', '/arm_controller/joint_trajectory')
        # self.declare_parameter('output_topic', '/test/joint_trajectory')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('default_joint_pos_policy', [-1.309, -0.8727, 2.00713, 0.0, 1.5708, -0.2618])
        self.declare_parameter('action_scale', 0.5)
        self.declare_parameter('max_joint_velocity_rad_s', np.pi+np.pi/2)
        self.declare_parameter('publish_rate_hz', 500.0)
        self.declare_parameter('policy_rate_hz', 30.0)
        self.declare_parameter('policy_action_timeout_s', 0.05)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.joint_state_topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.joint_names: List[str] = list(self.get_parameter('joint_names').value)
        self.default_joint_pos_policy = np.array(self.get_parameter('default_joint_pos_policy').value, dtype=np.float64)
        self.action_scale = float(self.get_parameter('action_scale').value)
        self.max_joint_velocity_rad_s = float(self.get_parameter('max_joint_velocity_rad_s').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.policy_rate_hz = float(self.get_parameter('policy_rate_hz').value)
        self.policy_action_timeout_s = float(self.get_parameter('policy_action_timeout_s').value)

        if len(self.joint_names) != 6:
            raise ValueError('joint_names must contain exactly 6 arm joints')
        if self.default_joint_pos_policy.shape[0] != 6:
            raise ValueError('default_joint_pos_policy must have length 6')
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be > 0')

        self.max_delta_per_step = self.max_joint_velocity_rad_s / self.policy_rate_hz
        self.min_step_time_s = 1.0 / self.publish_rate_hz
        self._latest_joint_pos_phys: np.ndarray | None = None
        self._latest_policy_action: np.ndarray | None = None
        self._latest_policy_action_time_s: float | None = None
        self._last_target_policy: np.ndarray | None = None
        self._stale_action_warned = False
        self._stale_hold_sent = False

        self.sub = self.create_subscription(Float64MultiArray, self.input_topic, self._on_policy_action, 20)
        self.joint_state_sub = self.create_subscription(JointState, self.joint_state_topic, self._on_joint_states, 20)
        self.pub = self.create_publisher(JointTrajectory, self.output_topic, 20)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_latest_target)

        self.get_logger().info(
            f'Command bridge ready. Subscribing {self.input_topic} and publishing {self.output_topic}.'
        )

    def _on_joint_states(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        if any(name not in name_to_index for name in self.joint_names):
            return

        self._latest_joint_pos_phys = np.array(
            [float(msg.position[name_to_index[name]]) for name in self.joint_names],
            dtype=np.float64,
        )

    def _apply_velocity_limit(self, target_phys: np.ndarray) -> np.ndarray:
        if self._latest_joint_pos_phys is None:
            return None  # should not happen since we check before applying limits

        lower = self._latest_joint_pos_phys - self.max_delta_per_step
        upper = self._latest_joint_pos_phys + self.max_delta_per_step
        return np.clip(target_phys, lower, upper)

    def _policy_to_physical_arm_pos(self, pos_policy_6: np.ndarray) -> np.ndarray:
        # Reach DDS path publishes default_pos + action*scale directly.
        # Keep the same convention here without additional axis/sign remapping.
        return np.array(pos_policy_6, dtype=np.float64, copy=True)

    def _on_policy_action(self, msg: Float64MultiArray) -> None:
        expected_actions = int(self.default_joint_pos_policy.shape[0])
        if len(msg.data) < expected_actions:
            self.get_logger().warn(
                f'Expected at least {expected_actions} policy action values on input topic; dropping message.'
            )
            return
        raw = np.asarray(msg.data, dtype=np.float64)
        self._latest_policy_action = raw
        # print(f'Received policy action: {raw}')
        self._latest_policy_action_time_s = time.monotonic()

    def _compute_time_from_start(self) -> float:
        if self._latest_policy_action_time_s is None:
            return 1/self.policy_rate_hz
        elapsed = time.monotonic() - self._latest_policy_action_time_s
        remaining = 1/self.policy_rate_hz - elapsed
        return max(self.min_step_time_s, remaining)

    def _publish_latest_target(self) -> None:
        if self._latest_joint_pos_phys is None:
            return

        now_s = time.monotonic()
        use_stale_fallback = (
            self._latest_policy_action is None
            or self._latest_policy_action_time_s is None
            or (now_s - self._latest_policy_action_time_s) > self.policy_action_timeout_s
        )

        if use_stale_fallback:
            if self._stale_hold_sent:
                return

            if self._last_target_policy is not None:
                arm_targets_policy = self._last_target_policy.copy()
            else:
                arm_targets_policy = self.default_joint_pos_policy.copy()

            if not self._stale_action_warned:
                self.get_logger().warn('Policy action stream stale (>50ms). Holding last commanded target.')
                self._stale_action_warned = True
            secs = 0
            nanosecs = 200000000
            self._stale_hold_sent = True
        else:
            raw = self._latest_policy_action
            self._stale_action_warned = False
            self._stale_hold_sent = False
            # point_time_s = self._compute_time_from_start()
            # secs = int(point_time_s)
            # nanosecs = int((point_time_s - secs) * 1e9)
            secs = int(0)
            nanosecs = int(12000000)
            arm_targets_policy = self.default_joint_pos_policy + raw[:6] * self.action_scale
            self._last_target_policy = arm_targets_policy.copy()

        arm_targets_phys = self._policy_to_physical_arm_pos(arm_targets_policy)
        arm_targets_phys = self._apply_velocity_limit(arm_targets_phys)
        if arm_targets_phys is None:
            return

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = arm_targets_phys.tolist()
        point.time_from_start.sec = secs
        point.time_from_start.nanosec = nanosecs
        traj.points = [point]
        self.pub.publish(traj)

def main() -> None:
    rclpy.init()
    node = CommandBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
