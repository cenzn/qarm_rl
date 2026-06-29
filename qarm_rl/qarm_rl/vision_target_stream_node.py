import importlib.util
import time
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    from quanser.common import Timeout
except ImportError:
    from quanser.communications import Timeout


def _load_basic_stream_class():
    """Load BasicStream from stream.py shipped with this package."""
    try:
        from stream import BasicStream

        return BasicStream
    except Exception:
        pass

    candidate_paths: list[Path] = []
    try:
        share_dir = Path(get_package_share_directory('qarm_rl'))
        candidate_paths.append(share_dir / 'resource' / 'stream.py')
    except Exception:
        pass

    # Source workspace fallback: qarm_rl/qarm_rl/vision_target_stream_node.py -> qarm_rl/resource/stream.py
    candidate_paths.append(Path(__file__).resolve().parents[1] / 'resource' / 'stream.py')

    for stream_path in candidate_paths:
        if not stream_path.exists():
            continue
        spec = importlib.util.spec_from_file_location('qarm_rl_stream_resource', str(stream_path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'BasicStream'):
            return module.BasicStream

    raise ImportError('Could not load BasicStream from stream.py. Ensure qarm_rl/resource/stream.py is installed.')


class VisionTargetStreamNode(Node):
    """Receives object position in camera frame and publishes base-frame target pose."""

    def __init__(self) -> None:
        super().__init__('vision_target_stream_node')

        self.declare_parameter('stream_uri', 'tcpip://localhost:18086')
        self.declare_parameter('stream_agent', 'C')
        self.declare_parameter('stream_send_buffer_size', 1460)
        self.declare_parameter('stream_recv_buffer_size', 1460)
        self.declare_parameter('stream_non_blocking', False)
        self.declare_parameter('stream_reshape_order', 'F')
        self.declare_parameter('stream_receive_iterations', 2)
        self.declare_parameter('stream_timeout_ns', 1)
        self.declare_parameter('no_data_warn_threshold', 10)
        self.declare_parameter('control_rate_hz', 30.0)

        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])

        self.declare_parameter('target_topic', '/policy/target_pose_xyz_xyzw')
        self.declare_parameter('target_quat_xyzw', [0.70710678, 0.70710678, 0.0, 0.0])
        self.declare_parameter('target_z', 0.19)
        self.declare_parameter('cam_zero_epsilon', 1e-6)
        self.declare_parameter('stability_window_size', 10)
        self.declare_parameter('stability_position_tolerance_m', 0.2)
        self.declare_parameter('stable_publish_delay_s', 0.3)
        self.declare_parameter('home_joint_pos', [-1.309, -0.8727, 2.00713, 0.0, 1.5708, -0.2618])
        self.declare_parameter('home_tolerance_rad', 0.01)
        self.declare_parameter('home_hold_cycles', 10)

        self.stream_uri = self.get_parameter('stream_uri').get_parameter_value().string_value
        self.stream_agent = self.get_parameter('stream_agent').get_parameter_value().string_value
        self.stream_send_buffer_size = int(self.get_parameter('stream_send_buffer_size').value)
        self.stream_recv_buffer_size = int(self.get_parameter('stream_recv_buffer_size').value)
        self.stream_non_blocking = bool(self.get_parameter('stream_non_blocking').value)
        self.stream_reshape_order = self.get_parameter('stream_reshape_order').get_parameter_value().string_value
        self.stream_receive_iterations = int(self.get_parameter('stream_receive_iterations').value)
        self.stream_timeout_ns = int(self.get_parameter('stream_timeout_ns').value)
        self.no_data_warn_threshold = int(self.get_parameter('no_data_warn_threshold').value)
        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)

        self.joint_state_topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.joint_names: List[str] = list(self.get_parameter('joint_names').value)

        self.target_topic = self.get_parameter('target_topic').get_parameter_value().string_value
        self.target_quat_xyzw = np.array(self.get_parameter('target_quat_xyzw').value, dtype=np.float64)
        self.target_z = np.array(self.get_parameter('target_z').value, dtype=np.float64)
        self.cam_zero_epsilon = float(self.get_parameter('cam_zero_epsilon').value)
        self.stability_window_size = int(self.get_parameter('stability_window_size').value)
        self.stability_position_tolerance_m = float(self.get_parameter('stability_position_tolerance_m').value)
        self.stable_publish_delay_s = float(self.get_parameter('stable_publish_delay_s').value)
        self.home_joint_pos = np.array(self.get_parameter('home_joint_pos').value, dtype=np.float64)
        self.home_tolerance_rad = float(self.get_parameter('home_tolerance_rad').value)
        self.home_hold_cycles = int(self.get_parameter('home_hold_cycles').value)

        if len(self.joint_names) != 6:
            raise ValueError('joint_names must contain exactly 6 arm joints')
        if self.target_quat_xyzw.shape[0] != 4:
            raise ValueError('target_quat_xyzw must contain 4 values [qx, qy, qz, qw]')
        if self.control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be > 0')
        if self.stream_receive_iterations < 1:
            raise ValueError('stream_receive_iterations must be >= 1')
        if self.stability_window_size < 2:
            raise ValueError('stability_window_size must be >= 2')
        if self.stability_position_tolerance_m <= 0.0:
            raise ValueError('stability_position_tolerance_m must be > 0')
        if self.stable_publish_delay_s <= 0.0:
            raise ValueError('stable_publish_delay_s must be > 0')
        if self.home_joint_pos.shape[0] != len(self.joint_names):
            raise ValueError('home_joint_pos length must match joint_names length')
        if self.home_tolerance_rad <= 0.0:
            raise ValueError('home_tolerance_rad must be > 0')
        if self.home_hold_cycles < 1:
            raise ValueError('home_hold_cycles must be >= 1')

        basic_stream_cls = _load_basic_stream_class()
        self.stream = basic_stream_cls(
            self.stream_uri,
            agent=self.stream_agent,
            sendBufferSize=self.stream_send_buffer_size,
            recvBufferSize=self.stream_recv_buffer_size,
            receiveBuffer=np.zeros(3, dtype=np.float64),
            nonBlocking=self.stream_non_blocking,
            reshapeOrder=self.stream_reshape_order,
        )
        self.stream_timeout = Timeout(seconds=0, nanoseconds=self.stream_timeout_ns)

        self._latest_joint_pos_phys: np.ndarray | None = None
        self._latest_joint_vel_phys: np.ndarray | None = None
        self._joint_sum_pos: np.ndarray | None = None
        self._joint_sum_vel: np.ndarray | None = None
        self._joint_sample_count = 0
        self._state_wait_stable = 'wait_stable'
        self._state_stable_delay = 'stable_delay'
        self._state_idle_wait_home = 'idle_wait_home'
        self._state = self._state_wait_stable
        self._stable_since_s: float | None = None
        self._cam_history: List[np.ndarray] = []
        self._home_reached_hold_count = 0
        self._connected_prev = False
        self._no_data_counter = 0
        self._last_joint_warn_time = 0.0

        target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_pub = self.create_publisher(Float64MultiArray, self.target_topic, target_qos)
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._on_joint_states,
            20,
        )
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._on_timer)

        self.get_logger().info(
            f'Vision target stream node ready. stream_uri={self.stream_uri}, '
            f'joint_state_topic={self.joint_state_topic}, target_topic={self.target_topic}'
        )

    def _set_state(self, new_state: str) -> None:
        if self._state == new_state:
            return
        self.get_logger().info(f'Vision state: {self._state} -> {new_state}')
        self._state = new_state

    def _clear_cam_tracking(self) -> None:
        self._cam_history.clear()
        self.stream.clientStream.flush()
        self._stable_since_s = None

    def _append_cam_sample(self, cam_xyz: np.ndarray) -> None:
        self._cam_history.append(np.array(cam_xyz[:3], dtype=np.float64, copy=True))
        if len(self._cam_history) > self.stability_window_size:
            self._cam_history.pop(0)

    def _camera_is_stable(self) -> tuple[bool, np.ndarray | None]:
        if len(self._cam_history) < self.stability_window_size:
            return False, None

        samples = np.vstack(self._cam_history)
        mean_xyz = np.mean(samples, axis=0)
        distances = np.linalg.norm(samples - mean_xyz, axis=1)
        max_dist = float(np.max(distances))
        stable = max_dist <= self.stability_position_tolerance_m
        # self.get_logger().info(f'Camera stable: {stable} meas: {max_dist} tolarence {self.stability_position_tolerance_m}')
        return stable, mean_xyz

    def _joint_is_home(self) -> bool:
        if self._latest_joint_pos_phys is None:
            return False
        joint_error = np.abs(self._latest_joint_pos_phys - self.home_joint_pos)
        return bool(np.max(joint_error) <= self.home_tolerance_rad)

    def _on_joint_states(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        if any(name not in name_to_index for name in self.joint_names):
            return

        def velocity_at(index: int) -> float:
            if index < len(msg.velocity):
                return float(msg.velocity[index])
            return 0.0

        positions: List[float] = []
        velocities: List[float] = []
        for name in self.joint_names:
            idx = name_to_index[name]
            positions.append(float(msg.position[idx]))
            velocities.append(velocity_at(idx))
        
        pos = np.array(positions, dtype=np.float64)
        vel = np.array(velocities, dtype=np.float64)

        if self._joint_sum_pos is None:
            self._joint_sum_pos = np.zeros_like(pos)
            self._joint_sum_vel = np.zeros_like(vel)

        self._joint_sum_pos += pos
        self._joint_sum_vel += vel
        self._joint_sample_count += 1

    def _on_timer(self) -> None:
        if self._joint_sample_count > 0 and self._joint_sum_pos is not None and self._joint_sum_vel is not None:
            inv_count = 1.0 / float(self._joint_sample_count)
            self._latest_joint_pos_phys = self._joint_sum_pos * inv_count
            self._latest_joint_vel_phys = self._joint_sum_vel * inv_count
            self._joint_sum_pos.fill(0.0)
            self._joint_sum_vel.fill(0.0)
            self._joint_sample_count = 0

        if self._latest_joint_pos_phys is None:
            now = time.time()
            if now - self._last_joint_warn_time > 2.0:
                self.get_logger().warn('No complete arm joint state available yet.')
                self._last_joint_warn_time = now
            return

        if not self.stream.connected:
            self.stream.checkConnection(timeout=self.stream_timeout)

        if self.stream.connected and not self._connected_prev:
            self.get_logger().info('Connection to stream client was successful.')
        self._connected_prev = self.stream.connected

        if not self.stream.connected:
            return
        
        # moving the receive logic here to ensure we get the latest camera data 
        # even in the idle_wait_home state
        recv_flag, bytes_received = self.stream.receive(
            iterations=self.stream_receive_iterations,
            timeout=self.stream_timeout,
        )

        if self._state == self._state_idle_wait_home:
            # Ignore camera positions in idle state.
            self.stream.receive(
                iterations=self.stream_receive_iterations,
                timeout=self.stream_timeout,
            )
            if self._joint_is_home():
                self._home_reached_hold_count += 1
                if self._home_reached_hold_count >= self.home_hold_cycles:
                    self._home_reached_hold_count = 0
                    self._clear_cam_tracking()
                    self._set_state(self._state_wait_stable)
            else:
                self._home_reached_hold_count = 0
            return

        if int(recv_flag) <= 0 or int(bytes_received) <= 0:
            self._no_data_counter += 1
            if self._no_data_counter > self.no_data_warn_threshold:
                self.get_logger().warn('Stream connected but no object data received recently.')
                self._no_data_counter = 0
            if self._state == self._state_stable_delay:
                self._clear_cam_tracking()
                self._set_state(self._state_wait_stable)
            return

        self._no_data_counter = 0

        # self.get_logger().info(f'buffer shape: {self.stream.receiveBuffer.shape}, dtype: {self.stream.receiveBuffer.dtype}, buffer: {self.stream.receiveBuffer}')
        # cam_xyz = np.asarray(self.stream.receiveBuffer, dtype=np.float64).reshape(-1)
        cam_xyz = self.stream.receiveBuffer
        # self.get_logger().info(f'cam_xyz: {cam_xyz}')
        if cam_xyz.shape[0] < 3:
            self.get_logger().warn('Received stream payload with fewer than 3 elements; ignoring.')
            if self._state == self._state_stable_delay:
                self._clear_cam_tracking()
                self._set_state(self._state_wait_stable)
            return

        cam_xyz = np.array(cam_xyz[:3], dtype=np.float64, copy=False)
        if float(np.linalg.norm(cam_xyz)) <= self.cam_zero_epsilon:
            if self._state == self._state_stable_delay:
                self.get_logger().info('Camera stability lost: zero object pose seen. Restarting stability timer.')
            self._clear_cam_tracking()
            self._set_state(self._state_wait_stable)
            return

        self._append_cam_sample(cam_xyz)
        cam_stable, cam_mean_xyz = self._camera_is_stable()

        if not cam_stable or cam_mean_xyz is None:
            if self._state == self._state_stable_delay:
                self.get_logger().info('Camera stability lost. Restarting stability timer.')
                self._clear_cam_tracking()
                self._set_state(self._state_wait_stable)
            return

        now_s = time.monotonic()
        if self._state == self._state_wait_stable:
            self._stable_since_s = now_s
            self._set_state(self._state_stable_delay)
            return

        if self._stable_since_s is None:
            self._stable_since_s = now_s
            return

        if (now_s - self._stable_since_s) < self.stable_publish_delay_s:
            return

        pos_dh = self._physical_to_DH_joint_pos(self._latest_joint_pos_phys)
        t_0_to_cam = self._forward_kinematics_DH(pos_dh)

        obj_cam_h = np.array([cam_mean_xyz[0], cam_mean_xyz[1], cam_mean_xyz[2], 1.0], dtype=np.float64)
        obj_base_h = t_0_to_cam @ obj_cam_h

        target_msg = Float64MultiArray()
        target_msg.data = [
            float(obj_base_h[0]),
            float(obj_base_h[1]),
            float(self.target_z),
            float(self.target_quat_xyzw[0]),
            float(self.target_quat_xyzw[1]),
            float(self.target_quat_xyzw[2]),
            float(self.target_quat_xyzw[3]),
        ]
        self.target_pub.publish(target_msg)
        self.get_logger().info(f'Published stable target pose: {np.round(np.asarray(target_msg.data), 5)}')

        self._clear_cam_tracking()
        self._home_reached_hold_count = 0
        self._set_state(self._state_idle_wait_home)

    def _physical_to_DH_joint_pos(self, pos_phys: np.ndarray) -> np.ndarray:
        pos_policy = np.array(pos_phys, dtype=np.float64, copy=True)
        pos_policy[1] = -pos_policy[1]
        pos_policy[2] = -pos_policy[2]
        pos_policy[3] = -pos_policy[3]
        pos_policy[4] = pos_policy[4] - np.pi / 2.0
        pos_policy[5] = -pos_policy[5]

        pos_policy[1] = pos_policy[1] + np.pi / 2.0
        pos_policy[3] = pos_policy[3] + np.pi / 2.0
        pos_policy[4] = pos_policy[4] + np.pi / 2.0

        return pos_policy

    def _forward_kinematics_DH(self, pos_dh: np.ndarray) -> np.ndarray:
        def transform_from_dh_row(dh_row: list[float]) -> np.ndarray:
            a = dh_row[0]
            alpha = dh_row[1]
            d = dh_row[2]
            theta = dh_row[3]

            t_r_z = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            t_t_z = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, d],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            t_t_x = np.array(
                [
                    [1, 0, 0, a],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            t_r_x = np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha), 0],
                    [0, np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            return t_r_z @ t_t_z @ t_t_x @ t_r_x

        l1 = 0.1715
        l2 = 0.1215
        l3 = 0.247
        l4 = 0.124
        l5 = 0.2195
        l6 = 0.1155
        l7 = 0.1155
        e = 0.234

        dh = [
            [0.0, np.pi / 2.0, l1, pos_dh[0]],
            [l3, 0.0, l2, pos_dh[1]],
            [l5, 0.0, -l4, pos_dh[2]],
            [0.0, np.pi / 2.0, l6, pos_dh[3]],
            [0.0, -np.pi / 2.0, l7, pos_dh[4]],
            [0.0, 0.0, e, pos_dh[5]],
        ]

        t01 = transform_from_dh_row(dh[0])
        t12 = transform_from_dh_row(dh[1])
        t23 = transform_from_dh_row(dh[2])
        t34 = transform_from_dh_row(dh[3])
        t45 = transform_from_dh_row(dh[4])
        t56 = transform_from_dh_row(dh[5])

        t06 = t01 @ t12 @ t23 @ t34 @ t45 @ t56

        t_ee_to_camera = np.array(
            [
                [0, 1, 0, 0],
                [-1, 0, 0, -0.08],
                [0, 0, 1, -0.154],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        return t06 @ t_ee_to_camera

    def destroy_node(self) -> bool:
        try:
            self.stream.terminate()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VisionTargetStreamNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
