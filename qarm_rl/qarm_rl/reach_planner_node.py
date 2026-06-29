import time
from typing import List, Optional

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ReachPlannerNode(Node):
    """Publishes a latched target pose and starts/stops policy episode based on EE goal proximity."""

    def __init__(self) -> None:
        super().__init__('reach_planner_node')

        self._PHASE_WAIT_TARGET = 'wait_target'
        self._PHASE_PICK_POLICY = 'pick_policy'
        self._PHASE_WAIT_POLICY_STOP = 'wait_policy_stop'
        self._PHASE_GRASP_CLOSE = 'grasp_close'
        self._PHASE_MOVE_MIDPOINT = 'move_midpoint'
        self._PHASE_MOVE_PLACE = 'move_place'
        self._PHASE_DROP_OPEN = 'drop_open'
        self._PHASE_RETURN_MIDPOINT = 'return_midpoint'
        self._PHASE_MOVE_HOME = 'move_home'
        self._PHASE_DONE = 'done'

        self.declare_parameter('target_topic', '/policy/target_pose_xyz_xyzw')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('trajectory_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('gripper_action_name', '/gripper_controller/gripper_cmd')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('midpoint_joint_pos', [0.05, 0.05, 1.5,-1.5, 1.57, 0.05])
        self.declare_parameter('home_joint_pos', [-1.309, -0.8727, 2.00713, 0.0, 1.5708, -0.2618])
        self.declare_parameter('place_joint_pos', [ 1.76,0.516,0.618,0.4648,1.57,-0.1708])
        self.declare_parameter('position_tolerance_m', 0.035)
        self.declare_parameter('waypoint_joint_tolerance_rad', 0.1)
        self.declare_parameter('waypoint_hold_cycles', 5)
        self.declare_parameter('trajectory_time_from_start_sec', 1.5)
        self.declare_parameter('check_rate_hz', 20.0)
        self.declare_parameter('state_timeout_s', 5.0)
        self.declare_parameter('gripper_close_position', 0.7)
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('gripper_max_effort', 20.0)
        self.declare_parameter('gripper_wait_s', 1.0)

        self.target_topic = self.get_parameter('target_topic').get_parameter_value().string_value
        self.joint_state_topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.trajectory_topic = self.get_parameter('trajectory_topic').get_parameter_value().string_value
        self.gripper_action_name = self.get_parameter('gripper_action_name').get_parameter_value().string_value
        self.joint_names: List[str] = list(self.get_parameter('joint_names').value)
        self.midpoint_joint_pos = np.array(self.get_parameter('midpoint_joint_pos').value, dtype=np.float64)
        self.home_joint_pos = np.array(self.get_parameter('home_joint_pos').value, dtype=np.float64)
        self.place_joint_pos = np.array(self.get_parameter('place_joint_pos').value, dtype=np.float64)
        self.position_tolerance_m = float(self.get_parameter('position_tolerance_m').value)
        self.waypoint_joint_tolerance_rad = float(self.get_parameter('waypoint_joint_tolerance_rad').value)
        self.waypoint_hold_cycles = int(self.get_parameter('waypoint_hold_cycles').value)
        self.trajectory_time_from_start_sec = float(self.get_parameter('trajectory_time_from_start_sec').value)
        self.check_rate_hz = float(self.get_parameter('check_rate_hz').value)
        self.state_timeout_s = float(self.get_parameter('state_timeout_s').value)
        self.gripper_close_position = float(self.get_parameter('gripper_close_position').value)
        self.gripper_open_position = float(self.get_parameter('gripper_open_position').value)
        self.gripper_max_effort = float(self.get_parameter('gripper_max_effort').value)
        self.gripper_wait_s = float(self.get_parameter('gripper_wait_s').value)

        if self.midpoint_joint_pos.shape[0] != len(self.joint_names):
            raise ValueError('midpoint_joint_pos must match joint_names length')
        if self.home_joint_pos.shape[0] != len(self.joint_names):
            raise ValueError('home_joint_pos must match joint_names length')
        if self.place_joint_pos.shape[0] != len(self.joint_names):
            raise ValueError('place_joint_pos must match joint_names length')
        if self.position_tolerance_m <= 0.0:
            raise ValueError('position_tolerance_m must be > 0')
        if self.waypoint_joint_tolerance_rad <= 0.0:
            raise ValueError('waypoint_joint_tolerance_rad must be > 0')
        if self.waypoint_hold_cycles < 1:
            raise ValueError('waypoint_hold_cycles must be >= 1')
        if self.trajectory_time_from_start_sec <= 0.0:
            raise ValueError('trajectory_time_from_start_sec must be > 0')
        if self.check_rate_hz <= 0.0:
            raise ValueError('check_rate_hz must be > 0')
        if self.state_timeout_s <= 0.0:
            raise ValueError('state_timeout_s must be > 0')
        if self.gripper_wait_s <= 0.0:
            raise ValueError('gripper_wait_s must be > 0')

        self._latest_joint_pos: Optional[np.ndarray] = None
        self._policy_running = False
        self._start_request_sent = False
        self._stop_request_sent = False
        self._have_target_from_topic = False
        self._phase = self._PHASE_WAIT_TARGET
        self._phase_start_time = time.monotonic()
        self._stability_counter = 0
        self._gripper_deadline_s: Optional[float] = None
        self._waypoint_sent = False

        target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_pub = self.create_publisher(Float64MultiArray, self.target_topic, target_qos)
        self.trajectory_pub = self.create_publisher(JointTrajectory, self.trajectory_topic, 20)

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._on_joint_state,
            20,
        )
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            self.target_topic,
            self._on_target_pose,
            target_qos,
        )

        self.start_client: Client = self.create_client(Trigger, '/policy/start_episode')
        self.stop_client: Client = self.create_client(SetBool, '/policy/stop_episode')
        self.gripper_client = ActionClient(self, GripperCommand, self.gripper_action_name)

        self.control_timer = self.create_timer(1.0 / self.check_rate_hz, self._control_step)

        self.get_logger().info(
            f'Reach planner ready. target_topic={self.target_topic}, joint_state_topic={self.joint_state_topic}, '
            f'traj_topic={self.trajectory_topic}, tol={self.position_tolerance_m:.4f} m'
        )

    def _set_phase(self, phase: str) -> None:
        if self._phase == phase:
            return
        self._phase = phase
        self._phase_start_time = time.monotonic()
        self._stability_counter = 0
        self._waypoint_sent = False
        self.get_logger().info(f'Planner phase -> {phase}')

    def _reset_for_next_target(self) -> None:
        self._policy_running = False
        self._start_request_sent = False
        self._stop_request_sent = False
        self._have_target_from_topic = False
        self._gripper_deadline_s = None
        self._set_phase(self._PHASE_WAIT_TARGET)

    def _state_timed_out(self) -> bool:
        return (time.monotonic() - self._phase_start_time) > self.state_timeout_s


    def _publish_joint_waypoint(self, joint_positions: np.ndarray) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joint_positions.tolist()]
        point.time_from_start.sec = int(self.trajectory_time_from_start_sec)
        point.time_from_start.nanosec = int((self.trajectory_time_from_start_sec - int(self.trajectory_time_from_start_sec)) * 1e9)
        msg.points = [point]
        self.trajectory_pub.publish(msg)

    def _joint_target_reached(self, target: np.ndarray) -> bool:
        if self._latest_joint_pos is None:
            return False
        err = np.abs(self._latest_joint_pos - target)
        reached = bool(np.max(err) <= self.waypoint_joint_tolerance_rad)
        if reached:
            self._stability_counter += 1
        else:
            self._stability_counter = 0
        return self._stability_counter >= self.waypoint_hold_cycles

    def _send_gripper_goal(self, position: float) -> None:
        if not self.gripper_client.server_is_ready():
            self.get_logger().warn('Gripper action server not available yet; skip send this cycle.')
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(self.gripper_max_effort)
        self.gripper_client.send_goal_async(goal)

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        if any(name not in name_to_index for name in self.joint_names):
            return

        self._latest_joint_pos = np.array(
            [float(msg.position[name_to_index[name]]) for name in self.joint_names],
            dtype=np.float64,
        )

    def _on_target_pose(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 7:
            self.get_logger().warn('Expected 7 target pose values [x, y, z, qx, qy, qz, qw]. Ignoring.')
            return

        self.target_pose = np.asarray(msg.data, dtype=np.float64)
        first_target = not self._have_target_from_topic
        self._have_target_from_topic = True

        if first_target:
            self.get_logger().info(f'Received target pose from topic: {np.round(self.target_pose, 5)}')

        if self._phase == self._PHASE_WAIT_TARGET:
            self._set_phase(self._PHASE_PICK_POLICY)

    def _request_start_episode(self) -> None:
        if not self.start_client.service_is_ready():
            return
        if self._start_request_sent:
            return
        

        req = Trigger.Request()
        future = self.start_client.call_async(req)
        future.add_done_callback(self._on_start_response)
        self._start_request_sent = True

    def _on_start_response(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(f'Start service call failed: {exc}')
            self._start_request_sent = False
            return

        if resp.success:
            self._policy_running = True
            self.get_logger().info(f'Policy started: {resp.message}')
            self._start_request_sent = False
        else:
            self.get_logger().warn(f'Policy start rejected: {resp.message}')
            self._start_request_sent = False

    def _request_stop_episode(self) -> None:
        if self._stop_request_sent:
            return
        if not self.stop_client.service_is_ready():
            return

        req = SetBool.Request()
        req.data = True
        future = self.stop_client.call_async(req)
        future.add_done_callback(self._on_stop_response)
        self._stop_request_sent = True

    def _on_stop_response(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(f'Stop service call failed: {exc}')
            self._stop_request_sent = False
            return

        if resp.success:
            self._policy_running = False
            self._stop_request_sent = False
            self.get_logger().info(f'Policy stopped: {resp.message}')
            if self._phase == self._PHASE_WAIT_POLICY_STOP:
                self._send_gripper_goal(self.gripper_close_position)
                self._gripper_deadline_s = time.monotonic() + self.gripper_wait_s
                self._set_phase(self._PHASE_GRASP_CLOSE)
        else:
            self.get_logger().warn(f'Policy stop rejected: {resp.message}')
            self._stop_request_sent = False

    def _physical_to_DH_joint_pos(self, pos_phys: np.ndarray) -> np.ndarray:
        pos_policy = np.array(pos_phys, dtype=np.float64, copy=True)
        pos_policy[1] = -pos_policy[1]
        pos_policy[2] = -pos_policy[2]
        pos_policy[3] = -pos_policy[3]
        pos_policy[4] = pos_policy[4] - np.pi / 2.0
        pos_policy[5] = -pos_policy[5]

        #convert to DH
        pos_policy[1] = pos_policy[1] + np.pi / 2.0
        pos_policy[3] = pos_policy[3] + np.pi / 2.0
        pos_policy[4] = pos_policy[4] + np.pi / 2.0
        
        # self.get_logger().info(
        #         f'pos_phys: {pos_phys}, pos_DH: {pos_policy}'
        #     )
        
        return pos_policy

    def _forward_kinematics_DH (self, pos_DH: np.ndarray) -> np.ndarray:
        
        def transformFromDHrow(DHrow):
            a = DHrow[0]
            alpha = DHrow[1]
            d = DHrow[2]
            theta = DHrow[3]
            # Rotation Transformation about z axis by theta
            T_R_z = np.array(
                    [[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta),  np.cos(theta), 0, 0],
                     [0         ,  0, 1, 0],
                     [0         ,  0, 0, 1]],
                     dtype=np.float64
            )
            # Translation Transformation along z axis by d
            T_T_z = np.array(
                    [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]],
                     dtype=np.float64
            )
            # Translation Transformation along x axis by a
            T_T_x = np.array(
                    [[1, 0, 0, a],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]],
                     dtype=np.float64
            )
            # Rotation Transformation about x axis by alpha
            T_R_x = np.array(
                    [[1, 0, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha), 0],   
                     [0, np.sin(alpha),  np.cos(alpha), 0],
                     [0, 0, 0, 1]],
                     dtype=np.float64
            )
            T = T_R_z @ T_T_z @ T_T_x @ T_R_x
            # self.get_logger().info(f'a:{a}, alpha:{alpha}, d:{d}, theta:{theta}, transformations,{T}')
            return T
        
        # link lengths
        L1 = 0.1715
        L2 = 0.1215
        L3 = 0.247
        L4 = 0.124
        L5 = 0.2195
        L6 = 0.1155
        L7 = 0.1155
        E = 0.244; # distance link 6; reach policy reases link 6 as EE.

        #         a   alpha     d      theta
        DH = [[   0, np.pi/2,   L1,  pos_DH[0]],
              [  L3,       0,   L2,  pos_DH[1]],
              [  L5,       0,  -L4,  pos_DH[2]],
              [   0, np.pi/2,   L6,  pos_DH[3]],
              [   0,-np.pi/2,   L7,  pos_DH[4]],
              [   0,       0,    E,  pos_DH[5]]]
        T01 = transformFromDHrow( DH[0] )
        T12 = transformFromDHrow( DH[1] )
        T23 = transformFromDHrow( DH[2] )
        T34 = transformFromDHrow( DH[3] )
        T45 = transformFromDHrow( DH[4] )
        # self.get_logger().info(f'DH table,{DH}')

        # transforms relative to world frame {0}
        T02 = T01@T12
        T03 = T02@T23
        T04 = T03@T34
        T05 = T04@T45
        # self.get_logger().info(f'T05: {T05}')
        link6_xyz = T05[:3, 3]

        return link6_xyz
    
    def _compute_ee_position(self, joint_positions: np.ndarray) -> Optional[np.ndarray]:
        """
        Return np.array([x, y, z], dtype=np.float64) when FK is implemented.
        Returning None disables goal-distance stopping.
        """
        pos_DH = self._physical_to_DH_joint_pos(joint_positions)
    
        return self._forward_kinematics_DH(pos_DH)

    def _control_step(self) -> None:
        if not self._have_target_from_topic:
            self._set_phase(self._PHASE_WAIT_TARGET)
            return

        if self._phase == self._PHASE_PICK_POLICY:

            if not self._policy_running or self._latest_joint_pos is None:
                self._request_start_episode()
                if self._state_timed_out():
                    self.get_logger().error('Timed out waiting for pick policy progress.')
                    self._set_phase(self._PHASE_DONE)
                return

            ee_xyz = self._compute_ee_position(self._latest_joint_pos)
            if ee_xyz is None:
                return
            target_xyz = self.target_pose[:3]
            dist = float(np.linalg.norm(ee_xyz - target_xyz))
            # self.get_logger().info(
            #     f'Pick phase: ee={np.round(ee_xyz, 4)}, target={np.round(target_xyz, 4)}, dist={dist:.4f} m'
            # )
            if dist <= self.position_tolerance_m:
                self.get_logger().info(
                    f'Pick target reached. dist={dist:.4f} <= {self.position_tolerance_m:.4f}. Stopping policy.'
                )
                self._request_stop_episode()
                self._set_phase(self._PHASE_WAIT_POLICY_STOP)
            return

        if self._phase == self._PHASE_WAIT_POLICY_STOP:
            if self._state_timed_out():
                self.get_logger().error('Timed out waiting for policy to stop.')
                self._set_phase(self._PHASE_DONE)
            return

        if self._phase == self._PHASE_GRASP_CLOSE:
            if self._gripper_deadline_s is not None and time.monotonic() >= self._gripper_deadline_s:
                self._set_phase(self._PHASE_MOVE_MIDPOINT)
            return

        if self._phase == self._PHASE_MOVE_MIDPOINT:
            if not self._waypoint_sent:
                self._publish_joint_waypoint(self.midpoint_joint_pos)
                self._waypoint_sent = True
            if self._joint_target_reached(self.midpoint_joint_pos):
                self._set_phase(self._PHASE_MOVE_PLACE)
            elif self._state_timed_out():
                self.get_logger().error('Timed out while moving to midpoint.')
                self._set_phase(self._PHASE_DONE)
            return

        if self._phase == self._PHASE_MOVE_PLACE:
            if not self._waypoint_sent:
                self._publish_joint_waypoint(self.place_joint_pos)
                self._waypoint_sent = True
            if self._joint_target_reached(self.place_joint_pos):
                self._send_gripper_goal(self.gripper_open_position)
                self._gripper_deadline_s = time.monotonic() + self.gripper_wait_s
                self._set_phase(self._PHASE_DROP_OPEN)
            elif self._state_timed_out():
                self.get_logger().error('Timed out while moving to place waypoint.')
                self._set_phase(self._PHASE_DONE)
            return

        if self._phase == self._PHASE_DROP_OPEN:
            if self._gripper_deadline_s is not None and time.monotonic() >= self._gripper_deadline_s:
                self._set_phase(self._PHASE_RETURN_MIDPOINT)
            return

        if self._phase == self._PHASE_RETURN_MIDPOINT:
            if not self._waypoint_sent:
                self._publish_joint_waypoint(self.midpoint_joint_pos)
                self._waypoint_sent = True
            if self._joint_target_reached(self.midpoint_joint_pos):
                self._set_phase(self._PHASE_MOVE_HOME)
            elif self._state_timed_out():
                self.get_logger().error('Timed out while returning to midpoint after drop.')
                self._set_phase(self._PHASE_DONE)
            return

        if self._phase == self._PHASE_MOVE_HOME:
            if not self._waypoint_sent:
                self._publish_joint_waypoint(self.home_joint_pos)
                self._waypoint_sent = True
            if self._joint_target_reached(self.home_joint_pos):
                self.get_logger().info('Pick-and-place sequence complete. Reached home pose; waiting for new target.')
                self._reset_for_next_target()
            elif self._state_timed_out():
                self.get_logger().error('Timed out while moving to home pose.')
                self._set_phase(self._PHASE_DONE)
            return

def main() -> None:
    rclpy.init()
    node = ReachPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
