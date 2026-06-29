import os
import time
from typing import List, Optional

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger


class PolicyNodeReach(Node):
    """Runs a reach policy at fixed rate from live joint states."""

    def __init__(self) -> None:
        super().__init__('policy_node_reach')

        self.declare_parameter('checkpoint_path', '/home/nvidia/Documents/qarm_research/qarm_rl/resource/policy.pt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('episode_timeout_s', 5.0)
        self.declare_parameter('action_scale', 0.5)
        self.declare_parameter('min_bridge_action_dim', 6)
        self.declare_parameter('target_topic', '/policy/target_pose_xyz_xyzw')

        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('default_joint_pos', [-1.309, -0.8727, 2.00713, 0.0, 1.5708, -0.2618])
        self.declare_parameter('target_pose_xyz_xyzw', [-0.113, -0.53464, 0.20487, 0.70710678,0.70710678, 0.0, 0.0])
        self.declare_parameter('start_on_launch', False)

        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.device = self._resolve_device(self.get_parameter('device').value)
        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.episode_timeout_s = float(self.get_parameter('episode_timeout_s').value)
        self.action_scale = float(self.get_parameter('action_scale').value)
        self.min_bridge_action_dim = int(self.get_parameter('min_bridge_action_dim').value)
        self.target_topic = self.get_parameter('target_topic').get_parameter_value().string_value

        self.joint_names: List[str] = list(self.get_parameter('joint_names').value)
        self.default_joint_pos = np.array(self.get_parameter('default_joint_pos').value, dtype=np.float64)
        target_pose = np.array(self.get_parameter('target_pose_xyz_xyzw').value, dtype=np.float64)

        if len(self.joint_names) < 1:
            raise ValueError('joint_names must contain at least one joint')
        if len(self.default_joint_pos) != len(self.joint_names):
            raise ValueError('default_joint_pos length must match joint_names length')
        if target_pose.shape[0] != 7:
            raise ValueError('target_pose_xyz_xyzw must contain 7 values [x, y, z, qx, qy, qz, qw]')
        if self.control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be > 0')

        self.target_command = np.array(target_pose, dtype=np.float64)

        self.policy = self._load_policy(self.checkpoint_path)
        self.act_dim = self._infer_action_dim(len(self.joint_names), self.target_command.shape[0])

        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_joint_vel: Optional[np.ndarray] = None
        self._joint_sum_pos: Optional[np.ndarray] = None
        self._joint_sum_vel: Optional[np.ndarray] = None
        self._joint_sample_count = 0
        self._last_action = np.zeros(self.act_dim, dtype=np.float64)
        self._episode_running = False
        self._episode_start_time = 0.0

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._on_joint_states,
            20,
        )
        latched_target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            self.target_topic,
            self._on_target_pose,
            latched_target_qos,
        )
        self.raw_action_pub = self.create_publisher(Float64MultiArray, '/policy/joint_actions', 20)
        self.abs_arm_target_pub = self.create_publisher(Float64MultiArray, '/policy/absolute_joint_targets', 20)

        self.start_srv = self.create_service(Trigger, '/policy/start_episode', self._on_start_episode)
        self.stop_srv = self.create_service(SetBool, '/policy/stop_episode', self._on_stop_episode)

        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self._control_step)

        self.get_logger().info(
            f'Policy reach node ready on device={self.device} with act_dim={self.act_dim}. '
            f'Publishing /policy/joint_actions for command_bridge. '
            f'Subscribed target topic: {self.target_topic}'
        )
        if self.get_parameter('start_on_launch').value:
            self._start_episode()

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but unavailable. Falling back to CPU.')
            return torch.device('cpu')
        return torch.device(requested)

    def _load_policy(self, checkpoint_path: str) -> torch.nn.Module:
        if not checkpoint_path:
            raise ValueError('checkpoint_path parameter is required')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'checkpoint path does not exist: {checkpoint_path}')

        try:
            policy = torch.jit.load(checkpoint_path, map_location=self.device)
            policy.eval()
            self.get_logger().info('Loaded TorchScript policy via torch.jit.load().')
            return policy
        except Exception:
            pass

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(self.device)
            model.eval()
            self.get_logger().info('Loaded nn.Module checkpoint directly.')
            return model

        if not isinstance(checkpoint, dict):
            raise RuntimeError('Unsupported checkpoint type. Expected TorchScript, nn.Module, or dict state.')

        actor_state = checkpoint.get('actor_state_dict', checkpoint)
        if not isinstance(actor_state, dict):
            raise RuntimeError('Checkpoint does not contain actor_state_dict or direct actor state dict.')

        weight_keys = sorted(
            [k for k in actor_state.keys() if isinstance(k, str) and k.startswith('mlp.') and k.endswith('.weight')],
            key=lambda k: int(k.split('.')[1]),
        )
        if not weight_keys:
            raise RuntimeError('Could not find mlp.*.weight keys in checkpoint actor weights.')

        layer_sizes = [int(actor_state[weight_keys[0]].shape[1])]
        for key in weight_keys:
            layer_sizes.append(int(actor_state[key].shape[0]))

        layers: list[torch.nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.ELU())
        actor = torch.nn.Sequential(*layers).to(self.device)

        remapped = {
            key.replace('mlp.', ''): value
            for key, value in actor_state.items()
            if isinstance(key, str) and key.startswith('mlp.')
        }
        actor.load_state_dict(remapped, strict=True)
        actor.eval()
        self.get_logger().info(f'Loaded actor from mlp.* weights with layer sizes {layer_sizes}.')
        return actor

    def _infer_action_dim(self, num_joints: int, command_dim: int) -> int:
        obs_dim = (num_joints * 3) + command_dim
        with torch.no_grad():
            probe = torch.zeros(1, obs_dim, dtype=torch.float32, device=self.device)
            out = self.policy(probe)
            if out.ndim != 2 or out.shape[0] != 1:
                raise RuntimeError(f'Unexpected policy output shape from probe: {tuple(out.shape)}')
            return int(out.shape[1])

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

    def _on_target_pose(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 7:
            self.get_logger().warn('Expected 7 target pose values [x, y, z, qx, qy, qz, qw]. Ignoring message.')
            return
        self.target_command = np.asarray(msg.data, dtype=np.float64)
        self.get_logger().info(f'Updated target pose from topic: {np.round(self.target_command, 5)}')

    def _build_obs(self, q: np.ndarray, qdot: np.ndarray, command: np.ndarray) -> np.ndarray:
        joint_pos_rel = q - self.default_joint_pos
        obs = np.concatenate([
            joint_pos_rel,
            qdot,
            command,
            self._last_action,
        ]).astype(np.float64)
        return obs

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
            action = self.policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float64)
        if action.shape[0] != self.act_dim:
            raise RuntimeError(f'Action dimension mismatch: got {action.shape[0]}, expected {self.act_dim}')
        return action

    def _publish_actions(self, raw: np.ndarray) -> None:
        bridge_raw = raw
        if bridge_raw.shape[0] < self.min_bridge_action_dim:
            padded = np.zeros(self.min_bridge_action_dim, dtype=np.float64)
            padded[:bridge_raw.shape[0]] = bridge_raw
            bridge_raw = padded

        raw_msg = Float64MultiArray()
        raw_msg.data = bridge_raw.tolist()
        self.raw_action_pub.publish(raw_msg)

        if self.default_joint_pos.shape[0] >= 6 and raw.shape[0] >= 6:
            abs_targets = self.default_joint_pos[:6] + raw[:6] * self.action_scale
            # print(f'Publishing absolute joint targets: {abs_targets}')
            abs_msg = Float64MultiArray()
            abs_msg.data = abs_targets.tolist()
            self.abs_arm_target_pub.publish(abs_msg)

    def _on_start_episode(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if self._episode_running:
            resp.success = False
            resp.message = 'Episode already running.'
            return resp
        self._start_episode()
        resp.success = True
        resp.message = 'Episode started.'
        return resp

    def _on_stop_episode(self, req: SetBool.Request, resp: SetBool.Response) -> SetBool.Response:
        if req.data:
            self._episode_running = False
            resp.success = True
            resp.message = 'Episode stopped.'
        else:
            resp.success = False
            resp.message = 'Send true to stop the episode.'
        return resp

    def _start_episode(self) -> None:
        self._last_action[:] = 0.0
        self._episode_start_time = time.time()
        self._episode_running = True
        self.get_logger().info('Reach episode started.')

    def _control_step(self) -> None:
        if not self._episode_running:
            return

        if self._joint_sample_count > 0 and self._joint_sum_pos is not None and self._joint_sum_vel is not None:
            inv_count = 1.0 / float(self._joint_sample_count)
            self._latest_joint_pos = self._joint_sum_pos * inv_count
            self._latest_joint_vel = self._joint_sum_vel * inv_count
            self._joint_sum_pos.fill(0.0)
            self._joint_sum_vel.fill(0.0)
            self._joint_sample_count = 0

        if self._latest_joint_pos is None or self._latest_joint_vel is None:
            elapsed = time.time() - self._episode_start_time
            if elapsed > 2.0:
                self.get_logger().error('No /joint_states received within 2.0 s. Stopping episode.')
                self._episode_running = False
            return

        elapsed = time.time() - self._episode_start_time
        if elapsed >= self.episode_timeout_s:
            self.get_logger().info('Episode timeout reached.')
            self._episode_running = False
            return

        try:
            obs = self._build_obs(self._latest_joint_pos, self._latest_joint_vel, self.target_command)
            raw = self._infer(obs)
            print('-----------------------------------------')
            print(f'observations - pos {obs[:6]}')
            print(f'observations - vel {obs[6:12]}')
            print(f'observations - cmd {obs[12:19]}')
            print(f'observations - prev {obs[19:]}')
            print(f'action {raw}')
        except Exception as exc:
            self.get_logger().error(f'Inference step failed: {exc}')
            self._episode_running = False
            return

        self._publish_actions(raw)
        self._last_action[:] = raw
        # self._last_action[:] = 0.0



def main() -> None:
    rclpy.init()
    node = PolicyNodeReach()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()