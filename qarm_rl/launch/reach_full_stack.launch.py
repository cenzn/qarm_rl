import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _stop_vision_server_cb(context):
    subprocess.run(
        'quarc_run -q -t tcpip://localhost:17000 vision_server',
        shell=True,
        capture_output=True,
    )


def generate_launch_description() -> LaunchDescription:
    default_policy_params = PathJoinSubstitution([
        FindPackageShare('qarm_rl'),
        'config',
        'policy_node_reach.yaml',
    ])
    default_bridge_params = PathJoinSubstitution([
        FindPackageShare('qarm_rl'),
        'config',
        'command_bridge_reach.yaml',
    ])
    default_planner_params = PathJoinSubstitution([
        FindPackageShare('qarm_rl'),
        'config',
        'reach_planner_node.yaml',
    ])
    default_vision_params = PathJoinSubstitution([
        FindPackageShare('qarm_rl'),
        'config',
        'vision_target_stream_node.yaml',
    ])
    vision_server_rt_executable = PathJoinSubstitution([
        FindPackageShare('qarm_rl'),
        'resource',
        'vision_server.rt-linux_qcar2',
    ])

    policy_params_arg = DeclareLaunchArgument(
        'policy_params_file',
        default_value=default_policy_params,
        description='Path to ROS2 parameters YAML for policy_node_reach.',
    )
    bridge_params_arg = DeclareLaunchArgument(
        'bridge_params_file',
        default_value=default_bridge_params,
        description='Path to ROS2 parameters YAML for command_bridge_reach.',
    )
    planner_params_arg = DeclareLaunchArgument(
        'planner_params_file',
        default_value=default_planner_params,
        description='Path to ROS2 parameters YAML for reach_planner_node.',
    )
    vision_params_arg = DeclareLaunchArgument(
        'vision_params_file',
        default_value=default_vision_params,
        description='Path to ROS2 parameters YAML for vision_target_stream_node.',
    )

    policy_node_reach = Node(
        package='qarm_rl',
        executable='policy_node_reach',
        name='policy_node_reach',
        output='screen',
        parameters=[LaunchConfiguration('policy_params_file')],
    )

    command_bridge_reach = Node(
        package='qarm_rl',
        executable='command_bridge_reach',
        name='command_bridge_reach',
        output='screen',
        parameters=[LaunchConfiguration('bridge_params_file')],
    )

    reach_planner_node = Node(
        package='qarm_rl',
        executable='reach_planner_node',
        name='reach_planner_node',
        output='screen',
        parameters=[LaunchConfiguration('planner_params_file')],
    )

    vision_target_stream_node = Node(
        package='qarm_rl',
        executable='vision_target_stream_node',
        name='vision_target_stream_node',
        output='screen',
        parameters=[LaunchConfiguration('vision_params_file')],
    )

    rt_model_start = ExecuteProcess(
        cmd=[
            'quarc_run',
            '-r -t tcpip://localhost:17000',
            vision_server_rt_executable,
            '-d %d -uri tcpip://%m:17888',
        ],
        name='VisionServerModelStart',
        shell=True,
    )

    start_nodes_after_rt_model = RegisterEventHandler(
        OnProcessStart(
            target_action=rt_model_start,
            on_start=[
                LogInfo(msg='VisionServerModelStart started. Waiting 2 sec before starting reach stack nodes.'),
                TimerAction(
                    period=4.0,
                    actions=[
                        policy_node_reach,
                        command_bridge_reach,
                        reach_planner_node,
                        vision_target_stream_node,
                    ],
                ),
            ],
        )
    )

    stop_rt_model_on_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=vision_target_stream_node,
            on_exit=[
                OpaqueFunction(function=_stop_vision_server_cb),
                LogInfo(msg='vision_target_stream_node exiting, so stopping vision_server model as well.'),
            ],
        )
    )

    return LaunchDescription([
        policy_params_arg,
        bridge_params_arg,
        planner_params_arg,
        vision_params_arg,
        rt_model_start,
        start_nodes_after_rt_model,
        stop_rt_model_on_exit,
    ])
