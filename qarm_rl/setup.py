from setuptools import find_packages, setup

package_name = 'qarm_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', [
            'resource/stream.py',
            'resource/policy.pt',
            'resource/vision_server.rt-linux_qcar2',
        ]),
        ('share/' + package_name + '/config', [
            'config/policy_node_reach.yaml',
            'config/command_bridge_reach.yaml',
            'config/reach_planner_node.yaml',
            'config/vision_target_stream_node.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/reach_full_stack.launch.py',
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'torch'],
    zip_safe=True,
    maintainer='academic',
    maintainer_email='zinan.cen@quanser.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'policy_node_reach = qarm_rl.policy_node_reach:main',
            'command_bridge_reach = qarm_rl.command_bridge_reach:main',
            'reach_planner_node = qarm_rl.reach_planner_node:main',
            'vision_target_stream_node = qarm_rl.vision_target_stream_node:main',
        ],
    },
)
