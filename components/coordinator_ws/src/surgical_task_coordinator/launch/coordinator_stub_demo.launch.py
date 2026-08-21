"""Whole take-turn flow, runnable on this PC today with NO real detector.

Brings up the coordinator, a fake robot, and THREE stub detectors standing in
for the tool / hand / blood algorithms. Nothing here loads a real model, so
it starts in seconds and needs no GPU -- the point is to verify the
sequencing, the lifecycle switching and the abort path.

    ros2 launch surgical_task_coordinator coordinator_stub_demo.launch.py

Then, in a second terminal:

    # watch whose turn it is
    ros2 topic echo /surgery/task/state

    # tool -> grasp -> hand -> handover
    ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'REQUEST_TOOL'}"

    # blood -> suction
    ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'SUCK_BLOOD'}"

    # stop whatever is running
    ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'ABORT'}"

    Keep the -w 1: without it the publisher can exit before DDS discovery has
    matched the coordinator's subscription, and the command is silently lost.

    # confirm only one detector is ever active
    ros2 lifecycle get /tool_detection_node
    ros2 lifecycle get /hand_detection_node
    ros2 lifecycle get /blood_detection_node

To swap the stub hand detector for the REAL one, set use_stub_hand:=false and
separately launch hand_keypoint_ros (see that repo's launch files). The real
node must be launched with autostart:=false so the coordinator, not the node
itself, decides when it is allowed to run.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _stub(node_name, result_topic, xyz, condition=None):
    return Node(
        package='surgical_task_coordinator',
        executable='stub_detector',
        name=node_name,
        output='screen',
        condition=condition,
        parameters=[{
            'result_topic': result_topic,
            'target_xyz': xyz,
            'detection_delay_sec': 1.5,
            'fake_model_load_sec': 2.0,
        }],
    )


def generate_launch_description():
    use_stub_hand = LaunchConfiguration('use_stub_hand')
    release_gpu = LaunchConfiguration('release_gpu_between_tasks')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_stub_hand', default_value='true',
            description='false: do not start the stub hand detector, because you are '
                        'running the real hand_keypoint_ros node (with autostart:=false).'),
        DeclareLaunchArgument(
            'release_gpu_between_tasks', default_value='true',
            description='true: clean detectors to UNCONFIGURED between turns so their '
                        'models leave VRAM. Set false once nvidia-smi confirms all three '
                        'models co-fit on the 12 GB 3060 -- turns then switch instantly.'),

        Node(
            package='surgical_task_coordinator',
            executable='task_coordinator',
            name='task_coordinator',
            output='screen',
            parameters=[{'release_gpu_between_tasks': release_gpu}],
        ),
        Node(
            package='surgical_task_coordinator',
            executable='fake_robot_node',
            name='fake_robot_node',
            output='screen',
            parameters=[{'motion_duration_sec': 3.0}],
        ),

        _stub('tool_detection_node', '/perception/cam_4/tool/target_pose',
              [0.10, 0.05, 0.40]),
        _stub('hand_detection_node', '/perception/cam_4/hand/target_pose',
              [-0.08, 0.02, 0.55], condition=IfCondition(use_stub_hand)),
        _stub('blood_detection_node', '/perception/cam_4/blood/target_pose',
              [0.02, 0.12, 0.45]),
    ])
