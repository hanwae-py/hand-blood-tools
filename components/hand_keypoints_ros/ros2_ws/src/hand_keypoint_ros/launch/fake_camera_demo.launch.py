"""Demo launch file: fake_camera_publisher (replaying this repo's example
clip) + hand_detection_node, wired together with matching topic names, so
the whole pipeline can be exercised without real camera hardware.

Usage (defaults to the real-depth example clip bundled in data/,
depth_source:=auto on the detection node):
    ros2 launch hand_keypoint_ros fake_camera_demo.launch.py

Usage (force the monocular-depth fallback on the same clip, even though
real depth IS being published — mirrors run_hand_keypoints.py's
--force-mono-depth. NOTE: ROS2 launch args can't be set to an empty
string via name:="", hence a dedicated depth_source arg rather than
clearing depth_h5_path):
    ros2 launch hand_keypoint_ros fake_camera_demo.launch.py depth_source:=mono

Usage (robot handoff mode):
    ros2 launch hand_keypoint_ros fake_camera_demo.launch.py robot_position:=top-left

Then, in another terminal:
    ros2 topic hz /hand/keypoints
    ros2 topic echo /hand/keypoints --once
    ros2 run rqt_image_view rqt_image_view   # view /hand/overlay_image (needs a display)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path

# Defaults point at this repo's bundled example clip (see hand_keypoints/data/).
# Override with -p rgb_path:=... etc. for a different clip, or point these
# at wherever this container mounts the repo.
_native_data_dir = Path(__file__).resolve().parents[4] / 'data'
_DEFAULT_DATA_DIR = str(
    _native_data_dir if _native_data_dir.is_dir() else Path('/repo/data')
)


def generate_launch_description():
    rgb_path = LaunchConfiguration('rgb_path')
    calib_path = LaunchConfiguration('calib_path')
    depth_h5_path = LaunchConfiguration('depth_h5_path')
    cam_key = LaunchConfiguration('cam_key')
    rate_hz = LaunchConfiguration('rate_hz')
    robot_position = LaunchConfiguration('robot_position')
    cpu_only = LaunchConfiguration('cpu_only')
    depth_source = LaunchConfiguration('depth_source')
    loop = LaunchConfiguration('loop')
    preload_depth = LaunchConfiguration('preload_depth')

    return LaunchDescription([
        DeclareLaunchArgument('rgb_path', default_value=f'{_DEFAULT_DATA_DIR}/rgb/gnu_0704_rgb_03.avi'),
        DeclareLaunchArgument('calib_path', default_value=f'{_DEFAULT_DATA_DIR}/calibration/gnu_0704_calibration_03.json'),
        DeclareLaunchArgument('depth_h5_path', default_value=f'{_DEFAULT_DATA_DIR}/depth_raw/gnu_0704_depth_raw_03.h5'),
        DeclareLaunchArgument('cam_key', default_value='cam_4'),
        DeclareLaunchArgument('rate_hz', default_value='15.0'),
        DeclareLaunchArgument('robot_position', default_value=''),
        DeclareLaunchArgument('cpu_only', default_value='false'),
        DeclareLaunchArgument('depth_source', default_value='auto',
                               description='auto | real | mono -- see hand_detection_node docstring'),
        DeclareLaunchArgument('loop', default_value='true',
                               description='loop the clip at end-of-stream. loop:=false gives a quick, '
                                            'deterministic one-shot run instead.'),
        DeclareLaunchArgument('preload_depth', default_value='true',
                              description='true preloads recorded HDF5 depth for fast replay; false '
                                          'reads one depth frame per tick with low RAM usage'),

        Node(
            package='hand_keypoint_ros',
            executable='fake_camera_publisher',
            name='fake_camera_publisher',
            parameters=[{
                'rgb_path': rgb_path,
                'calib_path': calib_path,
                'depth_h5_path': depth_h5_path,
                'cam_key': cam_key,
                'rate_hz': rate_hz,
                'loop': loop,
                'preload_depth': preload_depth,
            }],
            output='screen',
        ),
        Node(
            package='hand_keypoint_ros',
            executable='hand_detection_node',
            name='hand_detection_node',
            parameters=[{
                'depth_source': depth_source,
                'robot_position': robot_position,
                'cpu_only': cpu_only,
            }],
            output='screen',
        ),
    ])
