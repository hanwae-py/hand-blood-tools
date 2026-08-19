#!/usr/bin/env bash
# Start only Hand detection. A compressed RGB-D camera/MCAP publisher must already be running.
set -euo pipefail
HAND_REPO=/home/hanwae/hand_keypoints_ros
HAND_VENV=/home/hanwae/hand_keypoints_ros_ws/.venv
set +u
source /opt/ros/jazzy/setup.bash
source "${HAND_VENV}/bin/activate"
source "${HAND_REPO}/ros2_ws/install/setup.bash"
set -u
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
exec "${HAND_VENV}/bin/python"   "${HAND_REPO}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node"   --ros-args -r __node:=hand_detection_node   -p autostart:=true   -p depth_source:=real   -p color_topic:=/synced/cam_4/color/image_raw/compressed   -p color_transport:=compressed   -p depth_topic:=/synced/cam_4/depth/image_rect_raw/compressedDepth   -p depth_transport:=compressed_depth   -p camera_info_topic:=/synced/cam_4/color/camera_info   -p depth_alignment_validated:=false   -p publish_overlay:=true "$@"
