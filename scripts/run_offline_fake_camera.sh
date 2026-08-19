#!/usr/bin/env bash
# Starts the raw RGB + H5 depth replay used by Hand's offline test mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
HAND="${ROOT}/components/hand_keypoints_ros"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/fake_camera_publisher" \
  --ros-args \
  -p rgb_path:="${OFFLINE_RGB_PATH}" \
  -p depth_h5_path:="${OFFLINE_DEPTH_H5_PATH}" \
  -p calib_path:="${OFFLINE_CALIB_PATH}" \
  -p cam_key:="${OFFLINE_CAM_KEY}" \
  -p rate_hz:="${OFFLINE_RATE_HZ}" \
  -p preload_depth:=false -p loop:=true
