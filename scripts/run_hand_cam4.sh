#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
HAND="${ROOT}/components/hand_keypoints_ros"
COLOR_CAMERA_INFO_TOPIC="${COLOR_CAMERA_INFO_TOPIC:-/synced/cam_4/color/camera_info}"
DEPTH_CAMERA_INFO_TOPIC="${DEPTH_CAMERA_INFO_TOPIC:-/synced/cam_4/depth/camera_info}"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node" \
  --ros-args -r __node:=hand_detection_node \
  --params-file "${ROOT}/config/cam4_depth_to_color.yaml" \
  -p color_topic:="${COLOR_TOPIC}" \
  -p color_transport:="${COLOR_TRANSPORT}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p depth_transport:="${DEPTH_TRANSPORT}" \
  -p camera_info_topic:="${COLOR_CAMERA_INFO_TOPIC}" \
  -p depth_camera_info_topic:="${DEPTH_CAMERA_INFO_TOPIC}" \
  -p depth_source:=real -p depth_alignment_validated:=false \
  -p publish_overlay:=true "$@"
