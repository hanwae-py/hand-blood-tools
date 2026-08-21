#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
HAND="${ROOT}/components/hand_keypoints_ros"
COLOR_CAMERA_INFO_TOPIC="${COLOR_CAMERA_INFO_TOPIC:-/synced/cam_4/color/camera_info}"
DEPTH_CAMERA_INFO_TOPIC="${DEPTH_CAMERA_INFO_TOPIC:-/synced/cam_4/depth/camera_info}"
CAM_OVERRIDES=()
NODE_NAME="hand_detection_node"
if [[ "${1:-}" =~ ^cam_[0-9]+$ ]]; then
  CAM="$1"
  shift
  COLOR_TOPIC="/synced/${CAM}/color/image_raw/compressed"
  COLOR_CAMERA_INFO_TOPIC="/synced/${CAM}/color/camera_info"
  DEPTH_TOPIC="/synced/${CAM}/depth/image_rect_raw/compressedDepth"
  DEPTH_CAMERA_INFO_TOPIC="/synced/${CAM}/depth/camera_info"
  NODE_NAME="hand_detection_node_${CAM}"
  CAM_OVERRIDES=(
    -p "keypoints_topic:=/perception/${CAM}/hand/keypoints"
    -p "overlay_topic:=/perception/${CAM}/hand/overlay/compressed"
    -p "target_pose_topic:=/perception/${CAM}/hand/target_pose"
    -p "health_topic:=/perception/${CAM}/hand/health"
    -p "diagnostics_topic:=/perception/${CAM}/hand/diagnostics"
  )
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node" \
  --ros-args -r "__node:=${NODE_NAME}" \
  --params-file "${ROOT}/config/cam4_depth_to_color.yaml" \
  -p color_topic:="${COLOR_TOPIC}" \
  -p color_transport:="${COLOR_TRANSPORT}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p depth_transport:="${DEPTH_TRANSPORT}" \
  -p camera_info_topic:="${COLOR_CAMERA_INFO_TOPIC}" \
  -p depth_camera_info_topic:="${DEPTH_CAMERA_INFO_TOPIC}" \
  -p depth_source:=real -p depth_alignment_validated:=false \
  -p publish_overlay:=true \
  "${CAM_OVERRIDES[@]}" \
  "$@"
