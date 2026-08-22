#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
source "${ROOT}/scripts/select_cam.sh"
HAND="${ROOT}/components/hand_keypoints_ros"
CAM="cam_4"
CAM_OVERRIDES=()
NODE_NAME="hand_detection_node"
PARAM_FILE_ARGS=()
if [[ "${1:-}" == "help" || "${1:-}" == "--help" ]]; then
  echo "usage: bash scripts/run_hand_cam4.sh [cam_1|cam_2|cam_3|cam_4|flir]" >&2
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
  print_synced_cameras
  exit 0
fi
if is_camera_selector "${1:-}"; then
  apply_ingress_cam "$1"
  shift
else
  apply_ingress_cam "${CAM}"
fi
NODE_NAME="hand_detection_node_${CAM}"
CAM_OVERRIDES=(
  -p "camera:=${CAM}"
  -p "keypoints_topic:=/perception/${CAM}/hand/keypoints"
  -p "overlay_topic:=/perception/${CAM}/hand/overlay/compressed"
  -p "target_pose_topic:=/perception/${CAM}/hand/target_pose"
  -p "health_topic:=/perception/${CAM}/hand/health"
  -p "diagnostics_topic:=/perception/${CAM}/hand/diagnostics"
)
if [[ -z "${CAM:-}" || "${CAM}" == "cam_4" ]]; then
  PARAM_FILE_ARGS=(--params-file "${ROOT}/config/cam4_depth_to_color.yaml")
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node" \
  --ros-args -r "__node:=${NODE_NAME}" \
  "${PARAM_FILE_ARGS[@]}" \
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
