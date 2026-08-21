#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
COORD="${ROOT}/components/coordinator_ws"
COLOR_CAMERA_INFO_TOPIC="${COLOR_CAMERA_INFO_TOPIC:-/synced/cam_4/color/camera_info}"
DEPTH_CAMERA_INFO_TOPIC="${DEPTH_CAMERA_INFO_TOPIC:-/synced/cam_4/depth/camera_info}"
CAM_OVERRIDES=()
NODE_NAME="blood_detection_node"
if [[ "${1:-}" =~ ^cam_[0-9]+$ ]]; then
  CAM="$1"
  shift
  COLOR_TOPIC="/synced/${CAM}/color/image_raw/compressed"
  COLOR_CAMERA_INFO_TOPIC="/synced/${CAM}/color/camera_info"
  DEPTH_TOPIC="/synced/${CAM}/depth/image_rect_raw/compressedDepth"
  DEPTH_CAMERA_INFO_TOPIC="/synced/${CAM}/depth/camera_info"
  NODE_NAME="blood_detection_node_${CAM}"
  CAM_OVERRIDES=(
    -p "mask_topic:=/perception/${CAM}/blood/mask"
    -p "overlay_topic:=/perception/${CAM}/blood/overlay/compressed"
    -p "semantics_topic:=/perception/${CAM}/blood/semantics"
    -p "health_topic:=/perception/${CAM}/blood/health"
    -p "diagnostics_topic:=/perception/${CAM}/blood/diagnostics"
  )
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${COORD}/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RFDETR_PYTHON}" -m surgical_task_coordinator.blood_detection_node \
  --ros-args -r "__node:=${NODE_NAME}" \
  --params-file "${ROOT}/config/cam4_depth_to_color.yaml" \
  -p color_topic:="${COLOR_TOPIC}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p color_camera_info_topic:="${COLOR_CAMERA_INFO_TOPIC}" \
  -p depth_camera_info_topic:="${DEPTH_CAMERA_INFO_TOPIC}" \
  -p checkpoint:="${BLOOD_CHECKPOINT}" \
  -p confidence_threshold:=0.5 -p optimize:=true \
  -p require_depth:=false \
  "${CAM_OVERRIDES[@]}" \
  "$@"
