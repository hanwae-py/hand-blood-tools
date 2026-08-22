#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
source "${ROOT}/scripts/select_cam.sh"
COORD="${ROOT}/components/coordinator_ws"
CAM="cam_4"
CAM_OVERRIDES=()
NODE_NAME="blood_detection_node"
PARAM_FILE_ARGS=()
if [[ "${1:-}" == "help" || "${1:-}" == "--help" ]]; then
  echo "usage: bash scripts/run_blood_cam4.sh [cam_1|cam_2|cam_3|cam_4|flir]" >&2
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
NODE_NAME="blood_detection_node_${CAM}"
CAM_OVERRIDES=(
  -p "camera:=${CAM}"
  -p "mask_topic:=/perception/${CAM}/blood/mask"
  -p "overlay_topic:=/perception/${CAM}/blood/overlay/compressed"
  -p "semantics_topic:=/perception/${CAM}/blood/semantics"
  -p "health_topic:=/perception/${CAM}/blood/health"
  -p "diagnostics_topic:=/perception/${CAM}/blood/diagnostics"
)
if [[ -z "${CAM:-}" || "${CAM}" == "cam_4" ]]; then
  PARAM_FILE_ARGS=(--params-file "${ROOT}/config/cam4_depth_to_color.yaml")
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${COORD}/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RFDETR_PYTHON}" -m surgical_task_coordinator.blood_detection_node \
  --ros-args -r "__node:=${NODE_NAME}" \
  "${PARAM_FILE_ARGS[@]}" \
  -p color_topic:="${COLOR_TOPIC}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p color_camera_info_topic:="${COLOR_CAMERA_INFO_TOPIC}" \
  -p depth_camera_info_topic:="${DEPTH_CAMERA_INFO_TOPIC}" \
  -p checkpoint:="${BLOOD_CHECKPOINT}" \
  -p confidence_threshold:=0.5 -p optimize:=true \
  -p require_depth:=false \
  "${CAM_OVERRIDES[@]}" \
  "$@"
