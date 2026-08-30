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
if [[ "${CAM}" == "flir" ]]; then
  CAM_OVERRIDES+=(
    -p reject_low_quality_input:=true
    -p minimum_gray_p99:=20.0
    -p minimum_gray_dynamic_range:=12.0
  )
fi
if [[ -z "${CAM:-}" || "${CAM}" == "cam_4" ]]; then
  PARAM_FILE_ARGS=(--params-file "${ROOT}/config/cam4_depth_to_color.yaml")
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${COORD}/install/setup.bash"
set -u
BLOOD_ROOT="${ROOT}/components/blood_detection"
BLOOD_CHECKPOINT="${BLOOD_CHECKPOINT:-${BLOOD_ROOT}/pretrained/blood_detection_full_all.pth}"
BLOOD_CUTIE_CHECKPOINT="${BLOOD_CUTIE_CHECKPOINT:-${BLOOD_ROOT}/pretrained/cutie_blood_full_all.pth}"
if [[ -z "${BLOOD_PYTHON:-}" ]]; then
  echo "BLOOD_PYTHON is not configured. Set it in config/system.env (not RFDETR_PYTHON)." >&2
  exit 1
fi
export PYTHONPATH="${BLOOD_ROOT}:${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${BLOOD_PYTHON}" -m surgical_task_coordinator.blood_detection_node \
  --ros-args -r "__node:=${NODE_NAME}" \
  "${PARAM_FILE_ARGS[@]}" \
  -p color_topic:="${COLOR_TOPIC}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p color_camera_info_topic:="${COLOR_CAMERA_INFO_TOPIC}" \
  -p depth_camera_info_topic:="${DEPTH_CAMERA_INFO_TOPIC}" \
  -p checkpoint:="${BLOOD_CHECKPOINT}" \
  -p cutie_checkpoint:="${BLOOD_CUTIE_CHECKPOINT}" \
  -p confidence_threshold:=0.5 \
  -p redetect_interval:=1 \
  -p require_depth:=false \
  "${CAM_OVERRIDES[@]}" \
  "$@"
