#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
source "${ROOT}/scripts/select_cam.sh"
TOOL="${ROOT}/components/tool_runtime_v1_6"
CAM="cam_4"
CAM_OVERRIDES=()
PARAM_FILE_ARGS=()

configure_tool_camera() {
  apply_ingress_cam "$1"
  case "${CAM}" in
    cam_3|cam_4) : ;;
    *)
      echo "Tool pose is configured only for cam_3 or cam_4, got: ${CAM}" >&2
      return 2
      ;;
  esac
  local camera_profile="${CAM/_/}"
  local parameter_file="${TOOL}/ros2_ws/src/pnu_surgical_perception/config/${camera_profile}_live_native_pose.yaml"
  if [[ ! -f "${parameter_file}" ]]; then
    echo "Missing Tool pose parameter file: ${parameter_file}" >&2
    return 2
  fi
  PARAM_FILE_ARGS=(--params-file "${parameter_file}")
  CAM_OVERRIDES=(
    -r "__node:=native_depth_tool_pose_${CAM}"
    -p "camera:=${CAM}"
    -p "view:=${CAM}"
    -p "pose_topic:=/perception/${CAM}/tool/poses"
    -p "observation_topic:=/perception/${CAM}/tool/observations"
    -p "overlay_topic:=/perception/${CAM}/tool/overlay/compressed"
    -p "pose_overlay_topic:=/perception/${CAM}/tool/pose_overlay/compressed"
    -p "diagnostics_topic:=/perception/${CAM}/tool/diagnostics"
    -p "health_topic:=/perception/${CAM}/tool/health"
  )
}

if [[ "${1:-}" == "help" || "${1:-}" == "--help" ]]; then
  echo "usage: bash scripts/run_tool_v16.sh [cam_3|cam_4]" >&2
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
  print_synced_cameras
  exit 0
fi
if is_camera_selector "${1:-}"; then
  configure_tool_camera "$1"
  shift
else
  configure_tool_camera "${CAM}"
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="${TOOL}/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RFDETR_PYTHON}" \
  "${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/native_depth_tool_pose" \
  --ros-args \
  "${PARAM_FILE_ARGS[@]}" \
  -p "algorithm_python_path:=${TOOL}/algorithm/src" \
  -p "checkpoint:=${TOOL_CHECKPOINT}" \
  -p "ontology:=${TOOL}/algorithm/model/ontology.json" \
  -p "rgb_topic:=${COLOR_TOPIC}" \
  -p "color_camera_info_topic:=${COLOR_CAMERA_INFO_TOPIC}" \
  -p "depth_topic:=${DEPTH_TOPIC}" \
  -p "depth_camera_info_topic:=${DEPTH_CAMERA_INFO_TOPIC}" \
  -p "extrinsics_topic:=${EXTRINSICS_TOPIC}" \
  -p "require_depth:=true" \
  -p "require_extrinsics_topic:=true" \
  "${CAM_OVERRIDES[@]}" \
  "$@"
