#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
TOOL="${ROOT}/components/tool_runtime_v1_6"
CAM_OVERRIDES=()
if [[ "${1:-}" =~ ^cam_[0-9]+$ ]]; then
  CAM="$1"
  shift
  COLOR_TOPIC="/synced/${CAM}/color/image_raw/compressed"
  COLOR_CAMERA_INFO_TOPIC="/synced/${CAM}/color/camera_info"
  DEPTH_TOPIC="/synced/${CAM}/depth/image_rect_raw/compressedDepth"
  DEPTH_CAMERA_INFO_TOPIC="/synced/${CAM}/depth/camera_info"
  CAM_OVERRIDES=(
    -r "__node:=native_depth_tool_pose_${CAM}"
    -p "view:=${CAM}"
    -p "expected_color_frame:=${CAM}_color_optical_frame"
    -p "expected_depth_frame:=${CAM}_depth_optical_frame"
    -p "pose_topic:=/perception/${CAM}/tool/poses"
    -p "observation_topic:=/perception/${CAM}/tool/observations"
    -p "overlay_topic:=/perception/${CAM}/tool/overlay/compressed"
    -p "pose_overlay_topic:=/perception/${CAM}/tool/pose_overlay/compressed"
    -p "diagnostics_topic:=/perception/${CAM}/tool/diagnostics"
    -p "health_topic:=/perception/${CAM}/tool/health"
  )
fi
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="${TOOL}/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RFDETR_PYTHON}" \
  "${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/native_depth_tool_pose" \
  --ros-args \
  --params-file "${TOOL}/ros2_ws/src/pnu_surgical_perception/config/cam4_reference_mcap_native_pose.yaml" \
  -p "algorithm_python_path:=${TOOL}/algorithm/src" \
  -p "checkpoint:=${TOOL_CHECKPOINT}" \
  -p "ontology:=${TOOL}/algorithm/model/ontology.json" \
  -p "rgb_topic:=${COLOR_TOPIC}" \
  -p "color_camera_info_topic:=${COLOR_CAMERA_INFO_TOPIC}" \
  -p "depth_topic:=${DEPTH_TOPIC}" \
  -p "depth_camera_info_topic:=${DEPTH_CAMERA_INFO_TOPIC}" \
  -p "require_depth:=false" \
  "${CAM_OVERRIDES[@]}" \
  "$@"
