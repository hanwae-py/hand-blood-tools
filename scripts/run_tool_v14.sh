#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
TOOL="${ROOT}/components/tool_runtime_v1_4"
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
  -p "checkpoint:=${TOOL_V14_CHECKPOINT}" \
  -p "color_topic:=${COLOR_TOPIC}" \
  -p "color_camera_info_topic:=${COLOR_CAMERA_INFO_TOPIC}" \
  -p "depth_topic:=${DEPTH_TOPIC}" \
  -p "depth_camera_info_topic:=${DEPTH_CAMERA_INFO_TOPIC}" \
  "$@"
