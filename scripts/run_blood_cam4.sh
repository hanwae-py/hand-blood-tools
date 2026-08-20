#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
COORD="${ROOT}/components/coordinator_ws"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${COORD}/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_4/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RFDETR_PYTHON}" -m surgical_task_coordinator.blood_detection_node \
  --ros-args -r __node:=blood_detection_node \
  -p color_topic:="${COLOR_TOPIC}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p checkpoint:="${BLOOD_CHECKPOINT}" \
  -p confidence_threshold:=0.5 -p optimize:=true \
  -p require_depth:=false "$@"
