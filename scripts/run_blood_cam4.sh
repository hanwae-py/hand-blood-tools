#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
COORD="${ROOT}/components/coordinator_ws"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${COORD}/install/setup.bash"
set -u
exec "${RFDETR_PYTHON}" -m surgical_task_coordinator.blood_detection_node \
  --ros-args -r __node:=blood_detection_node \
  -p color_topic:="${COLOR_TOPIC}" \
  -p checkpoint:="${BLOOD_CHECKPOINT}" \
  -p confidence_threshold:=0.5 -p optimize:=true "$@"
