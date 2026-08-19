#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
exec ros2 bag play "${MCAP_PATH}" --loop --topics \
  "${COLOR_CAMERA_INFO_TOPIC}" "${DEPTH_CAMERA_INFO_TOPIC}" \
  "${COLOR_TOPIC}" "${DEPTH_TOPIC}"
