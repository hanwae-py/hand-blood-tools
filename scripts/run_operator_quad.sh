#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" mtu-safe
TOOL="${ROOT}/components/tool_runtime_v1_6"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
set -u
exec "${PERCEPTION_PYTHON:-python3}" \
  "${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/operator_quad_compositor" \
  --ros-args -r __node:=operator_quad_compositor \
  -p output_rate_hz:=15.0 \
  "$@"
