#!/usr/bin/env bash
# Dedicated RGB-only ingress for EIR's right end-effector camera.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
TOOL="${ROOT}/components/tool_runtime_v1_6"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
set -u
BIN="${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/perception_ingress"
exec "${PERCEPTION_PYTHON:-python3}" "${BIN}" --ros-args \
  -r __node:=perception_ingress_right_ee \
  -p camera:=right_ee
