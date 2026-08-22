#!/usr/bin/env bash
# Single 2-up Debug JPEG from local CAM3/CAM4 ingress and structured results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" mtu-safe
TOOL="${ROOT}/components/tool_runtime_v1_6"
HAND="${ROOT}/components/hand_keypoints_ros"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
exec "${PERCEPTION_PYTHON:-python3}" \
  "${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/final_overlay_compositor" \
  --ros-args -r __node:=final_overlay_compositor "$@"
