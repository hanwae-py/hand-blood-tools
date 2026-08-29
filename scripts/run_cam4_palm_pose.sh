#!/usr/bin/env bash
# Perception-only CAM4 hand-palm pose transform.  It publishes no TF and never
# contacts a robot action/controller interface.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
TOOL="${ROOT}/components/tool_runtime_v1_6"
HAND="${ROOT}/components/hand_keypoints_ros"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
exec "${PERCEPTION_PYTHON:-python3}" \
  "${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/cam4_palm_pose_transform" \
  --ros-args -r __node:=cam4_palm_pose_transform "$@"
