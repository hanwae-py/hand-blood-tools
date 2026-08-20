#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u

for workspace in \
  "${ROOT}/components/coordinator_ws" \
  "${ROOT}/components/hand_keypoints_ros/ros2_ws" \
  "${ROOT}/components/tool_runtime_v1_6/ros2_ws"; do
  echo "Building ${workspace}"
  (cd "${workspace}" && colcon build --symlink-install)
done
