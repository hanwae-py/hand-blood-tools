#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
set -u
cd "${BUNDLE_ROOT}/ros2_ws"
colcon build \
  --packages-select surgical_perception_msgs pnu_surgical_perception \
  --symlink-install

echo "Build complete: ${BUNDLE_ROOT}/ros2_ws/install"
