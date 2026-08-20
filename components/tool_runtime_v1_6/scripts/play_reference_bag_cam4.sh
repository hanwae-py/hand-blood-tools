#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/reference.mcap [additional ros2 bag play options]" >&2
  exit 2
fi

BAG_PATH="$1"
shift
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup not found: ${ROS_SETUP}" >&2
  exit 2
fi
if [[ ! -e "${BAG_PATH}" ]]; then
  echo "Bag not found: ${BAG_PATH}" >&2
  exit 2
fi

set +u
source "${ROS_SETUP}"
set -u
exec ros2 bag play "${BAG_PATH}" \
  --topics \
  /synced/cam_4/color/camera_info \
  /synced/cam_4/depth/camera_info \
  /synced/cam_4/color/image_raw/compressed \
  /synced/cam_4/depth/image_rect_raw/compressedDepth \
  "$@"
