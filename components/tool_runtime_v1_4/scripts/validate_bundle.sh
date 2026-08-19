#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

cd "${BUNDLE_ROOT}"
sha256sum -c SHA256SUMS

export PYTHONPATH="${BUNDLE_ROOT}/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 algorithm/validation/validate_pose_contract.py
python3 algorithm/validation/validate_native_depth_registration.py

if [[ -f "${ROS_SETUP}" ]]; then
  # Keep target-directory Python packages used for RF-DETR validation from
  # shadowing the ROS/ament setuptools and pytest environment.
  unset PYTHONPATH
  set +u
  source "${ROS_SETUP}"
  if [[ -f "${BUNDLE_ROOT}/ros2_ws/install/setup.bash" ]]; then
    source "${BUNDLE_ROOT}/ros2_ws/install/setup.bash"
  else
    echo "ROS2 workspace is not built; run ./scripts/build_ros2.sh first" >&2
    exit 2
  fi
  set -u
  cd "${BUNDLE_ROOT}/ros2_ws"
  colcon test \
    --packages-select surgical_perception_msgs pnu_surgical_perception \
    --event-handlers console_direct+
  colcon test-result --verbose
else
  echo "ROS2 validation skipped; setup not found: ${ROS_SETUP}" >&2
fi
