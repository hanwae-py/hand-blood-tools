#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
WORKSPACE_SETUP="${BUNDLE_ROOT}/ros2_ws/install/setup.bash"
ALGORITHM_SOURCE="${BUNDLE_ROOT}/algorithm/src"
CHECKPOINT="${BUNDLE_ROOT}/algorithm/model/cam4_rfdetr_seg_small_regular_resume_e13_best.pth"
ONTOLOGY="${BUNDLE_ROOT}/algorithm/model/ontology.json"
PARAMETERS="${BUNDLE_ROOT}/ros2_ws/src/pnu_surgical_perception/config/cam4_reference_mcap_native_pose.yaml"

for required in "${ROS_SETUP}" "${WORKSPACE_SETUP}" "${CHECKPOINT}" "${ONTOLOGY}" "${PARAMETERS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path not found: ${required}" >&2
    exit 2
  fi
done

set +u
source "${ROS_SETUP}"
source "${WORKSPACE_SETUP}"
set -u
export PYTHONPATH="${ALGORITHM_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

exec ros2 run pnu_surgical_perception native_depth_tool_pose --ros-args \
  --params-file "${PARAMETERS}" \
  -p "algorithm_python_path:=${ALGORITHM_SOURCE}" \
  -p "checkpoint:=${CHECKPOINT}" \
  -p "ontology:=${ONTOLOGY}" \
  "$@"
