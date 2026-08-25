#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
export ROS_DOMAIN_ID=0
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
ROS_SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
WORKSPACE_SETUP="${BUNDLE_ROOT}/ros2_ws/install/setup.bash"
ALGORITHM_SOURCE="${BUNDLE_ROOT}/algorithm/src"
ONTOLOGY="${BUNDLE_ROOT}/algorithm/model/ontology.json"
PARAMETERS="${BUNDLE_ROOT}/ros2_ws/src/pnu_surgical_perception/config/cam4_reference_mcap_native_pose.yaml"

MODEL_SIZE="${TOOL_MODEL_SIZE:-small}"
case "${MODEL_SIZE}" in
  small)
    CHECKPOINT="${TOOL_CHECKPOINT_SMALL:-${TOOL_CHECKPOINT:-${BUNDLE_ROOT}/algorithm/model/cam4_rfdetr_seg_small_regular_resume_best.pth}}"
    CHECKPOINT_COLOR_ORDER="BGR"
    ENABLE_CLASS_AGNOSTIC_NMS="true"
    MODEL_VERSION="cam4-rfdetr-seg-small-regular-resume-best"
    MODEL_DEFAULT_THRESHOLD="0.30"
    ;;
  medium)
    CHECKPOINT="${TOOL_CHECKPOINT_MEDIUM:-${BUNDLE_ROOT}/algorithm/model/medium_best.pth}"
    CHECKPOINT_COLOR_ORDER="RGB"
    ENABLE_CLASS_AGNOSTIC_NMS="false"
    MODEL_VERSION="cam4-rfdetr-seg-medium-20260825-best"
    MODEL_DEFAULT_THRESHOLD="0.30"
    ;;
  large)
    CHECKPOINT="${TOOL_CHECKPOINT_LARGE:-${BUNDLE_ROOT}/algorithm/model/large_best.pth}"
    CHECKPOINT_COLOR_ORDER="RGB"
    ENABLE_CLASS_AGNOSTIC_NMS="false"
    MODEL_VERSION="cam4-rfdetr-seg-large-20260825-best"
    MODEL_DEFAULT_THRESHOLD="0.30"
    ;;
  xlarge)
    CHECKPOINT="${TOOL_CHECKPOINT_XLARGE:-${BUNDLE_ROOT}/algorithm/model/xlarge_best.pth}"
    CHECKPOINT_COLOR_ORDER="RGB"
    ENABLE_CLASS_AGNOSTIC_NMS="false"
    MODEL_VERSION="cam4-rfdetr-seg-xlarge-20260825-best"
    MODEL_DEFAULT_THRESHOLD="0.30"
    ;;
  *)
    echo "TOOL_MODEL_SIZE must be small, medium, large, or xlarge; got: ${MODEL_SIZE}" >&2
    exit 2
    ;;
esac
MODEL_THRESHOLD="${TOOL_CONFIDENCE_THRESHOLD:-${MODEL_DEFAULT_THRESHOLD}}"

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
  -r __node:=native_depth_tool_pose_cam_4 \
  --params-file "${PARAMETERS}" \
  -p "algorithm_python_path:=${ALGORITHM_SOURCE}" \
  -p "checkpoint:=${CHECKPOINT}" \
  -p "ontology:=${ONTOLOGY}" \
  -p "model_size:=${MODEL_SIZE}" \
  -p "checkpoint_color_order:=${CHECKPOINT_COLOR_ORDER}" \
  -p "model_version:=${MODEL_VERSION}" \
  -p "confidence_threshold:=${MODEL_THRESHOLD}" \
  -p "enable_class_agnostic_nms:=${ENABLE_CLASS_AGNOSTIC_NMS}" \
  "$@"
