#!/usr/bin/env bash
# Start real RF-DETR Blood segmentation. A compressed RGB camera/MCAP publisher
# must already be running. This is separate from the legacy Blood stub launcher.
set -eo pipefail

ROOT=/home/hanwae/surgical_robot
COORD="${ROOT}/coordinator_ws"
RFD_VENV="${ROOT}/rfdetr_perception_ros/.venv"

set +u
source /opt/ros/jazzy/setup.bash
source "${COORD}/install/setup.bash"
set -u
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"

exec "${RFD_VENV}/bin/python" -m surgical_task_coordinator.blood_detection_node \
  --ros-args \
  -r __node:=blood_detection_node \
  -p color_topic:=/synced/cam_4/color/image_raw/compressed \
  -p checkpoint:=/home/hanwae/blood/blood_detection.pth \
  -p confidence_threshold:=0.5 \
  -p optimize:=true \
  "$@"
