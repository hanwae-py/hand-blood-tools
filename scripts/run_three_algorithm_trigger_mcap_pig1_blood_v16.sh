#!/usr/bin/env bash
# MCAP Tool v1.6 + Hand, with pig1 images supplied only to the Blood node.
# Exactly one detector is active per coordinator command.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
COORD="${ROOT}/components/coordinator_ws"
GATE_TOPIC=/perception/cam_4/tool/processing_enabled
BLOOD_PIG1_TOPIC=/surgery/test/blood_pig1/color/image_raw/compressed
BLOOD_PIG1_IMAGES_DIR="${BLOOD_PIG1_IMAGES_DIR:-${HOME}/blood/pig1/imgs}"

if [[ ! -d "${BLOOD_PIG1_IMAGES_DIR}" ]]; then
  echo "Blood pig1 images not found: ${BLOOD_PIG1_IMAGES_DIR}" >&2
  exit 1
fi

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${COORD}/install/setup.bash"
set -u

PIDS=()
cleanup() {
  set +e
  for pid in "${PIDS[@]}"; do kill "${pid}" >/dev/null 2>&1; done
  for pid in "${PIDS[@]}"; do wait "${pid}" >/dev/null 2>&1; done
}
trap cleanup EXIT INT TERM

"${ROOT}/scripts/play_mcap.sh" & PIDS+=("$!")
/usr/bin/python3 "${ROOT}/scripts/play_blood_pig1.py" \
  --images-dir "${BLOOD_PIG1_IMAGES_DIR}" \
  --topic "${BLOOD_PIG1_TOPIC}" --fps 3 & PIDS+=("$!")
"${ROOT}/scripts/run_tool_v16.sh" \
  -r __node:=native_depth_tool_pose \
  -p processing_enabled:=false \
  -p processing_gate_topic:="${GATE_TOPIC}" & PIDS+=("$!")
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/v14_tool_lifecycle_gate" \
  --ros-args -r __node:=tool_detection_node -p gate_topic:="${GATE_TOPIC}" & PIDS+=("$!")
"${ROOT}/scripts/run_hand_cam4.sh" -p autostart:=false & PIDS+=("$!")
"${ROOT}/scripts/run_blood_cam4.sh" -p color_topic:="${BLOOD_PIG1_TOPIC}" & PIDS+=("$!")
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/perception_mode_coordinator" \
  --ros-args -p preload_models_on_startup:=true -p release_gpu_between_modes:=false & PIDS+=("$!")

echo 'Ready: Tool v1.6 and Hand use MCAP; DETECT_BLOOD uses looping pig1 images.'
wait
