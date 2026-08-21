#!/usr/bin/env bash
# Tool v1.6 + Hand + real Blood against an already-running real camera.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
COORD="${ROOT}/components/coordinator_ws"
GATE_TOPIC=/perception/cam_4/tool/processing_enabled

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

"${ROOT}/scripts/run_tool_v16.sh" \
  -r __node:=native_depth_tool_pose \
  -p processing_enabled:=false \
  -p processing_gate_topic:="${GATE_TOPIC}" & PIDS+=("$!")
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/v14_tool_lifecycle_gate" \
  --ros-args -r __node:=tool_detection_node -p gate_topic:="${GATE_TOPIC}" & PIDS+=("$!")
"${ROOT}/scripts/run_hand_cam4.sh" -p autostart:=false & PIDS+=("$!")
"${ROOT}/scripts/run_blood_cam4.sh" & PIDS+=("$!")
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/perception_mode_coordinator" \
  --ros-args -p preload_models_on_startup:=true -p release_gpu_between_modes:=false & PIDS+=("$!")

echo 'Ready: wait for [IDLE], then send DETECT_TOOL, DETECT_HAND, DETECT_BLOOD, or STOP.'
wait
