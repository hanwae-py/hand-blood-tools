#!/usr/bin/env bash
# MCAP replay + Tool v1.4 + Hand + real Blood segmentation.
# The coordinator selects exactly one active detector with mode_command.
set -eo pipefail

ROOT=/home/hanwae/surgical_robot
COORD="${ROOT}/coordinator_ws"
V14_ROOT="${ROOT}/tool_detection_runtime_v1_4_rc1"
MCAP=/home/hanwae/multicam_viplab_only_30s_20260814_134233.staging_0.mcap
GATE_TOPIC=/surgery/perception/cam4/tool_processing_enabled

export ROS_DOMAIN_ID=102
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset ROS_LOCALHOST_ONLY
export GALLIUM_DRIVER=d3d12
source /opt/ros/jazzy/setup.bash
source "${COORD}/install/setup.bash"

PIDS=()
cleanup() {
  set +e
  for pid in "${PIDS[@]}"; do kill "${pid}" >/dev/null 2>&1; done
  for pid in "${PIDS[@]}"; do wait "${pid}" >/dev/null 2>&1; done
}
trap cleanup EXIT INT TERM

for required in \
  "${MCAP}" \
  "${V14_ROOT}/scripts/play_reference_bag_cam4.sh" \
  "${COORD}/scripts/run_tool_v14.sh" \
  "${COORD}/scripts/run_hand_cam4.sh" \
  "${COORD}/scripts/run_blood_cam4.sh" \
  "${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/v14_tool_lifecycle_gate"; do
  [[ -e "${required}" ]] || { echo "Missing: ${required}" >&2; exit 1; }
done

echo 'Starting looping MCAP RGB + native compressedDepth replay...'
"${V14_ROOT}/scripts/play_reference_bag_cam4.sh" "${MCAP}" --loop &
PIDS+=("$!")

echo 'Starting Tool v1.4 (preloaded, initially gated off)...'
"${COORD}/scripts/run_tool_v14.sh" \
  -r __node:=native_depth_tool_pose \
  -p processing_enabled:=false \
  -p processing_gate_topic:="${GATE_TOPIC}" &
PIDS+=("$!")

echo 'Starting Tool lifecycle gate, Hand, real Blood, and coordinator...'
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/v14_tool_lifecycle_gate" \
  --ros-args -r __node:=tool_detection_node -p gate_topic:="${GATE_TOPIC}" &
PIDS+=("$!")

"${COORD}/scripts/run_hand_cam4.sh" -p autostart:=false &
PIDS+=("$!")

"${COORD}/scripts/run_blood_cam4.sh" &
PIDS+=("$!")

"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/perception_mode_coordinator" \
  --ros-args -p preload_models_on_startup:=true -p release_gpu_between_modes:=false &
PIDS+=("$!")

cat <<'EOF'

READY. Wait for both lines:
  [IDLE] all detector models preloaded; waiting for command
  RF-DETR and planar-pose algorithm loaded

Then send DETECT_TOOL, DETECT_HAND, DETECT_BLOOD, or STOP on:
  /surgery/perception/mode_command

Tool uses v1.4 native RGB-D planar pose. Hand uses MediaPipe. Blood uses the
real RF-DETR Seg-Small Blood model and publishes 2D masks/blue overlay only.
EOF
wait
