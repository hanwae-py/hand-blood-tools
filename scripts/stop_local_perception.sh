#!/usr/bin/env bash
# Stop local perception processes and the local ros2 daemon.
# Does not touch the remote camera publisher (multicam_node).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"

PATTERNS=(
  'pnu_surgical_perception/lib/pnu_surgical_perception/perception_ingress'
  'pnu_surgical_perception/lib/pnu_surgical_perception/native_depth_tool_pose'
  'pnu_surgical_perception/lib/pnu_surgical_perception/final_overlay_compositor'
  'pnu_surgical_perception/lib/pnu_surgical_perception/debug_overlay_compositor'
  'hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node'
  'surgical_task_coordinator/lib/surgical_task_coordinator/blood_detection_node'
  'surgical_task_coordinator/lib/surgical_task_coordinator/perception_mode_coordinator'
  'surgical_task_coordinator/lib/surgical_task_coordinator/v14_tool_lifecycle_gate'
  'scripts/run.sh'
  'scripts/run_perception_ingress.sh'
  'scripts/run_perception_concurrent_ingress.sh'
  'scripts/run_tool_v16.sh'
  'scripts/run_hand_cam4.sh'
  'scripts/run_blood_cam4.sh'
  'scripts/run_final_overlay.sh'
)

printf 'Stopping local perception processes...\n'
set +e
for pattern in "${PATTERNS[@]}"; do
  pids="$(pgrep -f -- "${pattern}" | tr '\n' ' ')"
  if [[ -n "${pids}" ]]; then
    printf '  kill %s  (%s)\n' "${pids}" "${pattern}"
    pkill -TERM -f -- "${pattern}" >/dev/null 2>&1
  else
    printf '  none  (%s)\n' "${pattern}"
  fi
done

sleep 1
for pattern in "${PATTERNS[@]}"; do
  pkill -KILL -f -- "${pattern}" >/dev/null 2>&1
done
set -e

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
printf 'Stopping local ros2 daemon...\n'
ros2 daemon stop >/dev/null 2>&1 || true

printf '\nRemaining local perception processes:\n'
left="$(ps -eo pid,cmd | grep -E 'perception_ingress|hand_detection_node|native_depth_tool_pose|blood_detection_node|final_overlay_compositor|debug_overlay_compositor|perception_mode_coordinator|v14_tool_lifecycle_gate' | grep -v grep || true)"
if [[ -z "${left}" ]]; then
  printf '  (none)\n'
else
  printf '%s\n' "${left}"
  exit 1
fi
