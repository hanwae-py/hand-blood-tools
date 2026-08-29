#!/usr/bin/env bash
# The complete perception stack: four-camera ingress, CAM3/CAM4 Tool,
# CAM1/CAM3/CAM4 Hand, FLIR Blood, quality selection, and 2x2 operator view.
# Each child script owns its own DDS profile and ROS environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The persistent user units restart on their own, so starting here while they
# run would give VIPLab /synced a second subscriber.
if pgrep -f -- 'pnu_surgical_perception/lib/pnu_surgical_perception/perception_ingress' >/dev/null; then
  echo 'Ingress is already running; refusing to add a second /synced subscriber.' >&2
  echo 'Inspect it with: ps -eo pid,cmd | grep perception_ingress' >&2
  exit 1
fi

PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

"${ROOT}/scripts/run_perception_ingress.sh" all & PIDS+=("$!")
"${ROOT}/scripts/run_tool_v16.sh" cam_3 & PIDS+=("$!")
"${ROOT}/scripts/run_perception_concurrent_ingress.sh" & PIDS+=("$!")
"${ROOT}/scripts/run_hand_cam4.sh" cam_1 -p autostart:=true & PIDS+=("$!")
"${ROOT}/scripts/run_hand_cam4.sh" cam_3 -p autostart:=true & PIDS+=("$!")
"${ROOT}/scripts/run_blood_cam4.sh" flir -p autostart:=true & PIDS+=("$!")
"${ROOT}/scripts/run_multiview_hand_fusion.sh" & PIDS+=("$!")
"${ROOT}/scripts/run_final_overlay.sh" & PIDS+=("$!")
"${ROOT}/scripts/run_operator_quad.sh" & PIDS+=("$!")
echo 'Perception stack started: CAM1/CAM3/CAM4 Hand, CAM3/CAM4 Tool, FLIR Blood, fused selection, 2x2 overlay'
wait -n "${PIDS[@]}"
exit 1
