#!/usr/bin/env bash
# The perception stack: CAM3/CAM4 ingress, CAM3 Tool, concurrent CAM4
# Tool/Hand/Blood, and the final Debug overlay.  Each child script owns its own
# DDS profile and ROS environment.
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

"${ROOT}/scripts/run_perception_ingress.sh" both & PIDS+=("$!")
"${ROOT}/scripts/run_tool_v16.sh" cam_3 & PIDS+=("$!")
"${ROOT}/scripts/run_perception_concurrent_ingress.sh" & PIDS+=("$!")
"${ROOT}/scripts/run_final_overlay.sh" & PIDS+=("$!")
echo 'Perception stack started: cam_3 + cam_4 ingress, CAM3 Tool, CAM4 Tool + Hand + Blood, final overlay'
wait
