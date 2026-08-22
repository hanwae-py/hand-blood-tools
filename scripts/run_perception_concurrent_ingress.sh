#!/usr/bin/env bash
# Supervise CAM4 Tool + Hand + Blood concurrently.  The old
# perception_mode_coordinator is deliberately not used: it enforces one active
# worker at a time, while this integration needs all three current layers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
PIDS=()
cleanup() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

"${ROOT}/scripts/run_tool_v16.sh" cam_4 & PIDS+=("$!")
"${ROOT}/scripts/run_hand_cam4.sh" cam_4 -p autostart:=true & PIDS+=("$!")
"${ROOT}/scripts/run_blood_cam4.sh" cam_4 -p autostart:=true & PIDS+=("$!")
echo 'CAM4 concurrent workers started: Tool + Hand + Blood (local ingress only)'
wait
