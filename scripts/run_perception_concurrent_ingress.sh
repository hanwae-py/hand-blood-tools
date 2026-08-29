#!/usr/bin/env bash
# Supervise CAM4 Tool + Hand concurrently.  Blood is intentionally isolated in
# the FLIR worker so image provenance cannot be confused with CAM4.  The old
# perception_mode_coordinator is deliberately not used: it enforces one active
# worker at a time, while this integration needs both current layers.
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
# The final compositor draws typed keypoints directly. Avoid encoding an
# unconsumed per-camera Hand JPEG after every inference callback.
"${ROOT}/scripts/run_hand_cam4.sh" cam_4 \
  -p publish_overlay:=false \
  -p autostart:=true & PIDS+=("$!")
echo 'CAM4 concurrent workers started: Tool + Hand (local ingress only)'
wait -n "${PIDS[@]}"
exit 1
