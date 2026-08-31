#!/usr/bin/env bash
# Record the published CAM3, CAM4, and right-EE recognition overlays
# (/perception/*/overlay/compressed), not camera raw.
# Starts immediately; Ctrl+C stops.
# Usage:
#   bash scripts/record_recog_overlay_mcap.sh
# Output: $HOME/Videos/recordings/<MMDDHHMM>/{metadata.json,recogcam_0.mcap}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
export PYTHONPATH="${ROOT}/scripts:${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
exec python3 "${ROOT}/scripts/record_recog_overlay_mcap.py" "$@"
