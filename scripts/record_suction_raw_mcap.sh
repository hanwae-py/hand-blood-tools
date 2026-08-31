#!/usr/bin/env bash
# Record the live EIR suction camera as raw sensor_msgs/Image into an MCAP
# bag plus metadata.json.  EIR only publishes JPEG and compressedDepth; this
# process decodes locally and does not republish the raw images.
#
# Default recording size is the suction camera's 848x480 at 4 FPS.
# Starts immediately; Ctrl+C stops.
# Usage:
#   bash scripts/record_suction_raw_mcap.sh
# Output: $HOME/Videos/recordings/<MMDDHHMM>/{metadata.json,bloodcam_0.mcap}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
exec python3 "${ROOT}/scripts/record_suction_raw_mcap.py" "$@"
