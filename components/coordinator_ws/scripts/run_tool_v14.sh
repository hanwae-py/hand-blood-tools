#!/usr/bin/env bash
# Start only Tool v1.4. A camera/MCAP publisher must already be running.
set -euo pipefail
ROOT=/home/hanwae/surgical_robot
set +u
source /opt/ros/jazzy/setup.bash
set -u
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
exec "${ROOT}/run_tool_runtime.sh" v14 "$@"
