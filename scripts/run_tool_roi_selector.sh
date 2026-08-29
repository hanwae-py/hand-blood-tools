#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

export PYTHONNOUSERSITE=1
GUI_PYTHON="${HAND_GUI_PYTHON:-${HAND_PYTHON}}"
if [[ ! -x "${GUI_PYTHON}" ]]; then
  echo "Tool ROI selector Python is not executable: ${GUI_PYTHON}" >&2
  exit 1
fi

exec "${GUI_PYTHON}" "${ROOT}/scripts/select_live_tool_roi.py" "$@"
