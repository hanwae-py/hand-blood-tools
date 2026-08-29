#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

export PYTHONNOUSERSITE=1
HAND_GUI_PYTHON="${HAND_GUI_PYTHON:-/home/pnucvlab/miniforge3/envs/hand/bin/python}"
if [[ ! -x "${HAND_GUI_PYTHON}" ]]; then
  echo "CAM4 ROI selector Python is not executable: ${HAND_GUI_PYTHON}" >&2
  exit 1
fi

exec "${HAND_GUI_PYTHON}" "${ROOT}/scripts/select_cam4_hand_roi.py" \
  --config "${ROOT}/config/cam4_hand_roi.yaml" \
  --topic "/perception/ingress/cam_4/color/image_raw/compressed" \
  --restart-service "taskplanner-perception-cam4-ingress.service" \
  --restart-service "taskplanner-perception-final-overlay.service" \
  "$@"
