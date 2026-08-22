#!/usr/bin/env bash
# One process per camera: this is the only external /synced RGB-D subscriber
# after the ingress cutover.  It forwards source messages unchanged locally.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
TOOL="${ROOT}/components/tool_runtime_v1_6"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
set -u
BIN="${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/perception_ingress"
PYTHON_BIN="${PERCEPTION_PYTHON:-python3}"

run_one() {
  local camera="$1"
  "${PYTHON_BIN}" "${BIN}" --ros-args \
    -r "__node:=perception_ingress_${camera}" \
    -p "camera:=${camera}"
}

case "${1:-cam_4}" in
  cam_3|cam3|3) exec bash -c '"$0" _one cam_3' "$0" ;;
  cam_4|cam4|4) exec bash -c '"$0" _one cam_4' "$0" ;;
  _one) run_one "${2:?camera required}" ;;
  both)
    run_one cam_3 & pid3=$!
    run_one cam_4 & pid4=$!
    cleanup() { kill "${pid3}" "${pid4}" 2>/dev/null || true; wait "${pid3}" "${pid4}" 2>/dev/null || true; }
    trap cleanup EXIT INT TERM
    wait "${pid3}" "${pid4}"
    ;;
  *)
    echo "usage: $0 [cam_4|cam_3|both]" >&2
    exit 2
    ;;
esac
