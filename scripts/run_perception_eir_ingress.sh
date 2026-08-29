#!/usr/bin/env bash
# Preserve one EIR camera's source messages while adapting CameraInfo QoS for
# local workers. Head forwards aligned RGB-D; the end-effectors remain RGB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
TOOL="${ROOT}/components/tool_runtime_v1_6"

CAMERA="${1:-}"
case "${CAMERA}" in
  head|left_ee|right_ee) ;;
  *)
    echo "usage: $0 head|left_ee|right_ee" >&2
    exit 2
    ;;
esac

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
set -u

BIN="${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/perception_ingress"
exec "${PERCEPTION_PYTHON:-python3}" "${BIN}" --ros-args \
  -r "__node:=perception_ingress_${CAMERA}" \
  -p "camera:=${CAMERA}"
