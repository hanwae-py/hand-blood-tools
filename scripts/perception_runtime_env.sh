#!/usr/bin/env bash
# Source from a perception-only runner. It intentionally does not alter a
# login shell or a robot-control unit. The default stays LAN-safe; local
# ingress/worker runners must opt into ``local-fast`` explicitly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
PROFILE="${1:-mtu-safe}"
if (( $# > 1 )); then
  echo "usage: source perception_runtime_env.sh [mtu-safe|local-fast]" >&2
  return 2 2>/dev/null || exit 2
fi

case "${PROFILE}" in
  mtu-safe)
    PROFILE_FILE="${ROOT}/config/cyclonedds_perception_mtu_safe.xml"
    ;;
  local-fast)
    PROFILE_FILE="${ROOT}/config/cyclonedds_perception_local_fast.xml"
    ;;
  *)
    echo "unknown perception DDS profile: ${PROFILE}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
test -f "${PROFILE_FILE}"

# The camera and 1.7 interfaces are intentionally normalized to ROS domain 0.
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PERCEPTION_DDS_PROFILE="${PROFILE}"
export CYCLONEDDS_URI="file://${PROFILE_FILE}"
unset ROS_LOCALHOST_ONLY

# Keep the dedicated Conda runtimes isolated from packages installed in the
# login user's ~/.local tree.  In particular, a user-site NumPy 2.x must not
# override the NumPy 1.x ABI used by ROS cv_bridge and these model runtimes.
export PYTHONNOUSERSITE=1
