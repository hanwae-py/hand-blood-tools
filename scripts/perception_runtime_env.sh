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

# NumPy/OpenBLAS and PyTorch otherwise each fan small per-frame geometry
# operations out across every logical CPU.  With several camera workers this
# oversubscribes the host (the head Tool worker alone used 16 busy BLAS
# threads) and delays latency-sensitive Hand/overlay callbacks.  These
# operations are small enough that one native math thread is faster in the
# complete concurrent pipeline.  Keep a dedicated override for profiling.
PERCEPTION_NATIVE_MATH_THREADS="${PERCEPTION_NATIVE_MATH_THREADS:-1}"
export OMP_NUM_THREADS="${PERCEPTION_OMP_NUM_THREADS:-${PERCEPTION_NATIVE_MATH_THREADS}}"
export OPENBLAS_NUM_THREADS="${PERCEPTION_OPENBLAS_NUM_THREADS:-${PERCEPTION_NATIVE_MATH_THREADS}}"
export MKL_NUM_THREADS="${PERCEPTION_MKL_NUM_THREADS:-${PERCEPTION_NATIVE_MATH_THREADS}}"
export NUMEXPR_NUM_THREADS="${PERCEPTION_NUMEXPR_NUM_THREADS:-${PERCEPTION_NATIVE_MATH_THREADS}}"
export VECLIB_MAXIMUM_THREADS="${PERCEPTION_VECLIB_NUM_THREADS:-${PERCEPTION_NATIVE_MATH_THREADS}}"
export BLIS_NUM_THREADS="${PERCEPTION_BLIS_NUM_THREADS:-${PERCEPTION_NATIVE_MATH_THREADS}}"
