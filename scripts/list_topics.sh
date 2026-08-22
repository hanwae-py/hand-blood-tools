#!/usr/bin/env bash
# List topics that currently have a publisher (receivable on this ROS domain).
# Usage: bash scripts/list_topics.sh [filter]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

FILTER="${1:-}"
SPIN_TIME="${SPIN_TIME:-2}"

printf 'ROS_DOMAIN_ID=%s  RMW=%s  discovery=%s  peers=%s\n' \
  "${ROS_DOMAIN_ID}" "${RMW_IMPLEMENTATION}" \
  "${ROS_AUTOMATIC_DISCOVERY_RANGE:-}" "${ROS_STATIC_PEERS:-}"

raw="$(ros2 topic list -t -v --no-daemon --spin-time "${SPIN_TIME}")"
published="$(printf '%s\n' "${raw}" | awk '
  /^Published topics:/{p=1; next}
  /^Subscribed topics:/{p=0}
  p && /^ \*/ {print}
')"

topic_is_published() {
  printf '%s\n' "${published}" | grep -qF -- " * ${1} ["
}

printf '\nSynced cameras (from published /synced/<cam>/... topics):\n'
synced_cams="$(printf '%s\n' "${published}" | sed -n 's|^ \* /synced/\([^/]*\)/.*|\1|p' | sort -u)"
if [[ -z "${synced_cams}" ]]; then
  printf '  (none)\n'
else
  printf '%s\n' "${synced_cams}" | sed 's/^/  /'
fi

printf '\nExpected default inputs (from config/system.env):\n'
check_expected() {
  local label="$1" topic="$2"
  if topic_is_published "${topic}"; then
    printf '  OK   %-28s %s\n' "${label}" "${topic}"
  else
    printf '  MISS %-28s %s\n' "${label}" "${topic}"
  fi
}
check_expected COLOR_TOPIC "${COLOR_TOPIC}"
check_expected COLOR_CAMERA_INFO_TOPIC "${COLOR_CAMERA_INFO_TOPIC}"
check_expected DEPTH_TOPIC "${DEPTH_TOPIC}"
check_expected DEPTH_CAMERA_INFO_TOPIC "${DEPTH_CAMERA_INFO_TOPIC}"

printf '\nReceivable topics (have a publisher):\n'
if [[ -n "${FILTER}" ]]; then
  filtered="$(printf '%s\n' "${published}" | grep -E -- "${FILTER}" || true)"
else
  filtered="$(printf '%s\n' "${published}" | grep -vE -- ' \* /(parameter_events|rosout) \[' || true)"
fi

if [[ -z "${filtered}" ]]; then
  printf '  (none)\n'
  exit 0
fi
printf '%s\n' "${filtered}"
