#!/usr/bin/env bash
# Five direct rqt views of the requested per-camera recognition overlays.
#
# No image is composed or republished for the screen. Each rqt instance
# subscribes directly to one native-resolution ``.../overlay/compressed``
# topic from the corresponding perception publisher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RQT_OVERLAY="${ROOT}/components/rqt_image_view_overlay_ws"
POSITIONER="${ROOT}/scripts/position_x11_window.py"
set +u
source /opt/ros/jazzy/setup.bash
source "${RQT_OVERLAY}/install/setup.bash"
set -u

RQT_IMAGE_VIEW="${RQT_OVERLAY}/install/lib/rqt_image_view/rqt_image_view"
if [[ ! -x "${RQT_IMAGE_VIEW}" ]]; then
  echo "Patched rqt_image_view is not built: ${RQT_IMAGE_VIEW}" >&2
  exit 1
fi
if [[ ! -x "${POSITIONER}" ]]; then
  echo "Missing X11 viewer positioner: ${POSITIONER}" >&2
  exit 1
fi

declare -a VIEWER_PIDS=()

stop_viewers() {
  local pid
  local attempt
  local any_alive
  for pid in "${VIEWER_PIDS[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for attempt in {1..20}; do
    any_alive=false
    for pid in "${VIEWER_PIDS[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=true
        break
      fi
    done
    "${any_alive}" || break
    sleep 0.25
  done
  # rqt can keep its Qt event loop alive after SIGTERM. These are only the
  # five explicitly launched viewer children, so bound shutdown instead of
  # making systemd wait for its full timeout on every viewer restart.
  for pid in "${VIEWER_PIDS[@]:-}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  for pid in "${VIEWER_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

cleanup() {
  trap - EXIT INT TERM HUP
  stop_viewers
}
trap cleanup EXIT INT TERM HUP

launch_view() {
  local topic="$1"
  local x="$2"
  local y="$3"
  local pid

  # The custom rqt image view takes ``base transport`` as one topic argument.
  "${RQT_IMAGE_VIEW}" -ht "${topic} compressed" &
  pid=$!
  VIEWER_PIDS+=("${pid}")
  # rqt standalone windows have the same title. Match X11's _NET_WM_PID so
  # each viewer is placed deterministically without touching user windows.
  # rqt restores its saved geometry shortly after mapping. Delay the EWMH
  # request until that restoration is complete, then make our layout win.
  (
    sleep 3
    /usr/bin/python3 "${POSITIONER}" \
      --pid "${pid}" --title "Image View" \
      --x "${x}" --y "${y}" \
      --width 620 --height 444 --timeout-sec 12
    /usr/bin/python3 "${POSITIONER}" \
      --pid "${pid}" --title "rqt_image_view__ImageView - rqt" \
      --hide --timeout-sec 12
  ) &
}

# 1920x1080 GNOME work area: three views on the top row and two centered on
# the bottom row. Streams remain native on ROS; only rqt's local widget scales
# them to fit the physical screen.
launch_view /perception/cam_3/overlay 0 0
launch_view /perception/cam_4/overlay 650 0
launch_view /perception/head/hand/overlay 1300 0
launch_view /perception/left_ee/hand/overlay 325 540
launch_view /perception/right_ee/overlay 975 540

# If any direct view exits, systemd restarts this group as one coherent 3+2
# display instead of leaving stale or mispositioned windows behind.
wait -n "${VIEWER_PIDS[@]}"
