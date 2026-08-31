#!/usr/bin/env bash
# Native-resolution final overlays from local ingress and structured results.
# The rqt viewer subscribes directly to the four view topics; it does not need
# a second 2x2 image topic generated only for display.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL_ROI_PROFILE_OVERRIDE="${TOOL_ROI_PROFILE:-}"
source "${ROOT}/scripts/perception_runtime_env.sh" mtu-safe
TOOL="${ROOT}/components/tool_runtime_v1_6"
HAND="${ROOT}/components/hand_keypoints_ros"
ROI_PROFILE_ROOT="${TOOL}/ros2_ws/src/pnu_surgical_perception/config/roi_profiles"
ROI_ARGS=()

add_tool_roi_profile() {
  local camera="$1"
  local camera_profile="${camera/_/}"
  local profile
  if [[ -n "${TOOL_ROI_PROFILE_OVERRIDE}" ]]; then
    profile="${TOOL_ROI_PROFILE_OVERRIDE}"
  elif [[ "${camera}" == "cam_3" ]]; then
    profile="${TOOL_ROI_PROFILE_CAM3:-${TOOL_ROI_PROFILE:-none}}"
  else
    profile="${TOOL_ROI_PROFILE_CAM4:-${TOOL_ROI_PROFILE:-none}}"
  fi
  if [[ "${profile}" == "none" ]]; then
    return
  fi
  if [[ ! "${profile}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "Invalid TOOL_ROI_PROFILE name: ${profile}" >&2
    return 2
  fi
  if [[ "${profile}" != "${camera_profile}_"* ]]; then
    echo "ROI profile ${profile} does not match ${camera_profile}" >&2
    return 2
  fi
  local profile_file="${ROI_PROFILE_ROOT}/${profile}.yaml"
  if [[ ! -f "${profile_file}" ]]; then
    echo "Missing Tool ROI profile: ${profile_file}" >&2
    return 2
  fi
  ROI_ARGS+=(
    -p "${camera}_tool_roi_profile_file:=${profile_file}"
  )
}

add_tool_roi_profile cam_3
add_tool_roi_profile cam_4
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
exec "${PERCEPTION_PYTHON:-python3}" \
  "${TOOL}/ros2_ws/install/pnu_surgical_perception/lib/pnu_surgical_perception/final_overlay_compositor" \
  --ros-args -r __node:=final_overlay_compositor \
  --params-file "${ROOT}/config/cam4_hand_roi.yaml" \
  "${ROI_ARGS[@]}" \
  -p enable_blood:=false \
  -p enable_composite_output:=false \
  -p per_view_native_resolution:=true \
  -p per_view_jpeg_quality:=100 \
  -p output_rate_hz:="${FINAL_OVERLAY_OUTPUT_RATE_HZ:-30.0}" \
  "$@"
