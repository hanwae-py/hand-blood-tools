#!/usr/bin/env bash
# RGB-only MediaPipe hand/gesture overlay for EIR head and end-effector views.
# The EIR publisher remains the sole camera owner; this process only subscribes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
HAND="${ROOT}/components/hand_keypoints_ros"
TOOL="${ROOT}/components/tool_runtime_v1_6"

CAMERA="${1:-}"
case "${CAMERA}" in
  head)
    GESTURE_PROFILE=topview
    MAX_HANDS=4
    ;;
  left_ee|right_ee)
    # The EE profile is viewpoint-specific, not handedness-specific.
    GESTURE_PROFILE=right_ee
    MAX_HANDS=1
    ;;
  *)
    echo "usage: $0 head|left_ee|right_ee [ROS arguments...]" >&2
    exit 2
    ;;
esac
shift

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u

exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node" \
  --ros-args -r "__node:=hand_detection_node_${CAMERA}" \
  -p "camera:=${CAMERA}" \
  -p "color_topic:=/perception/ingress/${CAMERA}/color/image_raw/compressed" \
  -p color_transport:=compressed \
  -p "camera_info_topic:=/perception/ingress/${CAMERA}/color/camera_info" \
  -p depth_source:=rgb_only \
  -p depth_alignment_validated:=false \
  -p require_extrinsics_topic:=false \
  -p palm_facing_enabled:=false \
  -p "gesture_profile:=${GESTURE_PROFILE}" \
  -p "max_hands:=${MAX_HANDS}" \
  -p publish_overlay:=true \
  -p publish_target_pose:=false \
  -p autostart:=true \
  "$@"
