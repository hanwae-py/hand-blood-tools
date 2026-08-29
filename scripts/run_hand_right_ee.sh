#!/usr/bin/env bash
# RGB-only MediaPipe Open-Palm / Closed-Fist evidence for EIR right EE view.
# This intentionally publishes perception evidence only; no pose/action path
# and no unprovided depth/extrinsics contract are enabled.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
HAND="${ROOT}/components/hand_keypoints_ros"
TOOL="${ROOT}/components/tool_runtime_v1_6"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node" \
  --ros-args -r __node:=hand_detection_node_right_ee \
  -p camera:=right_ee \
  -p color_topic:=/perception/ingress/right_ee/color/image_raw/compressed \
  -p color_transport:=compressed \
  -p camera_info_topic:=/perception/ingress/right_ee/color/camera_info \
  -p depth_source:=rgb_only \
  -p depth_alignment_validated:=false \
  -p require_extrinsics_topic:=false \
  -p palm_facing_enabled:=false \
  -p gesture_profile:=right_ee \
  -p max_hands:=1 \
  -p publish_overlay:=false \
  -p publish_target_pose:=false \
  -p autostart:=true \
  "$@"
