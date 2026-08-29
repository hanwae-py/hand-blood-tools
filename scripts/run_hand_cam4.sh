#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
source "${ROOT}/scripts/select_cam.sh"
HAND="${ROOT}/components/hand_keypoints_ros"
TOOL="${ROOT}/components/tool_runtime_v1_6"
CAM="cam_4"
CAM_OVERRIDES=()
NODE_NAME="hand_detection_node"
PARAM_FILE_ARGS=()
DEPTH_ALIGNMENT_VALIDATED="false"
DEPTH_SOURCE="rgb_only"
REQUIRE_EXTRINSICS_TOPIC="false"
if [[ "${1:-}" == "help" || "${1:-}" == "--help" ]]; then
  echo "usage: bash scripts/run_hand_cam4.sh [cam_1|cam_2|cam_3|cam_4|flir]" >&2
  set +u
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
  print_synced_cameras
  exit 0
fi
if is_camera_selector "${1:-}"; then
  apply_ingress_cam "$1"
  shift
else
  apply_ingress_cam "${CAM}"
fi
NODE_NAME="hand_detection_node_${CAM}"
CAM_OVERRIDES=(
  -p "camera:=${CAM}"
  -p "keypoints_topic:=/perception/${CAM}/hand/keypoints"
  -p "gesture_topic:=/perception/${CAM}/hand/gestures"
  -p "facing_topic:=/perception/${CAM}/hand/facing"
  -p "overlay_topic:=/perception/${CAM}/hand/overlay/compressed"
  -p "target_pose_topic:=/perception/${CAM}/hand/target_pose"
  -p "health_topic:=/perception/${CAM}/hand/health"
  -p "diagnostics_topic:=/perception/${CAM}/hand/diagnostics"
)
case "${CAM}" in
  cam_1)
    PARAM_FILE_ARGS=(--params-file "${ROOT}/config/cam1_depth_to_color.yaml")
    DEPTH_ALIGNMENT_VALIDATED="true"
    DEPTH_SOURCE="real"
    # CAM1 now publishes its native depth-to-color extrinsics.  Require that
    # live, serial-matched calibration so metric registration fails closed if
    # the relay disappears; the stored SDK values remain a validation record.
    REQUIRE_EXTRINSICS_TOPIC="true"
    CAM_OVERRIDES+=(
      -p rgb_fallback_when_real_depth_missing:=true
    )
    ;;
  cam_2)
    PARAM_FILE_ARGS=(--params-file "${ROOT}/config/cam2_depth_to_color.yaml")
    DEPTH_ALIGNMENT_VALIDATED="true"
    DEPTH_SOURCE="real"
    REQUIRE_EXTRINSICS_TOPIC="true"
    CAM_OVERRIDES+=(
      -p rgb_fallback_when_real_depth_missing:=true
    )
    ;;
  cam_3)
    PARAM_FILE_ARGS=(--params-file "${ROOT}/config/cam3_depth_to_color.yaml")
    DEPTH_ALIGNMENT_VALIDATED="true"
    DEPTH_SOURCE="real"
    REQUIRE_EXTRINSICS_TOPIC="true"
    ;;
  cam_4)
    PARAM_FILE_ARGS=(
      --params-file "${ROOT}/config/cam4_depth_to_color.yaml"
      --params-file "${ROOT}/config/cam4_hand_roi.yaml"
    )
    DEPTH_ALIGNMENT_VALIDATED="true"
    DEPTH_SOURCE="real"
    REQUIRE_EXTRINSICS_TOPIC="true"
    ;;
esac
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${TOOL}/ros2_ws/install/setup.bash"
source "${HAND}/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="${ROOT}/components/tool_runtime_v1_6/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${HAND_PYTHON}" \
  "${HAND}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node" \
  --ros-args -r "__node:=${NODE_NAME}" \
  "${PARAM_FILE_ARGS[@]}" \
  -p color_topic:="${COLOR_TOPIC}" \
  -p color_transport:="${COLOR_TRANSPORT}" \
  -p depth_topic:="${DEPTH_TOPIC}" \
  -p depth_transport:="${DEPTH_TRANSPORT}" \
  -p camera_info_topic:="${COLOR_CAMERA_INFO_TOPIC}" \
  -p depth_camera_info_topic:="${DEPTH_CAMERA_INFO_TOPIC}" \
  -p extrinsics_topic:="${EXTRINSICS_TOPIC}" \
  -p require_extrinsics_topic:="${REQUIRE_EXTRINSICS_TOPIC}" \
  -p depth_source:="${DEPTH_SOURCE}" \
  -p depth_alignment_validated:="${DEPTH_ALIGNMENT_VALIDATED}" \
  -p publish_overlay:=true \
  -p publish_target_pose:=false \
  "${CAM_OVERRIDES[@]}" \
  "$@"
