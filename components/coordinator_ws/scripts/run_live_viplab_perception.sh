#!/usr/bin/env bash
# Run CAM4 tool detection + hand keypoints against the LIVE viplab
# multicam_node stream, with no fake_camera_publisher anywhere in the graph.
#
# Source of truth for every name and QoS below is the Taskplanner side:
#   - Taskplanner ROS 2 External Interface Contract v0.3.0 (surgical_interop_msgs 0.3.0)
#   - the live /integration/cv_contract/status snapshot published by
#     cv_contract_monitor, which lists each input/output this lab owns
#     together with the QoS it is checked against.
#
# Live inputs (publisher: multicam_node on viplab, 192.168.1.6):
#   /synced/cam_4/color/image_raw/compressed  CompressedImage  RELIABLE/VOLATILE/KEEP_LAST(20)
#   /synced/cam_4/color/camera_info           CameraInfo       RELIABLE/VOLATILE/KEEP_LAST(20)
#   /synced/cam_4/depth/image_rect_raw/compressedDepth
#                                                CompressedImage RELIABLE/VOLATILE/KEEP_LAST(20)
#
# Both perception adapters decode the provider's 16UC1 compressedDepth PNG
# directly.  The source frame remains cam_4_depth_optical_frame while RGB is
# cam_4_color_optical_frame, so depth_alignment_validated stays false until a
# reviewed TF/calibration contract exists.  This keeps 3D poses invalid while
# proving and monitoring the real depth transport end to end.
set -eo pipefail

DDS_ENV=/home/hanwae/.config/ros/taskplanner_dds_env.sh
[[ -r "${DDS_ENV}" ]] || { printf 'Missing DDS environment: %s\n' "${DDS_ENV}" >&2; exit 1; }
source "${DDS_ENV}"

ROS_SETUP=/opt/ros/jazzy/setup.bash
HAND_REPO=/home/hanwae/hand_keypoints_ros
HAND_VENV=/home/hanwae/hand_keypoints_ros_ws/.venv
RF_VENV=/home/hanwae/surgical_robot/rfdetr_perception_ros/.venv
TOOL_BUNDLE=/home/hanwae/surgical_robot/tool_detection_component_v1_3_rc1
COORD=/home/hanwae/surgical_robot/coordinator_ws

RGB_TOPIC=/synced/cam_4/color/image_raw/compressed
INFO_TOPIC=/synced/cam_4/color/camera_info
DEPTH_TOPIC=/synced/cam_4/depth/image_rect_raw/compressedDepth

PIDS=()
cleanup() {
  set +e
  for pid in "${PIDS[@]}"; do kill "${pid}" >/dev/null 2>&1; done
  for pid in "${PIDS[@]}"; do wait "${pid}" >/dev/null 2>&1; done
}
trap cleanup EXIT INT TERM

source "${ROS_SETUP}"
source "${HAND_REPO}/ros2_ws/install/setup.bash"
source "${COORD}/install/setup.bash"
set -u

for required in \
  "${TOOL_BUNDLE}/algorithm/model/cam4_rfdetr_seg_small_v1.pth" \
  "${COORD}/scripts/tool_detection_v13_ros_node.py"; do
  [[ -f "${required}" ]] || { printf 'Missing: %s\n' "${required}" >&2; exit 1; }
done

# ---- preflight -------------------------------------------------------
# A name being present in the graph is not a live source. Count authoritative
# publishers for all three required synchronized inputs.
printf 'Checking the live viplab source...\n'
for topic in "${RGB_TOPIC}" "${INFO_TOPIC}" "${DEPTH_TOPIC}"; do
  count=$(ros2 topic info --no-daemon --spin-time 5 "${topic}" 2>/dev/null \
    | sed -n 's/^Publisher count: //p')
  if [[ "${count:-0}" -lt 1 ]]; then
    printf 'No publisher on %s.\n' "${topic}" >&2
    printf 'Check ROS_DOMAIN_ID=%s, RMW=%s, discovery=%s and that viplab is up.\n' \
      "${ROS_DOMAIN_ID}" "${RMW_IMPLEMENTATION}" "${ROS_AUTOMATIC_DISCOVERY_RANGE}" >&2
    exit 1
  fi
  printf '  %-46s publishers=%s\n' "${topic}" "${count}"
done
# ---- perception nodes ------------------------------------------------
# autostart:=false: perception_mode_coordinator owns the turn-taking, since
# the tool and hand models do not fit on the RTX 3060 at the same time.
printf 'Starting CAM4 tool detection on the live stream...\n'
bash -lc "source '${ROS_SETUP}'; source '${DDS_ENV}'; source '${RF_VENV}/bin/activate'; \
  source '${COORD}/install/setup.bash'; export GALLIUM_DRIVER=d3d12; \
  exec python '${COORD}/scripts/tool_detection_v13_ros_node.py' --ros-args \
    -p bundle_root:='${TOOL_BUNDLE}' \
    -p image_topic:='${RGB_TOPIC}' -p image_transport:=compressed \
    -p camera_info_topic:='${INFO_TOPIC}' -p depth_topic:='${DEPTH_TOPIC}' \
    -p depth_transport:=compressed_depth -p depth_alignment_validated:=false \
    -p threshold:=0.5 -p optimize:=true -p publish_overlay:=true \
    -p frame_name:=cam_4_color_optical_frame \
    -p calibration_version:=viplab_cam4_detection_only_v1 \
    -p autostart:=false" &
PIDS+=("$!")

printf 'Starting CAM4 hand keypoints on the live stream...\n'
bash -lc "source '${ROS_SETUP}'; source '${DDS_ENV}'; source '${HAND_VENV}/bin/activate'; \
  source '${HAND_REPO}/ros2_ws/install/setup.bash'; export GALLIUM_DRIVER=d3d12; \
  exec '${HAND_VENV}/bin/python' \
  '${HAND_REPO}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node' \
    --ros-args -p autostart:=false \
    -p color_topic:='${RGB_TOPIC}' -p color_transport:=compressed \
    -p camera_info_topic:='${INFO_TOPIC}' -p depth_topic:='${DEPTH_TOPIC}' \
    -p depth_transport:=compressed_depth -p depth_alignment_validated:=false \
    -p depth_source:=real -p publish_overlay:=true" &
PIDS+=("$!")

printf 'Starting the perception mode coordinator...\n'
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/perception_mode_coordinator" \
  --ros-args -p preload_models_on_startup:=true \
  -p release_gpu_between_modes:=false &
PIDS+=("$!")

printf '\nREADY. Wait for the preload line, then in a second sourced terminal:\n\n'
printf '  source /opt/ros/jazzy/setup.bash\n'
printf '  source %s\n' "${DDS_ENV}"
printf '  source %s/install/setup.bash\n\n' "${COORD}"
printf "  ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'DETECT_TOOL'}\"\n"
printf "  ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'DETECT_HAND'}\"\n"
printf "  ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'STOP'}\"\n\n"
printf 'Then check what the Taskplanner sees of us:\n'
printf '  ros2 topic echo --once --full-length /integration/cv_contract/status std_msgs/msg/String\n\n'
printf 'Ctrl+C ends the run.\n'

wait
