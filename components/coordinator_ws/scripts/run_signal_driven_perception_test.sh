#!/usr/bin/env bash
set -eo pipefail

DDS_ENV=/home/hanwae/.config/ros/taskplanner_dds_env.sh
[[ -r "${DDS_ENV}" ]] || { printf 'Missing DDS environment: %s\n' "${DDS_ENV}" >&2; exit 1; }
source "${DDS_ENV}"

ROS_SETUP=/opt/ros/jazzy/setup.bash
HAND_REPO=/home/hanwae/hand_keypoints_ros
HAND_VENV=/home/hanwae/hand_keypoints_ros_ws/.venv
RF_REPO=/home/hanwae/surgical_robot/rfdetr_perception_ros
RF_VENV=${RF_REPO}/.venv
TOOL_BUNDLE=/home/hanwae/surgical_robot/tool_detection_component_v1_3_rc1
COORD=/home/hanwae/surgical_robot/coordinator_ws
DATASET=/home/hanwae/surgical_robot/test_data/cam4_0618
VIDEO=${DATASET}/rgb/0618_2_cam4_rgb_11m11s_to_11m41s.avi
DEPTH_H5=${DATASET}/depth_raw/0618_2_cam4_depth_raw_11m11s_to_11m41s.h5
CALIB=${DATASET}/calibration/0618_2_calibration_data.json

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

for required in "${VIDEO}" "${DEPTH_H5}" "${CALIB}" \
  "${TOOL_BUNDLE}/algorithm/model/cam4_rfdetr_seg_small_v1.pth" \
  "${COORD}/scripts/tool_detection_v13_ros_node.py"; do
  [[ -f "${required}" ]] || { printf 'Missing: %s\n' "${required}" >&2; exit 1; }
done

printf 'Starting shared 0618 RGB + real-depth publisher...\n'
bash -lc "source '${ROS_SETUP}'; source '${HAND_VENV}/bin/activate'; \
  source '${HAND_REPO}/ros2_ws/install/setup.bash'; \
  exec '${HAND_VENV}/bin/python' '${HAND_REPO}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/fake_camera_publisher' \
    --ros-args -p rgb_path:='${VIDEO}' -p depth_h5_path:='${DEPTH_H5}' \
    -p calib_path:='${CALIB}' -p cam_key:=cam_4 -p rate_hz:=15.0 \
    -p loop:=true -p preload_depth:=false -p reliable_output:=true" &
PIDS+=("$!")

printf 'Starting real tool and hand nodes for startup preloading...\n'
bash -lc "source '${ROS_SETUP}'; source '${RF_VENV}/bin/activate'; \
  source '${COORD}/install/setup.bash'; export GALLIUM_DRIVER=d3d12; \
  exec python '${COORD}/scripts/tool_detection_v13_ros_node.py' --ros-args \
    -p bundle_root:='${TOOL_BUNDLE}' \
    -p threshold:=0.5 -p optimize:=true -p publish_overlay:=true \
    -p autostart:=false" &
PIDS+=("$!")

bash -lc "source '${ROS_SETUP}'; source '${HAND_VENV}/bin/activate'; \
  source '${HAND_REPO}/ros2_ws/install/setup.bash'; export GALLIUM_DRIVER=d3d12; \
  exec '${HAND_VENV}/bin/python' '${HAND_REPO}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node' \
    --ros-args -p autostart:=false -p depth_source:=real \
    -p publish_overlay:=true" &
PIDS+=("$!")

printf 'Starting blood stub (real blood weights are not available)...\n'
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/stub_detector" \
  --ros-args -r __node:=blood_detection_node \
  -p result_topic:=/surgery/perception/cam4/blood_target_pose \
  -p target_xyz:="[0.02, 0.12, 0.45]" \
  -p detection_delay_sec:=1.5 -p fake_model_load_sec:=2.0 &
PIDS+=("$!")

printf 'Starting signal coordinator and downstream result receiver...\n'
"${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/perception_mode_coordinator" \
  --ros-args -p preload_models_on_startup:=true \
  -p release_gpu_between_modes:=false &
PIDS+=("$!")

python3 "${COORD}/scripts/perception_result_receiver.py" \
  --ros-args -p timeout_sec:=1800.0 &
PIDS+=("$!")

printf '\nREADY. In a second sourced WSL terminal, send in order:\n\n'
printf "ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'DETECT_TOOL'}\"\n"
printf "ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'DETECT_HAND'}\"\n"
printf "ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'DETECT_BLOOD'}\"\n"
printf "ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String \"{data: 'STOP'}\"\n\n"
printf 'Wait until all detector models preloaded appears before the first command.\n'
printf 'Then wait for each ACTIVE/result line before sending the next command. Ctrl+C ends the test.\n'

wait
