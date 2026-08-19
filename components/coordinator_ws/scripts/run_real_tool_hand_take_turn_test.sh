#!/usr/bin/env bash
set -eo pipefail

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
RECEIVER_LOG=/tmp/tool_hand_result_receiver.log

PIDS=()

lifecycle_set() {
  local node_name=$1
  local transition=$2
  local limit_sec=${3:-30}
  local output rc
  for _ in $(seq 1 10); do
    if output=$(timeout "${limit_sec}" ros2 lifecycle set \
        --no-daemon --spin-time 3.0 "/${node_name}" "${transition}" 2>&1); then
      printf '%s\n' "${output}"
      return 0
    else
      rc=$?
      printf '%s\n' "${output}" >&2
      if [[ "${output}" != *"Node not found"* ]]; then
        return "${rc}"
      fi
      sleep 0.5
    fi
  done
  printf 'Lifecycle node remained undiscoverable: %s (%s)\n' \
    "${node_name}" "${transition}" >&2
  return 1
}

cleanup() {
  set +e
  timeout 3 ros2 lifecycle set --no-daemon --spin-time 0.2 \
    /tool_detection_node deactivate >/dev/null 2>&1
  timeout 3 ros2 lifecycle set --no-daemon --spin-time 0.2 \
    /tool_detection_node cleanup >/dev/null 2>&1
  timeout 3 ros2 lifecycle set --no-daemon --spin-time 0.2 \
    /hand_detection_node deactivate >/dev/null 2>&1
  timeout 3 ros2 lifecycle set --no-daemon --spin-time 0.2 \
    /hand_detection_node cleanup >/dev/null 2>&1
  for pid in "${PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" >/dev/null 2>&1
  done
}
trap cleanup EXIT INT TERM

source "${ROS_SETUP}"
source "${HAND_REPO}/ros2_ws/install/setup.bash"
source "${COORD}/install/setup.bash"
set -u

for required in "${VIDEO}" "${DEPTH_H5}" "${CALIB}" \
  "${TOOL_BUNDLE}/algorithm/model/cam4_rfdetr_seg_small_v1.pth" \
  "${COORD}/scripts/tool_detection_v13_ros_node.py"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Missing required file: %s\n' "${required}" >&2
    exit 1
  fi
done

printf '\n[1/6] Starting the shared looping RGB-video publisher...\n'
bash -lc "source '${ROS_SETUP}'; source '${HAND_VENV}/bin/activate'; \
  source '${HAND_REPO}/ros2_ws/install/setup.bash'; \
  exec '${HAND_REPO}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/fake_camera_publisher' --ros-args \
    -p rgb_path:='${VIDEO}' \
    -p depth_h5_path:='${DEPTH_H5}' \
    -p calib_path:='${CALIB}' \
    -p cam_key:=cam_4 \
    -p rate_hz:=15.0 \
    -p loop:=true \
    -p preload_depth:=false" &
PIDS+=("$!")

printf '[2/6] Starting both lifecycle detectors in UNCONFIGURED state...\n'
bash -lc "source '${ROS_SETUP}'; source '${RF_VENV}/bin/activate'; \
  source '${COORD}/install/setup.bash'; export GALLIUM_DRIVER=d3d12; \
  exec python '${COORD}/scripts/tool_detection_v13_ros_node.py' --ros-args \
    -p bundle_root:='${TOOL_BUNDLE}' \
    -p threshold:=0.5 \
    -p optimize:=true \
    -p publish_overlay:=true \
    -p autostart:=false" &
PIDS+=("$!")

bash -lc "source '${ROS_SETUP}'; source '${HAND_VENV}/bin/activate'; \
  source '${HAND_REPO}/ros2_ws/install/setup.bash'; \
  export GALLIUM_DRIVER=d3d12; \
  exec '${HAND_REPO}/ros2_ws/install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node' --ros-args \
    -p autostart:=false \
    -p depth_source:=real \
    -p publish_overlay:=false" &
PIDS+=("$!")

python3 /home/hanwae/surgical_robot/coordinator_ws/scripts/perception_result_receiver.py \
  --ros-args -p timeout_sec:=300.0 > >(tee "${RECEIVER_LOG}") 2>&1 &
PIDS+=("$!")

wait_for_lifecycle_node() {
  local node_name=$1
  for _ in $(seq 1 60); do
    if ros2 service list --no-daemon --spin-time 0.2 2>/dev/null \
        | grep -qx "/${node_name}/change_state"; then
      return 0
    fi
    sleep 0.5
  done
  printf 'Lifecycle node did not appear: %s\n' "${node_name}" >&2
  return 1
}

wait_for_receiver_line() {
  local pattern=$1
  local limit_sec=$2
  for _ in $(seq 1 $((limit_sec * 2))); do
    if [[ -f "${RECEIVER_LOG}" ]] && grep -q "${pattern}" "${RECEIVER_LOG}"; then
      return 0
    fi
    sleep 0.5
  done
  printf 'Timed out waiting for receiver line: %s\n' "${pattern}" >&2
  return 1
}

wait_for_lifecycle_node tool_detection_node
wait_for_lifecycle_node hand_detection_node

printf '\n[3/6] TOOL TURN: loading/optimizing RF-DETR, then activating it...\n'
lifecycle_set tool_detection_node configure 180
lifecycle_set tool_detection_node activate 30
printf 'Waiting for a real typed v1.3 Tool observation message...\n'
wait_for_receiver_line 'RECEIVED TOOL RESULT' 60
printf 'Tool result was received by a ROS subscriber.\n'

printf '[4/6] Ending tool turn and releasing RF-DETR GPU memory...\n'
lifecycle_set tool_detection_node deactivate 30
lifecycle_set tool_detection_node cleanup 30

printf '\n[5/6] HAND TURN: loading MediaPipe with real HDF5 depth, then activating it...\n'
lifecycle_set hand_detection_node configure 180
lifecycle_set hand_detection_node activate 30
printf 'Waiting for a real typed hand-keypoint message...\n'
wait_for_receiver_line 'RECEIVED HAND RESULT' 90
printf 'Hand result was received by a ROS subscriber.\n'

printf '[6/6] Ending hand turn and releasing its GPU memory...\n'
lifecycle_set hand_detection_node deactivate 30
lifecycle_set hand_detection_node cleanup 30

for _ in $(seq 1 20); do
  if grep -q 'SUCCESS: downstream node received both real detector outputs' \
      "${RECEIVER_LOG}"; then
    printf '\nSUCCESS: tool and hand ran sequentially on the same video, and a downstream node received both outputs.\n'
    exit 0
  fi
  sleep 0.5
done

printf 'Both topic checks passed, but the summary receiver did not report completion.\n' >&2
exit 1
