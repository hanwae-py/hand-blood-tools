#!/usr/bin/env bash
# Temporary Blood launcher. Replace this stub when the real Blood source/config arrives.
set -euo pipefail
COORD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source "${COORD}/install/setup.bash"
set -u
exec "${COORD}/install/surgical_task_coordinator/lib/surgical_task_coordinator/stub_detector" \
  --ros-args -r __node:=blood_detection_node \
  -p result_topic:=/surgery/perception/cam4/blood_target_pose \
  -p target_xyz:="[0.02, 0.12, 0.45]" \
  -p detection_delay_sec:=1.5 -p fake_model_load_sec:=2.0 "$@"
