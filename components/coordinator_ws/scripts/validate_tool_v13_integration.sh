#!/usr/bin/env bash
set -eo pipefail

TOOL_VENV=/home/hanwae/surgical_robot/rfdetr_perception_ros/.venv
COORD=/home/hanwae/surgical_robot/coordinator_ws
BUNDLE=/home/hanwae/surgical_robot/tool_detection_component_v1_3_rc1

source "${TOOL_VENV}/bin/activate"
source /opt/ros/jazzy/setup.bash
source "${COORD}/install/setup.bash"
set -u

python -c "import builtin_interfaces, surgical_perception_msgs, pnu_surgical_tool; print('imports OK')"

cd "${BUNDLE}"
export PYTHONPATH="algorithm/src:.:${PYTHONPATH:-}"
python validation/validate_static_contract.py
python algorithm/validation/validate_pose_contract.py
python validation/validate_ros_mapping.py

python -m py_compile \
  "${COORD}/scripts/tool_detection_v13_ros_node.py" \
  "${COORD}/scripts/perception_result_receiver.py"

bash -n "${COORD}/scripts/run_signal_driven_perception_test.sh"
bash -n "${COORD}/scripts/run_real_tool_hand_take_turn_test.sh"

printf 'PASS: Tool Component v1.3 coordinator integration validation\n'
