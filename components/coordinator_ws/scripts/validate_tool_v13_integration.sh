#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/config/system.env"
TOOL_VENV="$(cd "$(dirname "${RFDETR_PYTHON}")/.." && pwd)"
COORD="${ROOT}/components/coordinator_ws"
BUNDLE="${TOOL_V13_BUNDLE:-${HOME}/models/tool_detection_component_v1_3_rc1}"

source "${TOOL_VENV}/bin/activate"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
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
