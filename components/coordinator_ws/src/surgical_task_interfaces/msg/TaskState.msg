# One snapshot of the surgical task coordinator's state machine.
# Published on /surgery/task/state on every transition, plus at 1 Hz as a
# heartbeat so downstream nodes can detect a dead coordinator.
#
# The three perception algorithms (tool / hand / blood) deliberately never
# run at the same time -- see surgical_task_coordinator/task_coordinator.py.
# This message is how everyone else observes whose turn it currently is.
std_msgs/Header header

# --- state constants. Keep in sync with task_coordinator.py's State class. ---
string IDLE=IDLE
string TOOL_DETECTION=TOOL_DETECTION
string ROBOT_GRASP_TOOL=ROBOT_GRASP_TOOL
string HAND_DETECTION=HAND_DETECTION
string ROBOT_HANDOVER=ROBOT_HANDOVER
string BLOOD_DETECTION=BLOOD_DETECTION
string ROBOT_SUCTION=ROBOT_SUCTION
string FAULT=FAULT

string state

# Lifecycle node name of the detector that is ACTIVE right now, or '' when
# none is. Exactly one of the three is active at a time, by construction.
string active_detector

# Human-readable progress / failure reason. Not machine-parsed.
string detail

builtin_interfaces/Time state_entered_at
