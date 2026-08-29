# VIPLab top-view landmark-gesture observations for one camera frame.
# This is perception evidence only; it does not authorize a handover or robot
# action. A frame with no detected hands is represented by an empty hands list.

std_msgs/Header header

string model_name
string model_version
string model_asset_sha256

# Positive classes implemented by the named classifier. The separate
# rejection_category is returned for valid but ambiguous/rejected hand shapes.
string[] supported_gestures
string rejection_category

HandGesture[] hands
