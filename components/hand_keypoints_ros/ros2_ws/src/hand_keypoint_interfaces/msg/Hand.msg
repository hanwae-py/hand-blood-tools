# One detected hand within a single camera frame.
#
# Joint order for joints_2d / joints_3d / kp_scores / kp_valid_depth is
# fixed (MediaPipe Hand Landmarker order), 21 entries:
#   0            wrist
#   1,2,3,4      thumb_CMC, thumb_MCP, thumb_IP, thumb_TIP
#   5,6,7,8      index_MCP, index_PIP, index_DIP, index_TIP
#   9,10,11,12   middle_MCP, middle_PIP, middle_DIP, middle_TIP
#   13,14,15,16  ring_MCP, ring_PIP, ring_DIP, ring_TIP
#   17,18,19,20  pinky_MCP, pinky_PIP, pinky_DIP, pinky_TIP

int32 hand_index

bool has_handedness
string handedness_label      # "Left" or "Right" -- valid only if has_handedness
float32 handedness_score     # valid only if has_handedness

Point2D[21] joints_2d              # pixel coordinates
geometry_msgs/Point[21] joints_3d  # metres, camera optical frame; zeroed where kp_valid_depth is false
float32[21] kp_scores              # per-joint confidence (MediaPipe: always 1.0 -- reserved for future detectors)
bool[21] kp_valid_depth            # per-joint: did the depth lookup succeed

bool has_palm_6d
PalmPose6D palm_6d           # valid only if has_palm_6d
