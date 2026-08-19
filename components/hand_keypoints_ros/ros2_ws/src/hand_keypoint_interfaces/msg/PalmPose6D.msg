# Palm 6D pose, v2 formula (see hand_keypoints_core.palm_frame_v2 in the
# hand_keypoint_ros package):
#   translation = midpoint(wrist, middle_MCP)
#   X axis = wrist -> middle_MCP
#   Y axis = across the palm (Gram-Schmidt against X)
#   Z axis = palm normal (X cross Y)
# Only meaningful when the parent Hand.has_palm_6d is true -- computing it
# requires joints {wrist, index_MCP, middle_MCP, pinky_MCP} to all have
# valid depth.
geometry_msgs/Point translation        # metres, camera optical frame
geometry_msgs/Quaternion orientation   # same rotation as rotation_matrix, Hamilton convention
float32[9] rotation_matrix             # row-major 3x3: columns are (X, Y, Z) axes above
