# One camera frame's hand-detection result. Published on hand/keypoints
# by hand_keypoint_ros/hand_detection_node -- combines what the original
# flow diagram split across "section 2" (detection) and "section 3"
# (2D->3D conversion): this message already carries both joints_2d and
# joints_3d (+ palm_6d) together, so no separate 3D-conversion node is
# needed downstream.
std_msgs/Header header
string depth_source   # "real" (RealSense-style aligned depth) or "mono" (Depth-Anything V2 fallback)
Hand[] hands
