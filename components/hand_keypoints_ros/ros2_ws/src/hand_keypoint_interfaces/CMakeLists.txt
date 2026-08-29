cmake_minimum_required(VERSION 3.8)
project(hand_keypoint_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Point2D.msg"
  "msg/PalmPose6D.msg"
  "msg/Hand.msg"
  "msg/HandKeypoints.msg"
  "msg/HandGesture.msg"
  "msg/HandGestureArray.msg"
  "msg/HandFacing.msg"
  "msg/HandFacingArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
