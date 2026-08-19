cmake_minimum_required(VERSION 3.8)
project(surgical_perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ToolObservation2D.msg"
  "msg/ToolObservation2DArray.msg"
  "msg/ToolPose.msg"
  "msg/ToolPoseArray.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
