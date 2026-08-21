# surgical_perception_msgs

부산대학교 컴퓨터 비전연구실 surgical-tool perception의 ROS 2 Jazzy interface package다.

```bash
cd <bundle-root>/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select surgical_perception_msgs
source install/setup.bash

ros2 interface show surgical_perception_msgs/msg/ToolPoseArray
ros2 interface show surgical_perception_msgs/msg/ToolObservation2DArray
```

- `ToolPoseArray`: instance별 class와 metric pose, pose mode, quality, validity.
  동일 type을 `/perception/cam_4/tool/poses`와
  `/perception/tray/tool/poses`에서 사용하되 배열마다 한 view만 담는다.
- 위치만 필요한 consumer는 `class_name`과 `pose.position`을 읽는다.
- 전체 pose가 필요한 consumer는 `class_name`과 `pose` 전체를 읽는다.
- `pose.position`은 mask centroid가 아니라 tool 위에서 선택한 3D 관측점 `P_obs`다.
- `ToolObservation2DArray`: lossless instance mask, bbox, `P_obs`의 2D projection과 선택 근거
- 두 배열은 `header`, `sequence`, `observation_id`, `frame_local_instance_id`로 연결한다.
- enum의 zero 값은 의도적으로 invalid다.
- `/surgery/perception/surgical_tools/states`는 향후 common-frame cross-view fusion용으로
  예약하며 v1 publisher가 발행하지 않는다.

외부 계약의 정본은
[`PNU_CVLAB_SURGICAL_TOOL_INTERFACE_CONTRACT_V1.md`](../../../PNU_CVLAB_SURGICAL_TOOL_INTERFACE_CONTRACT_V1.md)를
참조한다.
