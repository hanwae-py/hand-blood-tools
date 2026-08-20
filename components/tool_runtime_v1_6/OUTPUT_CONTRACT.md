# ROS2 출력 계약 요약

## Canonical topics

| Topic | Type | 의미 |
|---|---|---|
| `/surgery/perception/cam4/tool_poses` | `surgical_perception_msgs/msg/ToolPoseArray` | class와 metric constrained pose |
| `/surgery/perception/cam4/observations` | `surgical_perception_msgs/msg/ToolObservation2DArray` | bbox, mask RLE, 2D 관측점 |
| `/surgery/images/cam4/detection_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | 사람이 확인하는 overlay |
| `/surgery/images/cam4/pose_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | pose 축 디버그 overlay |
| `/surgery/perception/rfdetr/diagnostics/json` | `std_msgs/msg/String` | 프레임 처리 진단 JSON |
| `/surgery/perception/rfdetr/health` | `std_msgs/msg/String` | readiness와 오류 상태 JSON |

`tool_poses`와 `observations`는 reliable/volatile QoS를 사용한다. Overlay는 sensor-data QoS다.

## Pose 필드

- `header.stamp`: paired RGB source stamp
- `header.frame_id`: 기본 `cam_4_color_optical_frame`
- `pose.position`: meter
- `pose.orientation`: quaternion `(x,y,z,w)`
- `pose_mode`: `POSE_MODE_PLANAR_4DOF_WITH_NORMAL_PRIOR`
- `dof_observed`: `[true,true,true,false,false,true]`
- `validity`: consumer가 사용 전에 반드시 검사
- `status_flags`: calibration과 pose 품질 상태

빈 검출 프레임은 array message 자체는 발행하되 `tools`와 `instances`가 빈 배열이다. 통신 누락과
"검출 결과 없음"을 구분해야 한다.
