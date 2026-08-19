# Surgical Tool ROS2 Runtime v1.4.0-rc1

이 전달본은 알고리즘 통합 담당자가 보유한 reference rosbag2 또는 동일 계약의 live ROS2
입력을 이용해 surgical-tool class, localization, mask와 평면 제약 pose를 실행하기 위한 묶음이다.
rosbag 파일 자체는 포함하지 않는다.

## 제공 기능

- RF-DETRSegSmall 기반 8종 surgical-tool instance segmentation
- `16UC1; compressedDepth png` decode
- RGB/depth `CameraInfo`와 depth-to-color extrinsic을 이용한 spatial registration
- `depth_scale_m_per_unit` 적용 후 meter 단위 위치 계산
- Mayo stand/tray support plane을 이용한 `PLANAR_4DOF_WITH_NORMAL_PRIOR` pose
- ROS2 `ToolPoseArray`, `ToolObservation2DArray`, overlay, diagnostics, health 발행

현재 quaternion은 평면 위 도구를 위한 constrained orientation이다. 자유공간에서 roll, pitch,
yaw를 모두 독립 관측하는 `FULL_6D`가 아니다.

## 폴더 구조

```text
algorithm/                         비-ROS 알고리즘, 모델, ontology
ros2_ws/src/pnu_surgical_perception/
ros2_ws/src/surgical_perception_msgs/
docs/                              입력·pose·검증 설명
scripts/                           build, run, bag play, validation
validation/                        reference MCAP 검증 도구
MANIFEST.json                      파일별 크기와 SHA-256
SHA256SUMS                         무결성 검증 목록
```

## 기준 입력

기본 설정은 reference bag의 CAM4 동기화 토픽을 사용한다.

```text
/synced/cam_4/color/image_raw/compressed
/synced/cam_4/color/camera_info
/synced/cam_4/depth/image_rect_raw/compressedDepth
/synced/cam_4/depth/camera_info
```

토픽 이름은 `ros2_ws/src/pnu_surgical_perception/config/`의 YAML에서 변경할 수 있다.
입력 계약과 현재 임시 calibration의 범위는 `INPUT_AND_CALIBRATION.md`를 먼저 확인한다.

## 빠른 실행

요구 환경:

- Ubuntu 24.04 / ROS2 Jazzy 기준
- Python 3.12 기준
- NVIDIA GPU 권장
- 현재 검증 조합: PyTorch 2.7.0+cu118, torchvision 0.22.0+cu118, RF-DETR 1.8.3

알고리즘 Python 의존성은 ROS2 노드가 사용하는 Python 환경에 설치해야 한다.

```bash
python3 -m pip install -e ./algorithm
./scripts/build_ros2.sh
```

터미널 1에서 노드를 실행한다.

```bash
./scripts/run_cam4_native_pose.sh
```

터미널 2에서 담당자가 보유한 bag을 재생한다.

```bash
./scripts/play_reference_bag_cam4.sh /absolute/path/to/reference.mcap
```

결과 확인:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 topic echo /surgery/perception/cam4/tool_poses
```

## 기준 출력

```text
/surgery/perception/cam4/tool_poses
/surgery/perception/cam4/observations
/surgery/images/cam4/detection_overlay/compressed
/surgery/perception/rfdetr/diagnostics/json
/surgery/perception/rfdetr/health
```

`ToolPoseArray.header.stamp`는 원본 RGB stamp이며 `header.frame_id`는 기본적으로
`cam_4_color_optical_frame`이다. 위치 단위는 meter, quaternion 순서는 `(x,y,z,w)`다.

## 반드시 교체하거나 확인할 값

기본 YAML은 reference bag에서 복원한 RC 설정이며 다음 값이 아직 production 확정값이 아니다.

- `depth_scale_m_per_unit: 0.001`: bag metadata에 공식 depth unit이 없어 미검증
- support plane: reference bag 첫 프레임에서 추정
- calibration version: reference bag 기준 provisional

이 상태에서는 결과에 `DEPTH_SCALE_UNVERIFIED`, `SUPPORT_PLANE_PROVISIONAL`,
`CALIBRATION_PROVISIONAL` flag가 붙고 validity가 `DEGRADED`로 내려간다. 실제 Mayo stand/tray
설치 calibration으로 교체한 뒤에만 production-valid로 사용한다.

## 현재 알려진 제한

- 첫 reference bag 프레임에서 confidence 0.5 기준 RF-DETR detection은 0개였다.
- RF-DETR 과적합/일반화 문제는 이 runtime 패키징으로 해결되지 않는다.
- Mayo stand와 tray의 높이 또는 기울기가 다르면 view/ROI별 support plane이 필요하다.
- depth-to-color extrinsic은 bag의 1회 메시지에서 추출해 YAML에 기록했다. 현재 노드는 해당
  RealSense 메시지를 직접 subscribe하지 않는다.
- `/tf_static`은 입력 bag에 있지만 현재 출력은 color optical frame 기준이며 world/tag frame으로
  자동 변환하지 않는다.

## 검증

```bash
./scripts/validate_bundle.sh
```

상세 검증 경계는 `docs/20260818_MULTICAM_MCAP_NATIVE_DEPTH_API.md`를 참고한다.
