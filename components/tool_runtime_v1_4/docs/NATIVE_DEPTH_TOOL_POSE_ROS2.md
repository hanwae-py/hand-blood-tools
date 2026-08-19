# Native-depth Surgical Tool Pose ROS 2 경로

이 구현은 미완성 XLSM이 아니라 reference MCAP의 실제 ROS 2 입력을 기준으로 한다.

## 입력

기본 CAM4 입력은 다음 네 토픽이다.

```text
/synced/cam_4/color/image_raw/compressed
/synced/cam_4/color/camera_info
/synced/cam_4/depth/image_rect_raw/compressedDepth
/synced/cam_4/depth/camera_info
```

Depth는 `16UC1; compressedDepth png`이며 depth optical frame에 있다. 노드는 RGB와 depth를
`maximum_stamp_delta_ns` 이내에서 one-to-one pairing한 뒤 다음 calibration parameter를 이용해
color optical frame으로 registration한다.

```text
depth_scale_m_per_unit
depth_to_color_rotation[9]
depth_to_color_translation_m[3]
support_plane_normal[3]
support_plane_offset_m
```

`CameraInfo`에는 센서 내부 intrinsic만 있으므로 depth-to-color extrinsic과 depth scale을 대신할 수
없다. 값이 없거나 유한하지 않으면 노드는 시작하지 않는다.

## 처리와 출력

```text
Compressed RGB ──► BGR decode ──► RF-DETR instance segmentation
16UC1 depth ──► PNG decode ──► scale 적용 ──► depth-to-color registration
mask + RGB-aligned metric depth + support plane
  ──► P_obs position[m] + constrained quaternion
  ──► ToolPoseArray + ToolObservation2DArray + overlay
```

출력 토픽은 다음과 같다.

```text
/surgery/perception/cam4/tool_poses
/surgery/perception/cam4/observations
/surgery/images/cam4/detection_overlay/compressed
/surgery/perception/rfdetr/diagnostics/json
/surgery/perception/rfdetr/health
```

현재 알고리즘은 quaternion을 전달하지만 unconstrained full 6DoF가 아니다.

```text
pose_mode = PLANAR_4DOF_WITH_NORMAL_PRIOR
dof_observed = [x, y, z, false, false, yaw]
```

Roll/pitch는 실제 관측값이 아니라 support-plane normal에서 정해진다. Full 6D로 발행하지 않는다.

## Reference MCAP 실행

패키지 루트에서 빌드:

```bash
./scripts/build_ros2.sh
```

노드:

```bash
./scripts/run_cam4_native_pose.sh
```

Reference bag의 CAM4 입력만 재생:

```bash
./scripts/play_reference_bag_cam4.sh /absolute/path/to/reference.mcap
```

## Reference 설정의 안전 상태

`cam4_reference_mcap_native_pose.yaml`의 다음 값은 reference bag에서 복원하거나 추정한 값이다.

- depth-to-color extrinsic: bag의 `/synced/cam_4/extrinsics/depth_to_color`
- depth scale `0.001 m/unit`: 값 분포상 타당하지만 metadata에 공식 값 없음
- support plane: 첫 frame의 blue/depth-valid point RANSAC 결과

따라서 reference 설정에서는 pose에 다음 flag를 추가하고 `VALIDITY_DEGRADED`로 낮춘다.

```text
DEPTH_SCALE_UNVERIFIED
SUPPORT_PLANE_PROVISIONAL
CALIBRATION_PROVISIONAL
```

실제 설치 calibration으로 값을 교체하고 `depth_scale_verified=true`로 설정하기 전에는 health의
`ready`와 `metric_calibration_verified`가 true가 되지 않는다.
