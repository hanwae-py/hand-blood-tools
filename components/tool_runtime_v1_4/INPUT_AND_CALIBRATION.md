# 입력 및 calibration 계약

## 입력 메시지

| 역할 | ROS type | 요구 사항 |
|---|---|---|
| RGB | `sensor_msgs/msg/CompressedImage` | JPEG/PNG color image, color CameraInfo와 같은 optical frame |
| Native depth | `sensor_msgs/msg/CompressedImage` | `format="16UC1; compressedDepth png"` |
| RGB intrinsic | `sensor_msgs/msg/CameraInfo` | RGB width, height, K, D, frame_id |
| Depth intrinsic | `sensor_msgs/msg/CameraInfo` | Depth width, height, K, D, frame_id |

RGB와 depth는 `maximum_stamp_delta_ns` 이내에서 one-to-one으로 pair한다. 기준 설정은 1 ms이며
reference bag에서 측정한 최근접 stamp 차이는 median 약 0.063 ms다.

## Geometry parameter

```text
depth_scale_m_per_unit
depth_to_color_rotation[9]
depth_to_color_translation_m[3]
support_plane_normal[3]
support_plane_offset_m
```

Registration transform의 방향은 다음과 같다.

```text
P_color = R_color_from_depth @ P_depth + t_color_from_depth
```

`16UC1` raw value는 다음처럼 meter z-depth로 바꾼다.

```text
z_m = raw_depth_unit * depth_scale_m_per_unit
```

reference 설정의 `0.001 m/unit`은 plausibility 확인만 된 값이다. 카메라 provider가 장치의
depth units 설정을 확인하기 전까지 metric calibration이 검증되었다고 표시하면 안 된다.

## Support plane

평면은 color optical frame에서 다음 식을 따른다.

```text
normal dot point + offset_m = 0
```

normal 방향은 tool frame의 `+Z`, 즉 support plane에서 free space 방향이 되도록 설정한다. 실제
설치에서 Mayo stand와 tray가 서로 다른 평면이면 설정을 공유하지 말고 view 또는 ROI별로 분리한다.

## Pose 의미

```text
position_m: mask 내부 depth-valid 표면 관측점 P_obs
+Y: handle/proximal -> working tip
+Z: support plane -> free space
+X: +Y x +Z
pose_mode: PLANAR_4DOF_WITH_NORMAL_PRIOR
dof_observed: [x, y, z, false, false, yaw]
```

`P_obs`는 CAD origin, 중심질량, TCP 또는 grasp point가 아니다.
