# 비-ROS Python API 계약

## Detection

```python
detections = detector.predict(
    image=image,                  # uint8 HxWx3
    color_order="RGB",           # 또는 "BGR", 생략 불가
    confidence_threshold=0.5,
)
```

현재 v1 checkpoint의 검증된 내부 색상 순서는 BGR이다. `color_order`는 호출 배열의 의미이며,
adapter가 checkpoint 계약에 맞게 변환한다. 이 값을 생략하거나 추정하지 않는다.

`DetectionBatch.instances`의 각 원소에는 다음 값이 있다.

- `frame_local_instance_id`: 해당 호출 안에서만 유효
- `canonical_class_id`: 기관 간 교환용 1..8
- `model_class_index`: checkpoint의 0..7
- `class_name`, `class_confidence`
- `bbox_xyxy_px`: source image pixel 좌표
- `mask`: source image와 같은 크기의 bool array

도구가 없는 정상 frame은 빈 `instances`를 반환한다. 입력/모델 오류를 빈 결과로 위장하지 않고
exception을 발생시킨다.

## Pose

```python
result = estimator.estimate(
    detections=detections,
    aligned_depth_m=depth,        # float32/64 HxW, meter, RGB-aligned
    camera=camera,
    support_plane=plane,
    frame_key=frame_key,
)
```

### Native RealSense depth 입력

기준 MCAP처럼 depth가 color와 시간 동기화되었지만 `depth_optical_frame`의 native
`16UC1 compressedDepth`인 경우 다음 경로를 사용한다.

```python
from pnu_surgical_tool import (
    decode_compressed_depth_16uc1,
    DepthToColorRegistrar,
    rigid_transform_from_realsense_extrinsics,
    validate_rgb_depth_timestamps,
)

validate_rgb_depth_timestamps(
    rgb_stamp_ns,
    depth_stamp_ns,
    maximum_delta_ns=1_000_000,  # reference MCAP은 약 0.02~0.063 ms
)
native_depth = decode_compressed_depth_16uc1(
    depth_message.data,
    depth_message.format,
)
color_from_depth = rigid_transform_from_realsense_extrinsics(
    extrinsics.rotation,
    extrinsics.translation,
    source_frame=depth_camera.frame_name,
    target_frame=color_camera.frame_name,
    calibration_version="device-serial-and-calibration-hash",
)
registrar = DepthToColorRegistrar(
    depth_camera=depth_camera,
    color_camera=color_camera,
    color_from_depth=color_from_depth,
)
result, registration = algorithm.detect_and_estimate_from_native_depth(
    image=image,
    native_depth=native_depth,
    depth_registrar=registrar,
    depth_scale_m_per_unit=0.001,  # 장치 설정으로 반드시 확인
    support_plane=plane,
    color_order="BGR",
    frame_key=rgb_stamp_ns,
)
```

`DepthToColorRegistrar`는 depth pixel ray를 생성자에서 한 번만 계산하므로 view/calibration별로
하나를 만들고 재사용한다. 반환 `DepthRegistrationResult.aligned_depth_m`은 color 해상도의
`float32` z-depth이며 단위는 meter, invalid 값은 `NaN`이다. registration은
`P_color = R_color_from_depth @ P_depth + t_color_from_depth`, color distortion projection과
nearest-z buffer를 사용한다.

`decode_compressed_depth_16uc1`은 기준 MCAP의 `16UC1; compressedDepth png`만 지원한다.
`32FC1 compressedDepth`는 inverse-depth codec이므로 동일 decoder에 넣지 않는다.

두 메시지가 `/synced/...`에 있어도 stamp가 완전히 같다는 보장은 없다. 호출 adapter는
`validate_rgb_depth_timestamps`로 허용 오차를 검사하고 one-to-one pairing해야 한다. 입력 제공자가
동일 representative stamp를 부여한다면 exact matching을 사용할 수 있다.

`CameraCalibration`:

```python
CameraCalibration(width, height, k, distortion, frame_name, calibration_version)
```

`SupportPlane`은 `normal @ point + offset_m = 0`을 사용한다. normal은 camera frame에서 free
space 방향이어야 한다.

`ToolInstanceResult`에는 detection 값과 함께 다음 pose 필드가 있다.

- `observation_point_uv_px`
- `observation_point_selection_mode`, `observation_point_boundary_clearance_px`
- `position_m` 또는 invalid 시 `None`
- `orientation_xyzw` 또는 invalid 시 `None`
- `pose_mode`
- `position_valid`, `orientation_valid`, `validity`
- `symmetry_type`, `endpoint_sign_confidence`
- `valid_depth_ratio`, `pose_point_count`, `axis_anisotropy`
- `status_flags`, `invalid_reason`

## Frame join key

이 package는 transport timestamp를 생성하지 않는다. 호출자가 원본 입력 frame의 timestamp/sequence를
`frame_key`로 넘겨 결과와 유지해야 한다. detection과 pose는 같은 `DetectionBatch` instance 순서를
공유한다. frame 간 physical identity가 필요하면 별도의 tracker가 필요하다.

## JSON

`pnu_surgical_tool.types.result_to_dict(result)`는 mask를 제외한 JSON-safe dict를 반환한다.
mask가 필요하면 NumPy bool array를 직접 사용하거나 `rle.encode_uncompressed_coco_rle()`로 lossless
RLE 변환한다.
