# Multicam MCAP native-depth API validation — 2026-08-18

기준 파일:

```text
multicam_viplab_only_30s_20260814_134233.staging_0.mcap
```

## 입력 판정

`/synced/cam_1..4` RGB와 depth는 약 15Hz이고 대부분 0.1ms 이내로 시간 pairing되어 있다.
그러나 depth는 `16UC1; compressedDepth png`, `*_depth_optical_frame`, depth intrinsic을 사용한다.
color는 별도의 `*_color_optical_frame`, color intrinsic/distortion을 사용하며 카메라마다
`depth_to_color` extrinsic이 1회 기록되어 있다. 따라서 `/synced`는 시간 pairing을 의미하며
depth payload 자체가 color pixel로 registration됐다는 뜻이 아니다.

## 추가 API

- `decode_compressed_depth_16uc1`: ROS compressedDepth PNG 복원
- `validate_rgb_depth_timestamps`: tolerance 기반 pair 검사
- `RigidTransform`: `P_color = R @ P_depth + t` 계약
- `DepthToColorRegistrar`: ray cache, unprojection, color distortion projection, nearest-z buffer
- `SurgicalToolAlgorithm.detect_and_estimate_from_native_depth`: registration과 기존 pose core 연결

기존 `aligned_depth_m` API는 그대로 유지한다.

## CAM4 첫 프레임 검증

- RGB-depth stamp delta: 62,012ns
- depth→color translation norm: 59.283mm
- decoded depth: `uint16`, 1280×720
- source valid pixels: 729,251
- projected points: 729,064
- RGB-aligned valid pixels: 676,054 (73.36%)
- z-buffer collisions: 53,010
- ray-cache 생성 후 warm registration median: 약 33ms (현재 CPU 환경)

Depth scale `0.001m/unit`은 값 분포상 타당하지만 bag metadata에 `depth_units`가 없으므로 실제 장치
설정에서 최종 확인해야 한다.

## 재검증 명령

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=algorithm/src:$PYTHONPATH \
python3 validation/validate_multicam_mcap_native_depth.py \
  multicam_viplab_only_30s_20260814_134233.staging_0.mcap \
  --camera cam_4 --depth-scale 0.001 --maximum-delta-ns 1000000
```

현재 exact-equality stamp matcher는 이 입력과 호환되지 않는다. Host adapter는 1ms 이내 one-to-one
pairing을 사용하거나 upstream이 pair에 공통 representative stamp를 부여해야 한다.
