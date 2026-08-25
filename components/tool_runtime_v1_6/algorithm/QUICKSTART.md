# Quick start

## 1. 설치

Python 3.12와 NVIDIA CUDA GPU 환경을 권장한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch CUDA wheel은 NUC의 NVIDIA driver/CUDA 조건에 맞게 먼저 설치하는 편이 안전하다.
정확한 기준 환경은 `environment/requirements-reference-cu118.txt`를 참고한다.

## 2. 체크섬

이 알고리즘은 ROS2 runtime bundle에 포함되어 전달된다. bundle 최상위 디렉터리에서 다음을
실행한다.

```bash
./scripts/validate_bundle.sh
```

## 3. pose-only 계약 검증

```bash
python validation/validate_pose_contract.py
```

이 검증은 임상 영상 없이 합성 mask/depth로 P_obs, quaternion, validity, RLE을 확인한다.

기준 MCAP과 같은 native compressedDepth 입력의 decoder, timestamp tolerance, depth-to-color
projection과 z-buffer는 다음으로 검증한다.

```bash
PYTHONPATH=src python3 validation/validate_native_depth_registration.py
```

## 4. RF-DETR detection 실행

```bash
python examples/run_detection.py \
  --image /path/to/input.jpg \
  --checkpoint /path/to/xlarge_best.pth \
  --model-size xlarge \
  --color-order BGR \
  --output-json detection.json \
  --output-overlay detection_overlay.jpg
```

OpenCV `imread()` 결과는 BGR이므로 위 예제는 `BGR`을 사용한다. 라이브 파이프라인이 RGB를
제공하면 반드시 `--color-order RGB`로 바꾼다. adapter가 Small의 기존 BGR 계약 또는
Medium/Large/XLarge의 RGB 계약에 맞춰 정확히 한 번 변환한다.

## 5. detection + pose

```bash
python examples/run_detection_and_pose.py \
  --image /path/to/input.jpg \
  --checkpoint /path/to/xlarge_best.pth \
  --model-size xlarge \
  --aligned-depth-npy /path/to/aligned_depth_m.npy \
  --camera-pose-config /path/to/camera_pose_config.json \
  --color-order BGR \
  --output-json tool_result.json
```

`config/camera_pose_config.template.json`의 `null` 항목은 실제 카메라별 공식 값으로 채워야 한다.
샘플을 임의 숫자로 채워 production에 사용하면 안 된다.

## 6. ROI와 temporal class smoothing

ROS runtime에서는 `workspace_roi_*`와 `temporal_class_*` parameter로 활성화한다. CAM4는 Mayo
stand camera이며 `cam4_live_native_pose.yaml`에 2026-08-14 기준 provisional polygon이 있다.
CAM3는 8월 tray camera지만 영상이 아직 전달되지 않아 `workspace_roi_enabled: false`다.

ROI polygon은 source image normalized `(x, y)`를 평평한 배열로 기록한다. 카메라나 작업대가
이동하면 반드시 다시 보정한다. 이 필터는 inference 후 적용되므로 accepted mask의 geometry는
보존하지만 RF-DETR inference 시간은 줄이지 않는다.
