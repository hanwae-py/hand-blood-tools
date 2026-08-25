# KAIST NUC 설치 메모

## 설치 전 회신이 필요한 사양

- NUC 모델, CPU, RAM
- NVIDIA GPU 모델과 VRAM(없으면 없다고 명시)
- OS, Python, NVIDIA driver, CUDA
- 입력 해상도와 요구 FPS
- CAM4 Mayo/CAM3 tray RGB 색상 순서와 카메라별 ROI polygon
- 각 view의 RGB-aligned metric depth 제공 가능 여부
- 각 view의 공식 color calibration과 support-plane 제공 방식

현재 reference 환경은 Python 3.12.12, PyTorch 2.7.0+cu118, torchvision
0.22.0+cu118, RF-DETR 1.8.3이다. 이 버전을 무조건 설치하라는 뜻이 아니라 현재 checkpoint를
검증한 기준이다. NUC driver가 CUDA 11.8 wheel과 호환되지 않으면 해당 driver에 맞는 PyTorch
조합을 먼저 결정해야 한다.

## 실행 모드

- NVIDIA CUDA GPU: `optimize=True`, FP16, JIT 권장
- 첫 최적화 호출에는 compile 시간이 추가된다. 실제 latency 측정 전 warm-up을 수행한다.
- CPU: `optimize=False`로 기능 smoke test는 가능하지만 실시간성을 보장하지 않는다.

## acceptance 기록

최종 NUC에서 아래를 JSON 또는 문서로 남긴다.

- package/model checksum
- 정확한 Python/PyTorch/torchvision/RF-DETR/OpenCV 버전
- GPU/driver/CUDA 정보
- 입력 해상도·view·instance 수
- warm-up 횟수, 측정 frame 수
- inference/pose/end-to-end p50·p95·FPS
- peak GPU memory, RAM
- detection class/mask sanity와 pose validity 분포
