# 현재 내부 성능 근거

## 순수 알고리즘 core 진단

2026-08-11/12, NVIDIA GeForce RTX 3060, CAM4 archived video 30 frames, FP16/JIT:

- sustained: 28.76 FPS
- RF-DETR inference mean: 21.55 ms
- pose core mean: 10.18 ms
- decode + inference + pose mean: 34.69 ms
- end-to-end core p95: 39.50 ms
- 15 Hz gate: pass
- strict 30 Hz gate: fail

이 결과는 고정된 frame-99 RGB-aligned point geometry를 재사용해 소비자 core 비용을 측정한
속도 진단이다. video 전체 pose 정확도 검증이 아니며 raw registration, plane 재추정, tracking,
JSON/RLE serialization과 통신은 제외했다.

## 전달 API 회귀

전달 API의 입력 계약은 organized XYZ point가 아니라 `aligned_depth_m + color K/D`다. historical
frame-99의 z-buffered organized point 정본과 이 API의 재투영 P_obs를 같은 클래스 내 최근접으로
비교했을 때 최대 차이는 8.59 mm였다(허용치 10 mm). 이 수치는 방법 정확도나 실제 ground truth
오차가 아니라 두 depth 표현 방식 간 회귀 차이다. 실제 demo calibration과 aligned-depth 의미가
확정되면 independent pose reference로 다시 acceptance해야 한다.

## 해석

NUC의 GPU/driver/입력 instance 수가 다르면 수치가 그대로 재현되지 않는다. CAM4+tray 동시 처리와
Hand/Blood 통합 후에는 shared-resource benchmark를 새로 수행해야 한다.
