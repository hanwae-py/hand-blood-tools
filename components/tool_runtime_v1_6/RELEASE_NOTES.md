# Release notes — v1.4.0-rc1 (2026-08-18)

- Reference MCAP의 native `16UC1 compressedDepth` 입력 경로 추가
- RGB/depth approximate timestamp pairing 추가
- Depth decode, metric scale, depth-to-color projection 및 z-buffer registration 추가
- RF-DETR + planar pose를 실행하는 ROS2 node 추가
- Typed `ToolPoseArray`와 `ToolObservation2DArray` mapping 추가
- 미검증 depth scale/support plane/calibration에 대한 fail-closed/degraded 상태 추가
- 로컬 프로젝트 절대경로를 제거하고 bundle-relative 실행 스크립트 제공

모델 checkpoint 자체는 기존 CAM4 RF-DETR 모델과 동일하며 과적합 문제는 미해결 상태다.
