# 외부 전달 전 체크리스트

## 이번 폴더에 완료된 것

- [x] 비-ROS importable Python package
- [x] RF-DETRSegSmall fine-tuned checkpoint와 SHA256
- [x] 8-class canonical ontology
- [x] detection class/bbox/mask API
- [x] aligned depth 기반 P_obs와 constrained quaternion pose API
- [x] validity, pose mode, C2 symmetry, 품질 정보
- [x] detection-only / detection+pose / NUC adapter 예제
- [x] 합성 pose test와 내부 frame-99 회귀 test
- [x] CAM4 내부 realtime 진단 결과
- [x] 임상 원본·전체 dataset·ROS·Hand/Blood 제외

## KAIST에 먼저 확인할 것

- [ ] NUC OS/CPU/RAM/GPU/VRAM/driver/CUDA/Python
- [ ] 목표 해상도, CAM4 Mayo/CAM3 tray별 FPS와 최대 latency
- [ ] 입력 array가 RGB인지 BGR인지
- [ ] RGB-aligned depth의 dtype, meter 단위, invalid 표현
- [ ] CAM4 Mayo와 CAM3 tray 각각의 공식 ROI, K/D, frame name, support plane
- [ ] canonical class ID 1..8을 그대로 받을지 mapping이 필요한지
- [ ] `P_obs`를 그대로 사용할지 CAD/TCP/grasp offset이 별도로 필요한지
- [ ] `DEGRADED/INVALID`, C2 symmetry를 기존 알고리즘이 처리하는 방법

## final release 승격 전

- [ ] 데이터/IRB/외부 반출 승인
- [ ] RF-DETR/base weights 및 dependency 라이선스 검토·고지
- [ ] NUC clean install 및 checksum 검증
- [ ] NUC detection/pose acceptance와 p50/p95/FPS 기록
- [ ] 실제 CAM4 synchronized input 정확도 검증
- [ ] CAM3 8월 tray 영상 수령 및 ROI 보정·검증
- [ ] 수신자/담당자/버전/전달일 기록

ROS 통신은 현장 세팅 이후 별도 통합 전달본에서 다룬다.
