# Surgical Tool ROS2 Runtime v1.6 - 호환성 변형

이 디렉터리는 로컬 통합 후보입니다. v1.6 검출기 릴리스를 현재 저장소의
Tool 코디네이터 연동 동작과 결합한 것이며, 변경되지 않은 공급사 v1.6 아카이브가 아닙니다.

## 현재 런타임에서 유지한 동작

- Tool/Hand/Blood 순차 실행을 위한 processing_enabled 및 processing_gate_topic
- require_depth가 false일 때의 RGB 전용 2D 관측 발행
- ToolObservation2D의 observation_point_depth_m 필드

위 항목은 현재 perception coordinator 및 하위 ROS 소비자와의 호환성을 유지합니다.

## 포함한 v1.6 변경

- 검출기 체크포인트: algorithm/model/cam4_rfdetr_seg_small_regular_resume_e13_best.pth
- 다운로드: [Google Drive checkpoint](https://drive.google.com/file/d/13JW_AVPgiJZ_XdWmOReSeSCg2d35wHSC/view?usp=drive_link). clone 후 위 경로에 파일을 배치합니다.
- 기본 confidence threshold: 0.30
- class-agnostic bounding-box NMS: 기본 활성화, IoU 0.80
- 새 pose-axis 디버그 오버레이: /surgery/images/cam4/pose_overlay/compressed

입력 토픽, native 16UC1 compressedDepth 디코딩, depth-to-color 등록,
그리고 constrained planar pose 의미는 현재 런타임과 호환됩니다.

## 빌드

이 디렉터리에서 실행:

    bash scripts/build_ros2.sh

## 실행

저장소 최상위에서 실행:

    bash scripts/run_tool_v16.sh

v1.4 런타임과 실행 스크립트는 변경하지 않았습니다. v1.6 선택 전에는
전체 시스템에서 MCAP/live 입력과 coordinator 전환을 함께 검증해야 합니다.

## 로컬 검증 완료

- ROS workspace 빌드: surgical_perception_msgs 및 pnu_surgical_perception
- ROS 단위 테스트: 10개 통과
- class-agnostic NMS 검증
- pose contract 및 native-depth registration 검증

이 로컬 후보에 대해 Git commit 또는 push를 수행하지 않았습니다.
