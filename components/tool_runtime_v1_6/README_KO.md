# Surgical Tool ROS2 Runtime v1.6

이 디렉터리가 저장소의 유일한 Tool runtime이다. v1.6 검출기 릴리스에
코디네이터 연동(`processing_enabled` / `processing_gate_topic`)과
Hand/Blood가 쓰는 depth-to-color helper를 포함한다.

## 동작

- Tool/Hand/Blood 순차 실행을 위한 processing_enabled 및 processing_gate_topic
- require_depth가 false일 때의 RGB 전용 2D 관측 발행
- ToolObservation2D의 observation_point_depth_m 필드
- native 16UC1 compressedDepth 디코딩 및 depth-to-color 등록
- constrained planar pose (`PLANAR_4DOF_WITH_NORMAL_PRIOR`)

## v1.6 검출기

- 체크포인트: `cam4_rfdetr_seg_small_regular_resume_e13_best.pth`
- 다운로드: [Google Drive 폴더](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h). 받은 `.pth` 경로를 `config/system.env`의 `TOOL_CHECKPOINT`에 설정한다.
- 기본 confidence threshold: 0.30
- class-agnostic bounding-box NMS: 기본 활성화, IoU 0.80
- pose-axis 디버그 오버레이: `/perception/cam_4/tool/pose_overlay/compressed`

## 빌드

이 디렉터리에서 실행:

    bash scripts/build_ros2.sh

저장소 최상위 `bash scripts/build_all.sh`도 이 workspace를 빌드한다.

## 실행

저장소 최상위에서 실행:

    bash scripts/run_tool_v16.sh
