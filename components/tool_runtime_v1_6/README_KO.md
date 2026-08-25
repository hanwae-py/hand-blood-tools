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

- 체크포인트: Small `cam4_rfdetr_seg_small_regular_resume_best.pth`,
  Medium `medium_best.pth`, Large `large_best.pth`, XLarge `xlarge_best.pth` 중 선택
- 다운로드: [Google Drive 폴더](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h).
  받은 `.pth` 경로와 `TOOL_MODEL_SIZE`를 `config/system.env`에 설정한다.
- 기본 confidence threshold: 0.30
- Small: 기존 BGR 입력과 class-agnostic bbox NMS IoU 0.80 유지
- Medium/Large/XLarge: RGB 입력, 별도 NMS 없음
- rosbag/live 화각별 ROI 프로파일과 mask-overlap ROI 필터
- class와 무관한 mask/bbox association 및 최근 7-frame confidence-weighted
  class smoothing(3-frame 전환 확인)
- CAM3는 8월 tray camera이나 영상 미수령 상태이므로 ROI 비활성화
- pose-axis 디버그 오버레이: `/perception/cam_4/tool/pose_overlay/compressed`

## 빌드

이 디렉터리에서 실행:

    bash scripts/build_ros2.sh

저장소 최상위 `bash scripts/build_all.sh`도 이 workspace를 빌드한다.

## 실행

저장소 최상위에서, CAM4 ingress
(`bash scripts/run_perception_ingress.sh`) 실행 후:

    TOOL_MODEL_SIZE=xlarge bash scripts/run_tool_v16.sh

실시간 ROS 처리에서는 현재 설치 화각에 맞춰 보정한 프로파일을 명시한다.
미지정 기본값 `TOOL_ROI_PROFILE=none`은 ROI 필터를 적용하지 않는다.

    TOOL_MODEL_SIZE=xlarge TOOL_ROI_PROFILE=cam4_live_room1_mayo_20260825 \
      bash scripts/run_tool_v16.sh cam_4
