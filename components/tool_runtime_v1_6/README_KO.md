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
  Medium `medium_best.pth`, Large `large_best.pth`, XLarge
  `checkpoint_selected_external_0825_holdout_conf030.pth` 중 선택
- 다운로드: [Google Drive 폴더](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h).
  받은 `.pth` 경로와 `TOOL_MODEL_SIZE`를 `config/system.env`에 설정한다.
- 기본 confidence threshold: 0.30
- 정확도 우선 기본값: XLarge. 측정한 지연이 허용 범위를 넘을 때만
  Large, 이후 Medium 순으로 낮춘다.
- Small: 기존 BGR 입력과 class-agnostic bbox NMS IoU 0.80 유지
- Medium/Large/XLarge: RGB 입력, 별도 NMS 없음
- rosbag/live 화각별 ROI 프로파일과 mask-overlap ROI 필터
- class와 무관한 mask/bbox association 및 최근 7-frame confidence-weighted
  class smoothing(3-frame 전환 확인)
- 제어용 동적 TF 위치에는 EMA와 2-frame relocation 확인을 적용한다.
  원시 `ToolPoseArray`는 비교·재튜닝을 위해 변경하지 않는다.
- v5 XLarge best 모델로 수행한 2026-08-14 CAM4 MCAP 447 RGB-D frame
  replay에서 연속 비-relocation 3D step p95가 16.75 mm에서 4.36 mm로
  감소했다(74.0%). 이는 안정성 평가이며 support plane/depth scale이
  provisional이므로 절대 pose 정확도 평가는 아니다.
- 2026-08-25 Arpa RGB용 CAM3 tray/CAM4 Mayo ROI 프로파일 포함
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

라이브 CAM3/CAM4 Tool ROI는 각 카메라 화면에서 polygon을 직접 지정할 수 있다.

    bash scripts/run_tool_roi_selector.sh cam_3
    bash scripts/run_tool_roi_selector.sh cam_4

동시 실행 환경에서는 `config/system.env`의 `TOOL_ROI_PROFILE_CAM3`와
`TOOL_ROI_PROFILE_CAM4`에 저장된 프로파일 이름을 각각 설정한다.
Taskplanner Final Overlay는 두 Tool worker와 같은 프로파일을 읽어 실제
인식 허용 polygon을 각 카메라 영상에 `TOOL ROI`로 표시한다.
현재 라이브 CAM3/CAM4 프로파일은 mask 면적의 70% 이상이 ROI 안에 있고
mask centroid도 안쪽일 때만 검출을 승인한다.
