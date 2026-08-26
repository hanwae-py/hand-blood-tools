# 부산대학교 컴퓨터 비전연구실 Surgical Tool Algorithm v1.1.0-rc1

KAIST 로봇 내부 NUC PC의 기존 알고리즘 일부를 교체하기 위한 **비-ROS** 전달본이다.

## 포함 범위

- 선택 가능한 RF-DETRSegSmall/Medium/Large/XLarge 기반 수술도구 8종 instance class,
  confidence, bbox, mask
- 카메라별 Mayo/tray polygon ROI와 mask-overlap 기반 instance 필터
- class-independent spatial association과 최근 class evidence smoothing
- RGB에 정합된 metric depth와 카메라 정보를 이용한 관측 기준점 `P_obs`
- native `16UC1 compressedDepth`, depth/color CameraInfo와 depth→color extrinsic을 이용한
  RGB-aligned metric depth 생성
- support-plane normal과 mask 장축을 이용한 constrained pose
- quaternion `(x,y,z,w)`, pose validity, 품질 및 대칭 정보
- 설치·실행 예제와 독립 검증기

## 포함하지 않는 범위

- ROS node, topic, message, DDS/QoS
- Hand Pose와 Blood Detection
- ROS subscription/publishing과 RGB-depth pair queue 관리
- frame 간 영속 instance ID(내부 temporal association ID는 외부로 발행하지 않음)
- 자유공간의 roll/pitch/yaw를 모두 독립 추정하는 unconstrained full-6D pose
- 실제 tray 데이터로 검증된 모델 성능

## 현재 pose의 정확한 의미

출력 quaternion은 4개 숫자이지만 관측 자유도는
`PLANAR_4DOF_WITH_NORMAL_PRIOR`이다. 위치 3축은 RGB-aligned depth에서 관측하고,
도구의 평면 내 heading은 mask 장축에서 구한다. 나머지 orientation은 입력으로 받은 support-plane
normal에 의해 정해진다. 따라서 downstream은 반드시 다음 조건을 확인해야 한다.

```python
item.position_valid
item.orientation_valid
item.validity == "VALID"
item.pose_mode == "PLANAR_4DOF_WITH_NORMAL_PRIOR"
```

`position_m`은 mask 내부의 depth-valid 표면 관측점 `P_obs`다. centroid, CAD origin,
center of mass, TCP 또는 grasp point가 아니다.

## 입력 책임 경계

호출 측은 다음 두 입력 mode 중 하나를 제공해야 한다.

Aligned mode:

- `uint8 HxWx3` RGB 또는 BGR 영상(호출 시 색상 순서를 명시)
- checkpoint 내부 계약은 Small BGR, Medium/Large/XLarge RGB이며 adapter가 모델별로 변환
- RGB pixel에 이미 정합된 `float32 HxW` depth, 단위 meter
- color camera intrinsic `K`, distortion `D`, 정확한 camera frame
- 같은 camera frame의 support-plane normal과 offset

Native-depth mode:

- RGB와 허용 오차 안에서 pair된 native `16UC1` depth
- 명시적으로 확인된 `depth_scale_m_per_unit`
- depth와 color 각각의 intrinsic/distortion/frame
- `P_color = R @ P_depth + t` 방향의 depth→color extrinsic
- 같은 color camera frame의 support-plane normal과 offset

ROS `compressedDepth` 복원과 timestamp tolerance 검사는 제공 utility로 수행할 수 있지만,
subscription, queue와 one-to-one pairing 정책은 transport adapter가 소유한다.

CAM4는 Mayo stand camera이고 CAM3는 8월 tray camera다. 각각 자기 ROI, calibration,
support-plane 설정과 frame name을 사용해야 한다. 2026-08-25 Arpa RGB 영상용 CAM3 tray와
CAM4 Mayo ROI는 각각 별도 프로파일로 제공한다. 실제 설치 화각이 다르면 새 프로파일을 보정한다.
동일 checkpoint를 쓰더라도 이 geometry를 공유하면 안 된다.

## 권장 읽기 순서

1. `QUICKSTART.md`
2. `docs/ALGORITHM_API.md`
3. `docs/POSE_DEFINITION.md`
4. `docs/LIMITATIONS.md`
5. `environment/install_nuc.md`

본 폴더는 외부 전송 전 검토용 release candidate다. 실제 NUC 사양, CAM4 Mayo/CAM3 tray 공식 calibration,
데이터 반출 승인 및 제3자 라이선스 검토가 완료되면 final release로 승격한다.
