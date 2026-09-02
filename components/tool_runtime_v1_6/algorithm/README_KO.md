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

- 핵심 Python algorithm library 자체의 ROS node, topic, message, DDS/QoS 소유
  (운영 환경에서는 별도 native ROS adapter가 담당)
- Hand Pose와 Blood Detection
- 핵심 library 내부의 ROS subscription/publishing과 RGB-depth pair queue 관리
  (운영 환경에서는 별도 native ROS adapter가 담당)
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

## 전체 처리 흐름

한 RGB-D frame은 다음 순서로 처리한다.

1. RGB를 선택한 RF-DETR checkpoint의 색상 계약에 맞춘다.
2. RF-DETR instance segmentation으로 class, confidence, bbox, mask를 얻는다.
3. confidence, 중복 instance, 작은 mask component와 workspace ROI를 후처리한다.
4. 이전 frame과 class-independent spatial association을 수행하고 class label을 안정화한다.
5. native depth를 사용하는 경우 depth를 color camera 좌표계로 변환하여 RGB pixel에 정합한다.
6. 각 mask의 PCA 장축과 양 끝 형상을 이용해 handle end와 working end를 정한다.
7. mask 내부의 유효 depth에서 위치 `P_obs`를 정하고, support-plane prior와 장축으로 orientation을 만든다.
8. raw `ToolPoseArray`는 그대로 발행하고, 제어용 dynamic Tool TF의 translation과 축 방향을 별도 시간 필터로 안정화한다.

## ROS adapter 토픽

핵심 algorithm library는 비-ROS이지만, 현재 운영 환경의 `native_depth_tool_pose_<camera>` adapter가
ROS 입출력을 담당한다. 아래 표는 실행 중인 CAM3, CAM4, Head 노드에서 확인한 실제 토픽 계약이다.
`<camera>`에는 `cam_3`, `cam_4`, `head` 중 하나가 들어간다.

| Camera | 실행 노드 |
| --- | --- |
| CAM3 | `/native_depth_tool_pose_cam_3` |
| CAM4 | `/native_depth_tool_pose_cam_4` |
| Head | `/native_depth_tool_pose_head` |

### Subscribe 토픽

| 토픽 | 메시지 타입 | Camera | 용도 |
| --- | --- | --- | --- |
| `/perception/ingress/<camera>/color/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | CAM3, CAM4, Head | RF-DETR 입력 RGB |
| `/perception/ingress/<camera>/color/camera_info` | `sensor_msgs/msg/CameraInfo` | CAM3, CAM4, Head | color intrinsic, distortion, frame 정보 |
| `/perception/ingress/<camera>/depth/image_rect_raw/compressedDepth` | `sensor_msgs/msg/CompressedImage` | CAM3, CAM4, Head | `16UC1 compressedDepth` 입력 |
| `/perception/ingress/<camera>/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | CAM3, CAM4, Head | depth intrinsic, distortion, frame 정보 |
| `/perception/ingress/<camera>/extrinsics/depth_to_color` | `realsense2_camera_msgs/msg/Extrinsics` | CAM3, CAM4 | native depth를 color frame으로 변환하는 extrinsic |

CAM3와 CAM4 depth는 color frame에 정합되지 않은 native depth이므로 `depth_to_color` extrinsic이
필수다. Head depth는 이미 color image에 정합되어 있어 Extrinsics 토픽을 subscribe하지 않는다.
RGB와 depth는 live profile의 `maximum_stamp_delta_ns` 이내에서 pair한다. 기준값은 `1 ms`이지만,
실험 중인 profile은 다른 값을 사용할 수 있으므로 실행 노드의 parameter와 diagnostics를 함께 확인한다.
CameraInfo와 Extrinsics는 최신 유효값을 보관하여 pair 처리에 사용한다.

`processing_gate_topic`을 설정하면 adapter가 해당 토픽의 `std_msgs/msg/Bool`도 subscribe할 수 있지만,
현재 CAM3·CAM4·Head live profile에서는 빈 문자열이므로 이 선택적 subscription은 생성되지 않는다.

운영 구조에서 `/synced/<camera>/...`는 ingress relay의 upstream 토픽이다. Tool algorithm adapter는
systemd 실행 시 `/perception/ingress/<camera>/...`를 subscribe하며 `/synced/...`를 직접 읽지 않는다.

### CAM3/CAM4 RGB-Depth timestamp 및 sync 설정 확인

다음 명령은 repository root에서 실행한다. 별도 ROS CLI daemon 상태에 의존하지 않도록 parameter는
각 node의 `get_parameters` service에서 직접 읽는다.

```bash
cd /home/pnucvlab/projects/hand-blood-tools
set +u
source /opt/ros/jazzy/setup.bash
source components/tool_runtime_v1_6/ros2_ws/install/setup.bash
source scripts/perception_runtime_env.sh mtu-safe
set -u
```

RealSense driver의 RGB-Depth 동기화 관련 parameter는 다음과 같이 CAM3와 CAM4를 각각 확인한다.
응답 배열의 순서는 요청한 이름의 순서와 같다.

```bash
for camera in cam_3 cam_4
do
  echo "[${camera}]"
  timeout 10 ros2 service call \
    "/camera/${camera}/get_parameters" \
    rcl_interfaces/srv/GetParameters \
    "{names: [enable_sync, depth_module.inter_cam_sync_mode, depth_module.global_time_enabled, rgb_camera.global_time_enabled, depth_module.depth_profile, rgb_camera.color_profile]}"
done
```

확인할 항목은 다음과 같다.

| Parameter | 의미 |
| --- | --- |
| `enable_sync` | RealSense wrapper의 RGB-Depth stream syncer 사용 여부 |
| `depth_module.inter_cam_sync_mode` | 외부 trigger 기반 카메라 간 master/slave mode; `0`은 비활성 |
| `depth_module.global_time_enabled` | Depth timestamp를 global clock domain으로 변환할지 여부 |
| `rgb_camera.global_time_enabled` | RGB timestamp를 global clock domain으로 변환할지 여부 |
| `depth_module.depth_profile` | Depth 해상도와 FPS |
| `rgb_camera.color_profile` | RGB 해상도와 FPS |

`global_time_enabled=true`는 두 stream의 clock 기준을 통일하지만 촬영 trigger의 동시성을 보장하지
않는다. `enable_sync=false`이면 RGB와 Depth가 같은 nominal FPS여도 시작 위상이나 실제 주기 차이로
source stamp가 어긋날 수 있다.

Tool adapter가 실제 사용하는 pairing gate는 다음 명령으로 확인한다.

```bash
for camera in cam_3 cam_4
do
  echo "[${camera}]"
  timeout 10 ros2 service call \
    "/native_depth_tool_pose_${camera}/get_parameters" \
    rcl_interfaces/srv/GetParameters \
    "{names: [require_depth, maximum_stamp_delta_ns, sync_queue_size]}"
done
```

실제로 처리된 RGB-Depth pair의 차이는 Tool diagnostics의 `rgb_depth_delta_ns`가 기준이다.
`1 ms = 1,000,000 ns`이며, 이 값은 선택된 RGB와 Depth source stamp의 절댓값 차이다.

```bash
for camera in cam_3 cam_4
do
  echo "[${camera}]"
  timeout 10 ros2 topic echo --no-daemon --full-length --once \
    "/perception/${camera}/tool/diagnostics" std_msgs/msg/String
done
```

diagnostics가 timeout 안에 나오지 않으면 새 valid pair가 생성되지 않는 경우를 먼저 의심한다. 이때
`input_fresh`, `paired_frames`, `dropped_unmatched_frames`와 node readiness는 health에서 확인한다.

```bash
for camera in cam_3 cam_4
do
  echo "[${camera}]"
  timeout 10 ros2 topic echo --no-daemon --full-length --once \
    "/perception/${camera}/tool/health" std_msgs/msg/String
done
```

Tool node보다 upstream에서 이미 stamp가 어긋나는지 확인하려면 multicam status의 `cam_3`/`cam_3_depth`,
`cam_4`/`cam_4_depth` 항목에 있는 `source_stamp`를 비교한다. 이 status는 원인 구간을 찾기 위한 최신
stream snapshot이고, Tool이 실제 선택한 pair의 최종 판정은 위 diagnostics를 사용한다.

```bash
timeout 10 ros2 topic echo --no-daemon --full-length --once \
  /synced/stream_status std_msgs/msg/String
```

### Publish 토픽

| 토픽 | 메시지 타입 | 내용 |
| --- | --- | --- |
| `/perception/<camera>/tool/poses` | `surgical_perception_msgs/msg/ToolPoseArray` | source frame별 raw metric position, quaternion, validity와 품질 정보 |
| `/perception/<camera>/tool/observations` | `surgical_perception_msgs/msg/ToolObservation2DArray` | class, confidence, bbox, COCO RLE mask와 `P_obs`의 2-D evidence |
| `/perception/<camera>/tool/masks/<class_slug>` | `sensor_msgs/msg/Image` | class별 union mask; source 해상도의 `mono8`, background `0`, foreground `255` |
| `/perception/<camera>/tool/overlay/compressed` | `sensor_msgs/msg/CompressedImage` | recognition bbox/mask 시각화 |
| `/perception/<camera>/tool/pose_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | tool pose 축과 상태 시각화 |
| `/perception/<camera>/tool/diagnostics` | `std_msgs/msg/String` | frame 처리량, latency, model·ROI·TF 상태를 담은 JSON diagnostics |
| `/perception/<camera>/tool/health` | `std_msgs/msg/String` | 입력 freshness와 model/registration 준비 상태를 담은 JSON health |
| `/tf` | `tf2_msgs/msg/TFMessage` | validity를 통과한 현재 tool의 control-facing dynamic TF |

`<class_slug>`는 다음 8개다.

```text
scalpel
allis_forceps
mosquito
adson_forceps
bipolar_forceps
bovie
army_navy_retractor
thyroid_retractor
```

`/perception/<camera>/tool/poses`의 `ToolPoseArray`는 필터링하지 않은 원 관측이다. CAM3·CAM4의 `/tf`에는
temporal position stability와 모든 tool class용 축 방향 안정화가 적용되며 raw position/quaternion은
변경하지 않는다. Head도 `/tf`를 publish하지만 현재 Head profile에는 두 stabilizer가 활성화되어 있지 않다.
`/rosout`과 `/parameter_events`는 ROS 2 node가 자동 생성하는 관리 토픽이므로 위 알고리즘 인터페이스에서
제외했다.

## 1. Recognition 알고리즘

### 1.1 Instance segmentation

기본 모델은 RF-DETRSegXLarge이고 Small/Medium/Large도 선택할 수 있다. 모델은 한 instance마다
다음을 출력한다.

- 8종 canonical surgical-tool class와 confidence
- source-image 좌표의 bbox `(x_min, y_min, x_max, y_max)`
- source image와 같은 해상도의 binary instance mask

8종 class는 Scalpel, Allis Forceps, Mosquito, Adson Forceps, Bipolar Forceps, Bovie,
Army-Navy Retractor, Thyroid Retractor다. 호출 배열의 `color_order`는 반드시 명시하며, adapter는
Small checkpoint에는 BGR, Medium/Large/XLarge checkpoint에는 RGB가 정확히 한 번 입력되도록 변환한다.
기본 confidence threshold는 `0.30`이고 camera/class profile이 이를 더 높일 수 있다. 예를 들어 현재
CAM4 live profile은 Adson Forceps에 `0.45`를 사용한다.

Small 경로에서는 bbox IoU `0.80`의 class-agnostic NMS를 적용한다. 따라서 같은 위치를 서로 다른
class로 중복 검출해도 confidence가 높은 하나만 남는다. 이 경로는 같은 class mask의 작은 쪽 면적 중
`95%` 이상이 다른 mask에 포함되면 중복으로 제거하는 containment 검사도 수행한다. Medium/Large/XLarge는
현재 모델 출력에 별도 class-agnostic NMS를 추가하지 않는다.

### 1.2 Mask와 workspace 후처리

활성화된 live profile은 instance bbox 내부의 8-connected component를 검사한다. 면적이
`max(16 pixel, 전체 mask 면적의 0.5%)`보다 작은 island는 제거하되, 가장 큰 component는 항상
유지한다. 정리한 mask 경계로 bbox도 다시 계산한다.

camera마다 Mayo/tray를 나타내는 normalized polygon ROI를 별도로 가진다. ROI acceptance는

```text
overlap = area(instance_mask AND roi) / area(instance_mask)
```

가 profile threshold 이상이고 mask centroid가 ROI 내부일 때만 instance 전체를 유지한다. 이 과정은
mask를 ROI 경계에서 잘라내지 않으며 inference 뒤에 실행되므로 GPU inference 연산량도 줄이지 않는다.
ROI가 실제 설치 화각과 맞지 않으면 valid tool을 제거할 수 있으므로 CAM3/CAM4 geometry를 공유하지 않는다.

### 1.3 Recognition stability: 공간 연계

한 frame의 순간적인 class 오인식이 label flicker로 이어지지 않도록 먼저 이전 track과 현재 instance를
공간적으로 연결한다. 이때 **class label은 matching cost에 사용하지 않는다.** 즉 같은 물체의 raw class가
바뀌어도 mask 위치와 모양이 이어지면 같은 내부 track으로 연결할 수 있다.

현재 instance `i`와 이전 track `j`는 다음 hard gate를 모두 통과해야 matching 후보가 된다.

```text
mask_area_ratio <= 3.0
centroid_distance / image_diagonal <= 0.06
mask_IoU >= 0.10 OR bbox_IoU >= 0.20
```

후보 score는 다음과 같다.

```text
distance_similarity = max(0, 1 - normalized_distance / 0.06)
association_score = 0.55 * mask_IoU
                  + 0.30 * bbox_IoU
                  + 0.15 * distance_similarity
```

모든 후보를 score 내림차순으로 정렬한 뒤 current instance와 이전 track이 각각 한 번만 선택되도록
greedy one-to-one matching한다. 연결되지 않은 instance에는 새 내부 track을 만들고, `3` frame을 초과해
관측되지 않은 track은 제거한다.

이 association은 class smoothing을 위한 내부 상태다. mask/bbox/centroid 자체를 시간 평균하지 않으며,
외부로 frame 간 persistent ID를 제공하지 않는다. 공개 `frame_local_instance_id`는 매 frame 안에서만
유효하다. 도구가 교차하거나 가려져 hard gate를 벗어나면 association이 바뀔 수 있다.

### 1.4 Recognition stability: class evidence hysteresis

각 내부 track은 최근 `7` frame의 `(raw class, confidence)` evidence를 보관한다. class `c`의 누적
score와 관측 횟수는 다음과 같다.

```text
S(c) = sum(confidence_t for samples classified as c)
N(c) = number of samples classified as c
```

누적 score가 가장 큰 class를 switch 후보로 선택하지만 다음 두 조건을 동시에 만족할 때만 stable class를
바꾼다.

```text
N(candidate) >= 3
S(candidate) >= S(current_stable_class) + 0.20
```

조건을 충족하지 못한 일시적인 오분류는 기존 stable class로 override한다. 출력 confidence는 history 안에서
stable class로 관측된 confidence의 평균이다. 이 방식은 1~2 frame class flicker를 억제하지만 실제 class
변화에도 최소 evidence가 쌓일 때까지 지연이 생긴다. timestamp 역행, bag 전환과 같은 discontinuity에서는
temporal state를 reset해야 이전 장면의 evidence가 섞이지 않는다.

## 2. Pose 알고리즘

### 2.1 Depth-to-color registration

native `16UC1` depth는 `depth_scale_m_per_unit`을 곱해 meter로 바꾼 뒤 depth pixel을 depth-camera
ray로 역투영한다. 각 3-D point에 다음 extrinsic을 적용하고 color camera로 재투영한다.

```text
P_color = R_color_from_depth * P_depth + t_color_from_depth
```

여러 depth point가 같은 color pixel에 들어오면 nearest-z를 유지한다. 결과는 RGB와 같은 해상도의
`float32` z-depth이고 invalid pixel은 `NaN`이다. RGB/depth timestamp 허용 오차, depth unit,
intrinsic, distortion, extrinsic 방향과 frame name이 맞지 않으면 metric pose도 맞지 않는다.

### 2.2 Mask 장축과 끝점 계산

mask foreground pixel `u_k = (x_k, y_k)`의 평균과 2-D covariance를 계산하고 가장 큰 eigenvalue의
eigenvector를 장축으로 사용한다.

```text
u_mean = mean(u_k)
C = mean((u_k - u_mean)(u_k - u_mean)^T)
axis_uv = eigenvector_of_largest_eigenvalue(C)
axis_anisotropy = lambda_max / lambda_min
```

PCA eigenvector는 부호가 없는 직선이므로 양 끝을 구한 뒤 class별 형상 규칙으로 부호를 정한다. mask
outlier의 영향을 줄이기 위해 장축 projection의 최솟값/최댓값 대신 `2%`와 `98%` quantile을 endpoint로
사용한다. 두 trimmed endpoint의 중점을 2-D longitudinal origin으로 사용한다.

### 2.3 위치 `P_obs`

longitudinal origin에 가장 가까운 한 pixel을 그대로 고르면 mask 경계의 noisy depth를 선택할 수 있다.
따라서 다음 절차로 mask 내부 관측점을 정한다.

1. 유효 depth가 있고 longitudinal origin에서 장축 방향 거리가
   `max(4 pixel, 장축 길이의 10%)` 이내인 중앙 band를 만든다.
2. 각 후보의 mask 경계까지 거리와 origin까지 Euclidean 거리를 계산한다.
3. 아래 score가 가장 큰 pixel을 선택한다.

```text
score = normalized_boundary_clearance
      - 0.35 * distance_to_origin / axis_length
```

중앙 band에 유효 depth가 없으면 mask 내부의 다른 depth-valid pixel로 fallback하고 pose를
`DEGRADED`로 표시한다. 선택한 color pixel `(u,v)`을 distortion-aware camera ray `r(u,v)`로 만들고
해당 pixel의 z-depth `z`로 역투영한다.

```text
P_obs = r(u,v) * z / r_z
```

`P_obs`는 실제 mask 내부에서 관측된 표면점이며 support plane 위로 투영하지 않는다. 따라서 CAD origin,
도구 중심, TCP나 grasp point로 해석하면 안 된다.

### 2.4 +Y 축 방향 결정

도구 좌표계의 의미는 다음과 같이 고정한다.

```text
+Y: handle/proximal end -> working tip
+Z: support plane -> free space 방향의 plane normal
+X: +Y cross +Z
R_tool_in_camera = [ +X  +Y  +Z ]
quaternion order = (x, y, z, w)
```

`positive_y_image_direction: class_based`일 때 endpoint 부호 규칙은 다음과 같다.

| Class | working tip을 정하는 규칙 |
| --- | --- |
| Scalpel, Allis Forceps, Mosquito, Thyroid Retractor | terminal mass가 큰 쪽을 handle로 보고 반대쪽을 tip으로 선택 |
| Adson Forceps | 전체 mask의 배치 형상을 먼저 판별; 폭 profile이 선형·단조롭게 넓어지는 삼각형이면 넓은 쪽을 tip으로 선택하고, 그렇지 않으면 two-jaw component와 terminal taper를 사용 |
| Bipolar Forceps | connector taper, black handle/blue tip 색상, terminal mass를 각각 signed vote로 계산한 뒤 비계층 가중 앙상블로 handle을 선택 |
| Bovie | RGB가 있으면 mask 바깥 wire 연결 evidence로 handle을 찾고, 그렇지 않으면 tip taper를 사용; 모호하면 큰 terminal mass를 handle 대표로 선택하고 low confidence 처리 |
| Army-Navy Retractor | 양 끝이 모두 blade인 C2 대칭이므로 물리적인 유일 +Y가 없음; image/CAM4 기준의 결정론적 대표 부호를 선택 |

Adson은 two-jaw gap이 segmentation mask에서 희미하거나 합쳐지는 경우를 고려하여 component 검출보다
전체 silhouette를 먼저 본다. 장축을 `9`개 구간으로 나누고 각 구간의 transverse `5%~95%` 폭으로
width profile을 만든다. 다음 조건을 모두 만족하면 `TRIANGULAR_WIDE_TIP` 배치로 분류한다.

```text
wide_end_width / narrow_end_width >= 1.40
linear width-profile R^2 >= 0.65
working 방향 단조 증가 구간 비율 >= 0.70
abs(wide_width - narrow_width) / mean_width >= 0.35
```

삼각형 배치에서는 jaw component가 분리되어 보이는지와 관계없이 넓은 쪽을 working tip으로 정한다.
삼각형 기준을 통과하지 않은 slender/non-triangular 배치에서만 two-jaw component를 검사하고, 그 다음
terminal taper, 마지막으로 low-confidence terminal-mass fallback을 사용한다.

Bipolar는 어느 한 cue가 다른 cue를 선점하지 않는다. 각 vote는 `[-1, 1]`이며 양수는 PCA high end가
handle, 음수는 low end가 handle이라는 뜻이다.

```text
taper_vote = signed connector-taper separation
black_vote = LAB-L darkness + black-pixel-fraction evidence  # black = handle
blue_vote = HSV blue-pixel-fraction evidence                 # blue = tip
colour_vote = agreement(black_vote, blue_vote)
mass_vote = (high_terminal_mass - low_terminal_mass) / total_terminal_mass

ensemble_score = 0.45 * taper_vote
               + 0.40 * colour_vote
               + 0.15 * mass_vote
```

색상 vote는 mask 내부 양 끝 `20%`에서 계산한다. LAB-L이 낮은 black 쪽은 proximal handle로, HSV에서
blue로 분류된 쪽은 working tip으로 투표한다. 두 cue가 반대 endpoint를 가리켜 해부학적으로 일치하면
완전한 colour vote를 사용한다. 한 cue만 검출되면 색상 강도를 `0.65`배로 낮추며, 서로 충돌하면 상쇄한다.
색상 방향성이 충분하지 않거나 RGB가 없으면 colour vote는 기권하고 활성 weight만 다시 normalize한다.
최종 score가 양수면 high end, 음수면 low end를 handle로 정하고 `abs(score)`를 endpoint sign confidence로
사용한다. 따라서 강한 색상 하나가 무조건 결과를 결정하지 않고 taper와 mass가 반대 방향으로 투표하면
서로 상쇄된다. Bipolar의 외부 cable/wire는 이 앙상블에 포함하지 않는다.

Bovie의 external-wire → tip-taper → terminal-mass fallback 규칙과 모든 threshold는 변경하지 않는다.

운영자가 tool anatomy와 무관하게 화면 방향을 강제해야 할 때는 `positive_y_image_direction`을 `down` 또는
`right`로 설정할 수 있다. 이 설정은 PCA 장축을 회전시키지 않고 eigenvector의 **부호만** 골라 projected
`+Y`가 각각 image 아래쪽 또는 오른쪽을 향하게 한다. `class_based`와 이 override를 동시에 적용하지 않는다.

두 2-D endpoint ray를 support plane

```text
n^T P + d = 0
```

과 교차시켜 `P_working`, `P_handle`을 얻은 뒤 3-D 축을 만든다.

```text
Y_raw = P_working - P_handle
+Y = normalize(Y_raw - (Y_raw dot n) * n)
+Z = n
+X = normalize(+Y cross +Z)
+Y = normalize(+Z cross +X)  # numerical re-orthogonalization
```

마지막으로 `[+X,+Y,+Z]` rotation matrix를 normalize된 quaternion `(x,y,z,w)`로 변환한다.

### 2.5 Pose validity

기본 유효성 기준은 mask `20 pixel` 이상, mask 내부 valid-depth ratio `0.05` 이상,
`axis_anisotropy >= 2.0`, `endpoint_sign_confidence >= 0.20`이다. C2 대칭인 Army-Navy Retractor에는
endpoint confidence threshold를 적용하지 않고 symmetry를 명시한다.

- `VALID`: position/orientation 기준을 모두 통과하고 observation-point fallback이 없음
- `DEGRADED`: position은 있으나 endpoint sign, 장축 또는 fallback에 주의가 필요
- `INVALID`: metric position을 포함해 pose를 사용하면 안 됨

`orientation_xyzw`가 계산되었더라도 `orientation_valid == false`일 수 있으므로 숫자의 존재만으로
사용 여부를 결정하지 않는다.

### 2.6 Pose stability

단일 frame의 기하 안정성은 다음 설계로 높인다.

- mask 전체 pixel PCA를 사용하여 일부 pixel 변화에 대한 장축 민감도를 낮춘다.
- endpoint는 2%/98% trimmed projection을 사용하여 작은 mask island와 끝점 outlier 영향을 줄인다.
- 위치는 mask 경계보다 내부 clearance가 큰 depth pixel을 선호하여 depth-edge noise를 피한다.
- roll/pitch를 매 frame noisy depth plane으로 다시 fitting하지 않고 camera별 support-plane normal prior로
  고정한다.
- 장축 anisotropy와 endpoint sign confidence가 낮으면 신뢰 가능한 orientation으로 표시하지 않는다.

시간축 안정화는 **제어용 dynamic Tool TF**에 적용한다. raw `ToolPoseArray`의 position과 quaternion은
수정하지 않아 원 관측과 필터 출력을 구분할 수 있다. state key는
`mayo_<class>#ordinal` 또는 `tray_<class>#ordinal` 같은 현재 화면의 좌→우 spatial selector다.

```text
delta = norm(P_raw - P_filtered)

delta <= deadband:
    P_out = P_filtered

deadband < delta <= 0.040 m:
    P_filtered = P_filtered + alpha * (P_raw - P_filtered)

delta > 0.040 m:
    이전 큰 이동 후보와 0.015 m 이내인 관측이 2 frame 연속 확인될 때까지
    이전 P_filtered를 유지하고, 확인되면 후보 평균으로 relocation
```

CAM3·CAM4 live profile은 대부분 정지한 도구를 우선하여 position deadband `0.002 m`, EMA alpha `0.10`을
사용한다. selector가 `3` frame을 초과해 사라지면 filter state를 만료한다. 같은 class의 검출 개수가 바뀌어
좌→우 ordinal의 의미가 달라질 때는 관련 state를 reset하여 다른 도구의 과거 위치를 물려받지 않게 한다.

축 방향 안정화는 Adson에 한정하지 않고 **모든 surgical-tool TF class**에 동일하게 적용한다. 이전에
채택한 tool-frame `+Y`와 현재 `+Y`의 dot product가 `0` 미만이면 반대 hemisphere의 급격한 축 반전 후보로
판정한다. CAM3·CAM4 live profile에서는 서로 `dot >= 0.85`이고 `orientation_valid == true`인 반대 방향
관측이 `5 frame` 연속될 때만 새 방향을 채택한다. 저신뢰도 관측은 이전 방향을 유지할 뿐 아니라 연속 확인
횟수도 초기화한다. selector cardinality 변화·clock reset·state expiry에서는 축 상태를 함께 reset한다.

quaternion의 `q`와 `-q`는 같은 회전이므로 먼저 같은 quaternion hemisphere로 정규화한다. `1.5 degree`
이내 각도 jitter는 이전 자세를 유지하고, 그 밖의 정상적인 점진 회전은 quaternion SLERP alpha `0.15`로
평활화한다.

이 안정화는 endpoint 판정 규칙을 변경하지 않고 그 결과의 일시적인 180도 flip만 제어용 TF에서 억제한다.
따라서 raw quaternion과 endpoint confidence/status flag는 진단 및 offline tuning에 그대로 사용할 수 있다.

### 2.7 "6D pose" 해석 시 주의사항

출력은 translation 3개와 quaternion을 가지므로 SE(3) interface로 전달할 수 있다. 그러나 영상과 depth에서
독립적으로 관측하는 자유도는 translation 3축과 mask의 평면 내 heading 1축뿐이다. +Z와 그에 따른 나머지
orientation은 support-plane normal prior에서 완성된다. 즉 현재 결과는 unconstrained full-6D 추정이 아니라
`PLANAR_4DOF_WITH_NORMAL_PRIOR`이며, 도구가 plane에서 들리거나 기울어진 경우 실제 roll/pitch를 표현하지
못한다.

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
