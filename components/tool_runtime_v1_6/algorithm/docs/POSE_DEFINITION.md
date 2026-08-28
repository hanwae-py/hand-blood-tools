# Pose 정의

## 위치

`position_m = P_obs`는 instance mask 내부에서 다음 기준으로 선택한 RGB-aligned depth 표면점이다.

1. mask의 trimmed longitudinal midpoint 근처를 우선한다.
2. depth가 유효한 foreground pixel만 후보로 쓴다.
3. 경계에서 멀고 midpoint에 가까운 후보를 고른다.
4. 중앙 band에 유효 depth가 없으면 mask 내부 다른 유효 pixel을 fallback으로 사용하고
   결과를 `DEGRADED`로 표시한다.

좌표 단위는 meter, 좌표계는 `camera.frame_name`이다.

## 방향

- `+Y`: handle/proximal end에서 working tip 방향
- `+Z`: support plane에서 free space 방향인 normal
- `+X = +Y × +Z`
- rotation matrix column 순서: `(+X,+Y,+Z)`
- quaternion 순서: `(x,y,z,w)`

orientation 중 image mask에서 새로 관측하는 값은 평면 내 heading이다. support-plane normal은 외부 입력
prior이므로 mode는 `PLANAR_4DOF_WITH_NORMAL_PRIOR`이다.

### Bipolar Forceps 끝단 부호

Bipolar는 `0704_6`의 굽고 벌어진 형태와 `0704_9`의 직선형 형태에서 tip 쪽 면적 관계가 반대로
나타난다. 따라서 단순히 큰 끝을 handle로 간주하지 않는다. PCA 장축 양 끝에서 terminal 10%와 바로
안쪽 shoulder 10%의 면적 수축률을 비교하고, cable이 mask에서 제외되면서 더 강하게 가늘어지는
connector 쪽을 handle(`-Y`)로 선택한다. 반대쪽 electrode tip이 항상 `+Y`다.

두 수축률이 비슷하거나 수축 신호가 약하면 quaternion은 계산하되 `endpoint_sign_confidence`가 낮아져
결과가 `DEGRADED`로 표시된다. 실루엣만으로 끝단을 구분할 수 없는 mask를 임의로 확정하지 않는다.

## 대칭

Army-Navy Retractor는 양쪽 끝이 물리적으로 유사한 `C2` 대칭이다. 출력 quaternion은 반복 가능한
대표 방향 하나를 택하지만, downstream은 180도 회전한 등가 pose가 있음을 인지해야 한다.

## 유효성

- `VALID`: 위치와 방향이 모두 기준을 통과하고 P_obs fallback이 없음
- `DEGRADED`: 위치는 사용할 수 있으나 endpoint sign/장축/fallback 등에 주의가 필요
- `INVALID`: 위치를 포함해 pose를 사용하면 안 됨

Quaternion 값이 존재하는 것 자체는 `VALID`를 뜻하지 않는다.
