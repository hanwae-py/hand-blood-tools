# RF-DETRSeg scale models — rosbag39 RGB evaluation

평가일: 2026-08-25

## 결론

- bbox 정확도는 Medium이 가장 높다.
- visible-mask 정확도, mask recall, 고신뢰 운영점은 XLarge가 가장 높다.
- 관심 클래스 publish 기준의 우선 후보는 XLarge threshold 0.40이다.
- FN을 FP보다 더 크게 벌점화하면 Large threshold 0.35도 비교 후보로 유지한다.
- Small은 같은 데이터와 초기화 조건에서 Medium/Large/XLarge보다 명확히 열세다.

## 평가 계약

- GT: `finalized39_gt_sam2refined.coco.json`
- frame: 사람이 완료한 rosbag 39장
- GT instance: 153개
- RGB contract: OpenCV BGR decode 후 `cv2.COLOR_BGR2RGB`를 정확히 한 번 적용
- checkpoint: 각 scale run의 `checkpoint_best_total.pth`
- candidate floor: 0.001
- COCO bbox/segm AP: 전체 score range
- operating point: 같은 class끼리 score 내림차순 greedy matching
- TP 조건: predicted mask와 GT mask의 IoU가 0.5 이상
- threshold sweep: 0.10–0.80, 0.05 간격

GT 클래스 구성:

| class | GT |
|---|---:|
| Mosquito | 37 |
| Adson Forceps | 38 |
| Army-Navy Retractor | 38 |
| Thyroid Retractor | 40 |
| Bipolar Forceps | 0 |
| Bovie | 0 |

Bipolar/Bovie recall은 이 rosbag으로 평가할 수 없다. 다만 Bipolar/Bovie로 잘못 publish되는 예측은
관심 4클래스 false positive에 포함했다.

## COCO 평가

| model | bbox mAP50:95 | bbox AP50 | bbox AP75 | mask mAP50:95 | mask AP50 | mask AP75 | mask mAR100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 0.1969 | 0.5092 | 0.0607 | 0.2741 | 0.5830 | 0.2037 | 0.5230 |
| Medium | **0.3663** | 0.7570 | **0.3969** | 0.4210 | 0.7919 | 0.3944 | 0.5690 |
| Large | 0.3643 | **0.7633** | 0.2184 | 0.4289 | 0.7445 | 0.4713 | 0.6130 |
| XLarge | 0.2674 | 0.7025 | 0.1336 | **0.5242** | **0.8683** | **0.6298** | **0.6744** |

관심 클래스별 AP:

| model | Mosquito bbox | Adson bbox | Mosquito mask | Adson mask |
|---|---:|---:|---:|---:|
| Small | 0.3576 | 0.1775 | 0.4675 | 0.3094 |
| Medium | **0.5797** | **0.3953** | 0.6362 | 0.4221 |
| Large | 0.4752 | 0.3689 | 0.6903 | 0.3329 |
| XLarge | 0.3629 | 0.2029 | **0.6988** | **0.5319** |

XLarge는 mask가 가장 좋지만 bbox AP는 Medium/Large보다 낮다. downstream이 mask 기반 depth와
axis를 사용하더라도 bbox를 ROI나 gating에 사용한다면 이 차이를 무시하면 안 된다.

## 고정 threshold 0.2/0.5

모든 153 GT instance를 포함한 mask IoU 0.5 operating point다.

| model | th | precision | recall | F1 | FP/frame | FN/frame |
|---|---:|---:|---:|---:|---:|---:|
| Small | 0.2 | 0.220 | 0.458 | 0.297 | 6.359 | 2.128 |
| Medium | 0.2 | 0.525 | 0.608 | 0.564 | 2.154 | 1.538 |
| Large | 0.2 | 0.554 | 0.673 | 0.608 | **2.128** | 1.282 |
| XLarge | 0.2 | 0.511 | **0.791** | **0.621** | 2.974 | **0.821** |
| Small | 0.5 | 0.645 | 0.320 | 0.428 | 0.692 | 2.667 |
| Medium | 0.5 | 0.902 | 0.301 | 0.451 | 0.128 | 2.744 |
| Large | 0.5 | 0.946 | 0.346 | 0.507 | 0.077 | 2.564 |
| XLarge | 0.5 | **0.969** | **0.405** | **0.571** | **0.051** | **2.333** |

0.5에서는 XLarge가 precision, recall, F1, FP/frame, FN/frame 모두 가장 좋다. 다만 전체 recall
0.405는 여전히 낮아서 0.5를 그대로 최종 threshold로 사용하기는 어렵다.

## 관심 4클래스 threshold sweep

Mosquito, Adson, Bipolar, Bovie만 합산했다. Bipolar/Bovie GT는 없으므로 이 두 클래스로 나온
예측은 FP로만 반영된다.

| model | best threshold | precision | recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Small | 0.45 | 0.684 | 0.720 | 0.701 | 25 | 21 |
| Medium | 0.35 | 0.840 | 0.840 | 0.840 | 12 | 12 |
| Large | 0.35 | 0.868 | **0.880** | 0.874 | 10 | **9** |
| XLarge | 0.40 | **0.953** | 0.813 | **0.878** | **3** | 14 |

판단:

- FP 억제가 우선이면 XLarge 0.40: FP 3, FN 14.
- FN 억제가 우선이면 Large 0.35: FP 10, FN 9.
- FP와 FN을 동일 비용으로 합산해도 XLarge 오류 17개, Large 오류 19개로 XLarge가 근소하게 우세하다.
- 실제 배포 후보는 XLarge 0.40을 1순위, Large 0.35를 2순위로 둔다.
- threshold는 이 39장에서 선택했으므로 독립 calibration/evaluation에서 다시 고정해야 한다.

## 중요한 한계

- 39프레임은 전체 rosbag의 작은 부분이며 모든 frame이 positive라 empty-background FP를 측정하지 못한다.
- GT는 Round1-assisted review에서 출발해 SAM2 mask로 refine되었으므로 confirmation bias 가능성이 있다.
- 이 결과에서 threshold를 고르고 같은 결과를 최종 성능으로 인용하면 낙관 편향이 생긴다.
- Bipolar/Bovie가 실제 존재하는 별도 외부 sequence가 필요하다.
- 프레임 단위 평가이므로 temporal flicker, ID switch, track fragmentation, pose jitter는 측정하지 않았다.
- 모델은 inference optimization 없이 평가했다. 정확도에는 영향이 없지만 이 실행 시간은 FPS benchmark가 아니다.

## 산출물

- `20260825_scale_small_medium_rosbag39_rgb.json`
- `20260825_scale_large_xlarge_rosbag39_rgb.json`
- `20260825_scale_rosbag39_rgb_predictions/`
