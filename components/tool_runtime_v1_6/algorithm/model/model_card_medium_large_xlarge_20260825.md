# CAM4 RF-DETRSeg Medium/Large/XLarge model card

## 목적

CAM4 RGB frame에서 8종 수술도구의 bbox와 visible instance mask를 예측한다. 관심 대상은
Mosquito, Adson Forceps, Bipolar Forceps, Bovie이지만 나머지 네 클래스도 class-confusion과
false-positive 억제를 위해 학습에 유지했다.

## 학습 데이터

- dataset schema: `pnu.cam4.round4_r1mask_sam2refined_dataset.v1`
- train: 1,443 images, 4,108 annotations
- train cases: Mayo SAM2 QC, 0704_5, 0704_6, 0704_8, 0704_10, 0704_14
- validation: 225 images, 423 annotations
- validation cases: 0704_11, 0704_17
- mask policy: visible rigid tool only; hand/cable/background excluded

Round1 RF-DETR mask geometry는 bbox-prompted SAM2 mask로 교체했고, 사람의 bbox+SAM2 수정본과 Mayo
SAM2 QC mask는 보존했다. object presence와 class가 일부 Round1-assisted이므로 독립 final GT로 보지 않는다.

## 학습 조건

- official segmentation pretrained initialization
- no fine-tuned checkpoint initialization and no resume
- batch size 16, BF16, 20 epochs, early stopping disabled
- learning rate 5e-5, encoder learning rate 7.5e-6
- Medium resolution 432, Large resolution 504, XLarge resolution 624
- full rotation, affine scale/translation, illumination/color, blur/noise, coarse dropout

현재 augmentation에는 검토 예정 항목이 있다. CoarseDropout은 image와 visible-mask target의 불일치를
만들 수 있고, shear/anisotropic scale/reflection padding은 실제 형상을 왜곡할 수 있다. 이 전달본은
동일 augmentation으로 model scale만 비교한 controlled experiment 결과다.

## 선택 checkpoint

전체 validation mask mAP50:95가 가장 높은 Regular checkpoint를 portable
`checkpoint_best_total.pth`로 전달한다.

전달 파일명은 `models/medium_best.pth`, `models/large_best.pth`, `models/xlarge_best.pth`이며
각각 다음처럼 직접 로드할 수 있다.

```python
from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegXLarge

medium = RFDETRSegMedium.from_checkpoint("models/medium_best.pth")
large = RFDETRSegLarge.from_checkpoint("models/large_best.pth")
xlarge = RFDETRSegXLarge.from_checkpoint("models/xlarge_best.pth")
```

NumPy image 입력은 RGB 순서여야 한다. OpenCV `imread()` 결과는 BGR이므로 반드시 RGB로 변환한다.
별도 NMS는 적용하지 않으며 RF-DETR의 DETR top-K postprocess와 confidence threshold를 사용한다.

## 기존 Small BGR/RGB ablation

기존 production Small checkpoint는 기존 통합 계약대로 BGR 입력을 유지하되, 동일한 225-image validation에서
채널 순서만 바꿔 평가했다.

| Small metric | legacy BGR | RGB candidate | RGB - BGR |
|---|---:|---:|---:|
| bbox mAP50:95 | 0.498968 | **0.511399** | +0.012431 |
| bbox mAP75 | 0.508713 | **0.635640** | +0.126927 |
| bbox mAR100 | 0.546983 | **0.587097** | +0.040115 |
| mask mAP50:95 | **0.495312** | 0.476658 | -0.018655 |
| mask mAP50 | **0.778539** | 0.692025 | -0.086514 |
| mask mAR100 | 0.559842 | **0.597419** | +0.037577 |

RGB는 bbox와 mask recall을 높였지만 전체 mask AP는 낮췄다. 또한 validation의 Mosquito와
Army-Navy는 각각 1 instance뿐이어서 class-macro 지표가 불안정하다. 따라서 기존 Small은 지금
마이그레이션하지 않고 `checkpoint_color_order=BGR` 계약과 재현 경로를 유지한다. Medium/Large/XLarge는
공식 NumPy 입력 계약에 맞춰 `checkpoint_color_order=RGB`로 고정한다. 원자료와 판단은
`metadata/small_bgr_rgb_ablation_20260825.json`에 기록했다.

## RTX A6000 FP16 benchmark

세 모델을 explicit variant loader로 로드하고 `optimize_for_inference(compile=True, batch_size=1,
dtype=torch.float16)`를 적용했다. 1280×720 RGB source 한 장, native square model input, warm-up 10회,
CUDA-synchronized 50회 `model.predict()`만 측정했다. file I/O, overlay, ROS는 제외했다.

| 항목 | Medium | Large | XLarge |
|---|---:|---:|---:|
| mean latency | 131.05 ms | 138.25 ms | 104.68 ms |
| p50 | 137.88 ms | 146.16 ms | 104.99 ms |
| p95 | 169.39 ms | 159.36 ms | 112.25 ms |
| FPS from mean | 7.63 | 7.23 | 9.55 |
| optimization peak allocated | 597.56 MiB | 734.98 MiB | 1,081.79 MiB |
| steady inference peak allocated | 761.70 MiB | 765.49 MiB | 1,049.69 MiB |
| steady inference peak reserved | 1,170 MiB | 1,454 MiB | 2,164 MiB |

이 수치는 A6000 단일 sample inference benchmark이며 로컬 GPU와 전체 video/ROS pipeline 속도를 보장하지 않는다.
XLarge는 별도 시점에 동일 절차로 측정했으므로 작은 latency 차이를 architecture 자체의 속도 우위로 해석하지 않는다.

| metric at selected checkpoint | Medium | Large | XLarge |
|---|---:|---:|---:|
| selected epoch | 12 | 18 | 16 |
| bbox mAP50:95 | **0.547089** | 0.538351 | 0.511454 |
| bbox mAP50 | 0.780549 | 0.777218 | **0.783447** |
| bbox mAP75 | **0.738644** | 0.718103 | 0.678326 |
| bbox mAR | 0.583462 | **0.667392** | 0.562751 |
| mask mAP50:95 | 0.523219 | 0.556900 | **0.610315** |
| mask mAP50 | 0.785223 | 0.789379 | **0.794268** |
| mask mAP75 | n/a | n/a | **0.768965** |
| mask mAR100 | n/a | n/a | **0.669340** |

클래스별 기록은 bbox AP이며 per-class mask AP가 아니다.

| bbox AP50:95 | Medium | Large | XLarge |
|---|---:|---:|---:|
| Adson Forceps | **0.794271** | 0.780327 | 0.743231 |
| Bipolar Forceps | **0.669242** | 0.667299 | 0.606279 |
| Bovie | **0.671932** | 0.641690 | 0.607760 |
| Mosquito | 0.600000 | **0.601010** | 0.600000 |

Mosquito validation GT가 1 instance이므로 해당 AP는 모델 선택 근거로 사용하지 않는다.

## rosbag39 RGB 평가

| model | bbox mAP50:95 | mask mAP50:95 | interest threshold | interest precision | interest recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medium | **0.3663** | 0.4210 | 0.35 | 0.840 | 0.840 | 12 | 12 |
| Large | 0.3643 | 0.4289 | 0.35 | 0.868 | **0.880** | 10 | **9** |
| XLarge | 0.2674 | **0.5242** | 0.40 | **0.953** | 0.813 | **3** | 14 |

Interest operating point는 mask IoU 0.5에서 Mosquito, Adson, Bipolar, Bovie를 합산했다. Bipolar와
Bovie GT는 없으므로 해당 class prediction은 FP로만 반영된다. threshold는 같은 39프레임에서 선택한
후보이며 최종 calibration 값이 아니다. 상세 결과는 `docs/ROSBAG39_EVALUATION_20260825.md`에 있다.

## 사용 판단

- Medium: validation bbox와 rosbag bbox가 가장 좋다.
- Large: XLarge보다 관심 클래스 FN이 적고 bbox/mask 절충이 좋다.
- XLarge: validation/rosbag mask가 가장 좋고 관심 클래스 FP가 가장 적어 현재 전달 우선 모델이다.
- XLarge 0.40을 1순위, Large 0.35를 2순위 후보로 두되 독립 calibration에서 threshold를 확정한다.
