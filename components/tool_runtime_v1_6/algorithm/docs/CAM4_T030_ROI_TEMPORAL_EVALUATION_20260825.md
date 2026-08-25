# CAM4 threshold 0.30 + Mayo ROI/temporal evaluation (2026-08-25)

## Scope

- Source: 447 unlabeled CAM4 frames extracted from the 2026-08-14 reference rosbag
- Model variants: RF-DETR Seg Small, Medium, Large, XLarge
- Common confidence threshold: 0.30
- Interpretation: external-scene prediction comparison only; this is not accuracy, precision,
  recall, IoU, or mAP because the sequence has no ground truth
- Local artifact: `rfdetr_all_models_t030_mayo_roi_temporal_v2_rosbag_20260825`

## Postprocessing configuration

- CAM4 workspace: Mayo stand
- Normalized ROI polygon: `(0.402, 0.215)`, `(0.698, 0.197)`, `(0.705, 0.651)`,
  `(0.409, 0.663)`
- Accept an instance only when at least 50% of its mask lies inside the ROI and the mask
  centroid is inside the ROI
- Temporal history: 7 frames; class switch requires at least 3 observations and a weighted
  confidence margin of 0.20
- Class-independent association: mask IoU >= 0.10 or bbox IoU >= 0.20, normalized centroid
  distance <= 0.06, mask-area ratio <= 3.0, maximum 3 missed frames

CAM3 is the tray camera from the August setup. Its ROI remains disabled until an August CAM3
image is delivered and the tray polygon can be calibrated from the actual view.

## Results

| Model | Input instances | ROI rejected | Output instances | Raw class transitions | Stable class switches | Mean pipeline latency |
|---|---:|---:|---:|---:|---:|---:|
| Small | 955 | 20 | 935 | 106 | 30 | 37.90 ms |
| Medium | 1268 | 76 | 1192 | 123 | 20 | 50.75 ms |
| Large | 1213 | 2 | 1211 | 85 | 23 | 56.09 ms |
| XLarge | 1581 | 4 | 1577 | 303 | 46 | 84.29 ms |

Compared with raw per-frame class transitions, stable output switches were 71.7% lower for
Small, 83.7% lower for Medium, 72.9% lower for Large, and 84.8% lower for XLarge. These are
temporal-consistency diagnostics, not proof that the stabilized class is correct. Ground-truth
evaluation and live camera acceptance remain required.

The ROI filter runs after RF-DETR inference, so it suppresses out-of-workspace publications but
does not reduce GPU inference time. Accepted masks are retained in full rather than clipped to
the ROI boundary.
