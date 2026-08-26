# Selected XLarge checkpoint evaluation (2026-08-26)

## Scope and interpretation

- Checkpoint: `checkpoint_selected_external_0825_holdout_conf030.pth`
- Architecture: `RFDETRSegXLarge`, native resolution 624, RGB checkpoint contract
- SHA-256: `694f92197ab0d1441eafc8fa4ecaafc29b16f6893b23476e9934e79905d67aeb`
- Confidence threshold: 0.30
- Postprocessing: camera-specific polygon ROI plus seven-frame class evidence history,
  three-observation switch confirmation and 0.20 weighted score margin
- The RGB videos and reference MCAP have no detection or pose ground truth. The values below
  describe predictions, filtering and output validity; they are not accuracy, precision,
  recall, IoU, mAP or metric pose error.

## 2026-08-25 Arpa CAM3/CAM4 RGB

The supplied videos total about 73.3 minutes per camera. For a reproducible first-pass test,
three contiguous 30-second windows centered at 25%, 50% and 75% of each of the three sessions
were selected. Temporal state was reset at every window boundary. This is 4,050 frames per
camera and is not a full-video evaluation.

CAM3 is the tray view. Its provisional normalized polygon is `(0.245, 0.305)`,
`(0.725, 0.300)`, `(0.760, 0.860)`, `(0.200, 0.860)`.

CAM4 is the Mayo view. The initial provisional polygon was rejected after visual review. The
final polygon was drawn directly by the operator on the supplied CAM4 frame:
`(0.430469, 0.197222)`, `(0.693750, 0.188889)`, `(0.705469, 0.695833)`,
`(0.439844, 0.733333)`. It intentionally covers the blue Mayo drape and excludes the left
side table and right laptop/glove area.

| Camera | Frames | Raw instances | ROI rejected | Output instances | Raw class transitions | Stable class switches | Mean inference |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM3 tray | 4,050 | 10,956 | 4,366 (39.9%) | 6,590 | 128 | 22 | 85.98 ms |
| CAM4 Mayo, operator ROI | 4,050 | 13,624 | 3,276 (24.0%) | 10,348 | 181 | 33 | 88.31 ms |

The stable-switch count is 82.8% lower than raw transitions for CAM3 and 81.8% lower for
CAM4. This measures temporal consistency only; it does not prove the retained class is correct.
The ROI runs after inference, so it changes published instances but does not reduce RF-DETR GPU
inference time.

Local result directories:

- CAM3: `experiments/arpa_sharing_rgb_selected_xlarge_t030_roi_temporal_20260826`
- CAM4 operator ROI: `experiments/arpa_sharing_rgb_selected_xlarge_t030_user_roi_temporal_20260826`

## 2026-08-14 reference RGB-D MCAP

The previous and selected XLarge checkpoints were run on the same 447 synchronized CAM4
RGB/native-depth pairs with threshold 0.30, `cam4_20260814_mayo`, the same temporal settings,
depth-to-color calibration topic and provisional support plane.

| Metric | Previous `xlarge_best.pth` | Selected external checkpoint |
|---|---:|---:|
| Output instances | 1,577 | 1,437 |
| Raw class transitions | 303 | 184 |
| Stable class switches | 46 | 33 |
| `VALID` pose outputs | 1,235 / 1,577 (78.31%) | 1,114 / 1,437 (77.52%) |
| Position and orientation flags both valid | 1,276 / 1,577 (80.91%) | 1,162 / 1,437 (80.86%) |
| Detection + pose mean | 147.27 ms | 137.87 ms |
| Whole registration + detection + pose mean | 192.43 ms | 182.36 ms |

The selected checkpoint produced 8.9% fewer instances, 39.3% fewer raw class transitions and
28.3% fewer stable switches. Its pose-valid fraction is effectively unchanged, while mean
detection-plus-pose time is 6.4% lower because fewer masks proceed to pose estimation. Without
ground truth, the reduced instance count cannot be classified as removal of false positives or
an increase in false negatives.

All 447 RGB-depth pairs were matched (maximum timestamp delta 63,233 ns). The configured depth
scale remains unverified and the support plane remains provisional. The output is
`PLANAR_4DOF_WITH_NORMAL_PRIOR`: position plus planar heading represented by a quaternion, not
unconstrained full 6D pose.

Local comparison directory:
`experiments/rfdetr_xlarge_pose_checkpoint_comparison_20260826`.
