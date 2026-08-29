# Release notes — v1.6.0-rc1-compatible (2026-08-20)

- Detector checkpoint: `cam4_rfdetr_seg_small_regular_resume_best.pth`
- Default confidence threshold 0.30 (was 0.50)
- Class-agnostic bounding-box NMS enabled by default (IoU 0.80)
- Pose-axis debug overlay on `/surgery/images/cam4/pose_overlay/compressed`
- Depth-to-color helpers for Hand/Blood (`metric_depth_in_rgb_frame` and related)
- Coordinator-compatible `processing_enabled` / `processing_gate_topic`

## 2026-08-25 model-scale integration

- Added selectable `RFDETRSegMedium` and `RFDETRSegLarge` loaders without
  removing the production Small path.
- Preserved Small's validated BGR + class-agnostic NMS behavior.
- Added Medium/Large RGB input conversion with no extra NMS.
- Added checkpoint hashes, server sanity results, validation metrics, and the
  Small BGR/RGB ablation metadata.

## 2026-08-25 XLarge integration

- Added `RFDETRSegXLarge` with RGB input, no extra NMS, and provisional
  rosbag39 threshold 0.40.
- Renamed the Small delivery artifact without an epoch suffix because the
  checkpoint records zero-based epoch index 13; `e13` was ambiguous.

## 2026-08-25 workspace and temporal postprocessing

- Corrected the camera-role contract to CAM4=Mayo stand and CAM3=tray.
- Added normalized polygon ROI filtering using mask-overlap and mask-centroid
  acceptance without clipping accepted tool masks.
- Added class-independent mask/bbox/centroid temporal association and a
  confidence-weighted recent-history class switch hysteresis.
- Added explicit per-view ROI profile selection for both rosbag replay and
  live ROS processing. The reference runner selects the provisional
  `cam4_20260814_mayo` profile; general live runs default to `none` until a
  matching installation profile is calibrated. CAM3 tray ROI remains disabled
  pending delivery of August CAM3 images.

Input topics, native `16UC1 compressedDepth` decode, depth-to-color
registration, and `PLANAR_4DOF_WITH_NORMAL_PRIOR` pose semantics are unchanged
from the previous runtime.

## 2026-08-26 selected XLarge checkpoint and August RGB profiles

- Updated the selectable XLarge artifact to
  `checkpoint_selected_external_0825_holdout_conf030.pth` (SHA-256
  `694f92197ab0d1441eafc8fa4ecaafc29b16f6893b23476e9934e79905d67aeb`).
- Added separate provisional ROI profiles for the 2026-08-25 CAM3 tray and
  CAM4 Mayo RGB views. They are dataset/view-specific and are not live-camera
  calibration substitutes.
- Added reproducible RGB-video and native-depth MCAP evaluators. The latter
  reports constrained planar pose validity, not unconstrained 6D accuracy.

## 2026-08-26 XLarge-first runtime policy

- Changed Tool runtime and library defaults from Small to XLarge.
- Defined the measured-latency step-down order as XLarge, Large, then Medium.
- Small remains available only as an explicit legacy selection.

## 2026-08-26 Final Overlay Tool ROI

- The CAM3/CAM4 Final Overlay now draws each Tool worker's selected recognition
  ROI polygon and profile name directly on the corresponding source image.
- Live CAM3/CAM4 recognition now requires at least 70 percent mask overlap plus
  an inside mask centroid, instead of the previous 50 percent overlap.
