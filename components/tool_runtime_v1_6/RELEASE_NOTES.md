# Release notes — v1.6.0-rc1-compatible (2026-08-20)

## 2026-09-02 stationary-first control TF stabilization

- Tuned CAM3/CAM4 control-facing position filtering for mostly stationary
  tools: 2 mm deadband and 0.10 EMA while preserving raw `ToolPoseArray`.
- Added a 1.5 degree angular deadband and 0.15 quaternion SLERP to reduce
  ordinary mask-axis attitude jitter.
- Required five consecutive, mutually consistent, orientation-valid opposite
  +Y observations before accepting a reversal for every surgical-tool class.
  Low-confidence samples hold the previous control-facing direction and reset
  the pending confirmation sequence.
- Added diagnostics and regression tests for angular hold, smoothing,
  persistent flip lock, and relocation reinitialization.

## 2026-09-02 endpoint-sign policy update

- Refined Bipolar colour evidence so black explicitly votes for the handle and
  blue explicitly votes for the working tip. Agreement receives full colour
  strength, a single cue is down-weighted, and conflicting cues cancel.
- Preserved the existing Bovie external-wire, tip-taper and terminal-mass
  fallback policy without parameter changes.
- Changed Bipolar from taper-only endpoint selection to a non-hierarchical
  signed-vote ensemble: connector taper 0.45, dark proximal colour 0.40 when
  available, and terminal mass 0.15. External wire remains excluded.
- Changed Adson to classify the whole-mask placement before looking for a
  separated jaw gap. An approximately linear, monotonically widening
  triangular silhouette selects its wider end as the working tip; other
  layouts continue through separated-jaw, taper and low-confidence fallback
  evidence.
- Added rotation-invariant regression coverage for triangular Adson masks,
  Bipolar colour votes, ensemble arithmetic, colour abstention and unchanged
  Bovie behavior.

## 2026-08-31 control-facing position stabilization

- Added spatial-selector-local translation stabilization for dynamic Tool TF:
  0.20 EMA and two-frame confirmation for jumps above 40 mm. An optional
  deadband defaults to zero to avoid persistent position bias.
- Preserved raw `ToolPoseArray` measurements and existing constrained planar
  4-DoF orientation semantics.
- Reset affected filter slots when same-class selector cardinality changes, so
  a left-to-right ordinal reassignment cannot inherit another tool's position.
- Added ROS-independent unit tests, an RGB rosbag mask-centroid proxy, and a
  frozen-detection MCAP RGB-D evaluator for repeatable parameter tuning.
- Replayed all 447 exact-hash CAM4 frames from
  `multicam_viplab_only_30s_20260814_134233.staging_0.mcap` with the v5
  XLarge `checkpoint_best_total.pth`. Consecutive non-relocation 3-D step p95
  decreased from 16.75 mm to 4.36 mm (74.0%). This is an unlabeled stability
  result, not absolute pose accuracy; the reference support plane and depth
  scale remain provisional.

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
