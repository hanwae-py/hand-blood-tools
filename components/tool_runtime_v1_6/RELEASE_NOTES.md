# Release notes — v1.6.0-rc1-compatible (2026-08-20)

- Detector checkpoint: `cam4_rfdetr_seg_small_regular_resume_e13_best.pth`
- Default confidence threshold 0.30 (was 0.50)
- Class-agnostic bounding-box NMS enabled by default (IoU 0.80)
- Pose-axis debug overlay on `/surgery/images/cam4/pose_overlay/compressed`
- Depth-to-color helpers for Hand/Blood (`metric_depth_in_rgb_frame` and related)
- Coordinator-compatible `processing_enabled` / `processing_gate_topic`

Input topics, native `16UC1 compressedDepth` decode, depth-to-color
registration, and `PLANAR_4DOF_WITH_NORMAL_PRIOR` pose semantics are unchanged
from the previous runtime.
