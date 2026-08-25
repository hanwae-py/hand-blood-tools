# Surgical Tool ROS2 Runtime v1.6

This directory is the only Tool runtime in the repository. It combines the
v1.6 detector release with coordinator turn-taking
(`processing_enabled` / `processing_gate_topic`) and the depth-to-color
helpers used by Hand and Blood.

## Behavior

- processing_enabled and processing_gate_topic for Tool/Hand/Blood turn-taking;
- RGB-only 2D observations when require_depth is false;
- the observation_point_depth_m field in ToolObservation2D;
- native 16UC1 compressedDepth decoding and depth-to-color registration;
- constrained planar pose (`PLANAR_4DOF_WITH_NORMAL_PRIOR`).

## v1.6 detector

- Checkpoint: selectable Small `cam4_rfdetr_seg_small_regular_resume_best.pth`,
  Medium `medium_best.pth`, Large `large_best.pth`, or XLarge `xlarge_best.pth`
- Download: [Google Drive folder](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h).
  Set `TOOL_MODEL_SIZE` and the matching checkpoint path in `config/system.env`.
- Default confidence threshold: 0.30
- Small: legacy BGR input and class-agnostic bbox NMS IoU 0.80
- Medium/Large/XLarge: RGB input and no additional NMS
- Mask-overlap workspace ROI filtering for the CAM4 Mayo stand
- Class-independent mask/bbox association and confidence-weighted recent
  class smoothing (three-frame switch confirmation)
- CAM3 is the August tray camera; its ROI remains disabled until CAM3 August
  images are delivered and calibrated
- Pose-axis debug overlay: `/perception/cam_4/tool/pose_overlay/compressed`

## Build

From this directory:

    bash scripts/build_ros2.sh

The repository-root `bash scripts/build_all.sh` also builds this workspace.

## Run

From the repository root, after CAM4 ingress is running
(`bash scripts/run_perception_ingress.sh`):

    TOOL_MODEL_SIZE=xlarge bash scripts/run_tool_v16.sh
