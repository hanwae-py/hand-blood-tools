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

- Checkpoint: `cam4_rfdetr_seg_small_regular_resume_e13_best.pth`
- Download: [Google Drive folder](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h). Set the local `.pth` path as `TOOL_CHECKPOINT` in `config/system.env`.
- Default confidence threshold: 0.30
- Class-agnostic bounding-box NMS: enabled, IoU 0.80
- Pose-axis debug overlay: `/surgery/images/cam4/pose_overlay/compressed`

## Build

From this directory:

    bash scripts/build_ros2.sh

The repository-root `bash scripts/build_all.sh` also builds this workspace.

## Run

From the repository root:

    bash scripts/run_tool_v16.sh
