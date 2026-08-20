# Surgical Tool ROS2 Runtime v1.6 - Compatibility Variant

This is a local integration candidate. It combines the v1.6 detector release
with the current repository's coordinator-facing Tool behavior. It is not the
unchanged vendor v1.6 archive.

## What is retained from the current runtime

- processing_enabled and processing_gate_topic for Tool/Hand/Blood turn-taking;
- RGB-only 2D observations when require_depth is false;
- the observation_point_depth_m field in ToolObservation2D.

These preserve compatibility with the current coordinator and downstream ROS
consumers.

## v1.6 changes included

- Detector checkpoint: algorithm/model/cam4_rfdetr_seg_small_regular_resume_e13_best.pth
- Download: [Google Drive checkpoint](https://drive.google.com/file/d/13JW_AVPgiJZ_XdWmOReSeSCg2d35wHSC/view?usp=drive_link). Place it at the path above after cloning.
- Default confidence threshold: 0.30
- Class-agnostic bounding-box NMS: enabled, IoU 0.80
- New pose-axis debug overlay: /surgery/images/cam4/pose_overlay/compressed

Input topics, native 16UC1 compressedDepth decoding, depth-to-color
registration, and constrained planar pose semantics remain compatible with
the current runtime.

## Build

From this directory:

    bash scripts/build_ros2.sh

## Run

From the repository root:

    bash scripts/run_tool_v16.sh

The v1.4 launcher and runtime remain unchanged. Use v1.6 only after a
coordinated MCAP/live-input test confirms its behavior in the full system.

## Validation completed locally

- ROS workspace build: surgical_perception_msgs and pnu_surgical_perception
- ROS unit tests: 10 passed
- Class-agnostic NMS validation
- Pose contract and native-depth registration validations

No Git commit or push was performed for this local candidate.
