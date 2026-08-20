# Surgical Perception System v1

Portable source bundle for the three perception algorithms and their ROS 2
take-turn controller.

## Included components

| Component | Version included | Role |
|---|---|---|
| `components/tool_runtime_v1_4` | Tool v1.4 | RGB-D Tool segmentation + planar 4-DOF-with-normal-prior pose |
| `components/hand_keypoints_ros` | current | MediaPipe hand keypoints + depth-derived hand pose |
| `components/blood_detection` | current | RF-DETR Seg-Small 2D Blood mask/blue overlay |
| `components/coordinator_ws` | current | `DETECT_TOOL`, `DETECT_HAND`, `DETECT_BLOOD`, `STOP` lifecycle selector |

This folder deliberately does **not** include checkpoints, MCAP, H5, AVI,
recorded images, Python virtual environments, or ROS `build/install/log`
folders. They must be supplied locally on the destination PC.

## Required external checkpoint

Download the Blood RF-DETR Seg-Small checkpoint from
[Google Drive](https://drive.google.com/file/d/1Srkw_3K3Feb7FyTy7kNv-eCF0Ev-W773/view),
then set `BLOOD_CHECKPOINT` in `config/system.env` to its local `.pth` path.

Download the Tool v1.4 RF-DETR Seg-Small checkpoint
[`cam4_rfdetr_seg_small_v1.pth` from Google Drive](https://drive.google.com/file/d/1oXid9UuSCEgOwCzDWOU8CJeDE8t0fL4y/view?usp=drive_link),
then set `TOOL_V14_CHECKPOINT` in `config/system.env` to its local `.pth` path.

## First setup on another PC

1. Install Ubuntu/WSL, ROS 2 Jazzy, NVIDIA CUDA/WSL GPU support, and the
   Python dependencies described by the Hand and Tool component documentation.
2. Create the RF-DETR and Hand Python virtual environments.
3. Copy the required checkpoints/data outside this source folder.
4. Create the local configuration file:

```bash
cd ~/surgical_perception_system_v1
cp config/system.env.example config/system.env
nano config/system.env
```

5. Build all ROS workspaces:

```bash
bash scripts/build_all.sh
```

## Input modes

| Mode | What starts | Configuration needed |
|---|---|---|
| Real camera | Camera publisher already running on ROS | `config/system.env` camera topic names/QoS/domain |
| Offline RGB/H5 video | Hand fake-camera publisher | RGB AVI, depth H5, calibration JSON paths |
| MCAP | `ros2 bag play` publishes the recorded topics | `MCAP_PATH` |

`config/input_profiles/` contains templates for each mode. Copy values from
the selected profile into `config/system.env`; never edit component source just
to change an input path or topic.

Offline RGB/H5 caveat: the current fake-camera publisher emits raw images.
Hand supports that input directly. Tool v1.4 requires compressed RGB and
`compressedDepth`, so use MCAP replay for a complete Tool/Hand/Blood test, or
add a raw-to-compressed RGB-D bridge before running Tool v1.4 on AVI/H5.

## Main commands

### Tool v1.4 only

```bash
bash scripts/run_tool_v14.sh
```

### Hand only

```bash
bash scripts/run_hand_cam4.sh
```

### Blood only

```bash
bash scripts/run_blood_cam4.sh
```

### Three-algorithm trigger test with MCAP

```bash
bash scripts/run_three_algorithm_trigger_mcap.sh
```

### Three-algorithm trigger test with a live camera publisher

```bash
bash scripts/run_three_algorithm_trigger_live.sh
```

In another sourced terminal, send `DETECT_TOOL`, `DETECT_HAND`,
`DETECT_BLOOD`, or `STOP` on `/surgery/perception/mode_command`.

## Important limits

- Tool v1.4 pose is `PLANAR_4DOF_WITH_NORMAL_PRIOR`, not unrestricted full 6D.
- Valid Tool pose needs verified depth scale, RGB-depth registration/extrinsic,
  and support-plane parameters.
- Blood currently publishes 2D masks. A 3D robot suction pose needs a separate
  depth-based target-selection step.
