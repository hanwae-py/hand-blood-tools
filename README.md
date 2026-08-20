# Surgical Perception System v1

Portable source bundle for the three perception algorithms and their ROS 2
take-turn controller.

## Included components

| Component | Version included | Role |
|---|---|---|
| `components/tool_runtime_v1_4` | Tool v1.4 | RGB Tool segmentation; optional depth sample + planar 4-DOF-with-normal-prior pose |
| `components/hand_keypoints_ros` | current | MediaPipe hand keypoints + depth-derived hand pose |
| `components/blood_detection` | current | RF-DETR Seg-Small 2D Blood mask/blue overlay; optional centroid depth |
| `components/coordinator_ws` | current | `DETECT_TOOL`, `DETECT_HAND`, `DETECT_BLOOD`, `STOP` lifecycle selector |

This folder deliberately does **not** include checkpoints, MCAP, H5, AVI,
recorded images, Python virtual environments, or ROS `build/install/log`
folders. They must be supplied locally on the destination PC.

## Required external checkpoints

Tool current weights:
[`cam4_rfdetr_seg_small_v1.pth`](https://drive.google.com/file/d/1Rscaaa2l4r9FUD4-s_LXWwETCOWJkNnt/view?usp=drive_link).
Newer Tool checkpoints will be added to
[the same Drive folder](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h).
Copy the `.pth` you want onto the destination PC and set
`TOOL_V14_CHECKPOINT` in `config/system.env`.

Blood:
[RF-DETR Seg-Small checkpoint](https://drive.google.com/file/d/1Srkw_3K3Feb7FyTy7kNv-eCF0Ev-W773/view).
Set `BLOOD_CHECKPOINT` in `config/system.env` to that local `.pth` path.

## First setup on another PC

Commands below assume the repository root after clone:

```bash
git clone https://github.com/hanwae-py/hand-blood-tools.git
cd hand-blood-tools
```

1. Install Ubuntu/WSL, ROS 2 Jazzy, NVIDIA CUDA/WSL GPU support, and the
   Python dependencies described by the Hand and Tool component documentation.
2. Create two Python 3.12 environments, one for Hand and one for Tool/Blood.
   Do not combine them: Hand pins `torch==2.11.0`, Tool/Blood pins
   `torch==2.7.0+cu118` and `rfdetr==1.8.3`. Use conda or venv.

   Conda:

   ```bash
   conda env create -f components/hand_keypoints_ros/environment.yml
   conda env create -f components/tool_runtime_v1_4/algorithm/environment/environment.yml
   conda run -n rfdetr pip install -e components/tool_runtime_v1_4/algorithm
   ```

   Point `HAND_PYTHON` and `RFDETR_PYTHON` at
   `$(conda info --base)/envs/hand/bin/python` and
   `$(conda info --base)/envs/rfdetr/bin/python`.

   venv (created inside this clone):

   ```bash
   python3.12 -m venv .venv-hand
   .venv-hand/bin/pip install -r components/hand_keypoints_ros/requirements.txt

   python3.12 -m venv .venv-rfdetr
   .venv-rfdetr/bin/pip install -r components/tool_runtime_v1_4/algorithm/environment/requirements-reference-cu118.txt
   .venv-rfdetr/bin/pip install -e components/tool_runtime_v1_4/algorithm
   ```

   Point `HAND_PYTHON` and `RFDETR_PYTHON` at
   `$PWD/.venv-hand/bin/python` and `$PWD/.venv-rfdetr/bin/python`.

   Do not `conda activate` or `source` the venv in the ROS `colcon` build
   shell. The run scripts only need those two interpreter paths.
3. Copy the required checkpoints/data outside this source folder.
4. Create the local configuration file:

```bash
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

## Depth

All three nodes can consume the same RealSense metric depth stream.
The live/MCAP contract is RealSense `16UC1` `compressedDepth` on
`DEPTH_TOPIC` (default `/synced/cam_4/depth/image_rect_raw/compressedDepth`).
Each uint16 unit is millimetres and is converted with `0.001 m/unit`. This is
**not** normalized monocular depth.

| Node | RGB without depth | When matching depth is present |
|---|---|---|
| Tool | Always runs RF-DETR and publishes 2D at the longitudinal-axis midpoint | Samples metric depth at that UV; with camera infos also runs planar 4-DOF pose |
| Blood | Always runs RF-DETR and publishes 2D masks | Samples metric depth at the mask centroid (`centroid_depth_m`); no RGB-D registration and no suction pose |
| Hand | 2D MediaPipe keypoints | Back-projects keypoints with RealSense metric depth |

Tool and Blood default to `require_depth:=false`: missing depth skips those
metric fields and still publishes 2D. Set `require_depth:=true` to drop
frames that have no usable depth.

Hand's run script sets `depth_source:=real`, so the deploy path is the
RealSense `compressedDepth` topic, not Depth-Anything V2. `depth_source:=mono`
(or `auto` when no depth publisher is visible) is a relative monocular
fallback only. `run_hand_cam4.sh` also sets
`depth_alignment_validated:=false`, so Hand keeps 2D and leaves 3D invalid
until alignment is explicitly approved.

Tool pose additionally needs a confirmed `depth_scale_m_per_unit`, RGB-depth
extrinsics, and support-plane parameters. The reference config uses
`0.001 m/unit` with `depth_scale_verified: false` until the camera provider
confirms the device scale.

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
- Valid Tool pose and Hand 3D both need confirmed metric scale and RGB-depth
  alignment; the current run defaults leave those flags false.
- Blood currently publishes 2D masks plus an optional centroid depth sample.
  A 3D robot suction pose needs a separate depth-based target-selection step.
