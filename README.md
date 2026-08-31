# Surgical Perception System v1

Portable source bundle for the three perception algorithms, a hand-state
postprocessor handoff, and their ROS 2 take-turn controller.

## Included components

| Component | Version included | Role |
|---|---|---|
| `components/tool_runtime_v1_6` | Tool v1.6 | Selectable RF-DETR Seg Small/Medium/Large/XLarge Tool segmentation; optional depth sample + planar 4-DOF-with-normal-prior pose |
| `components/hand_keypoints_ros` | current | MediaPipe hand keypoints + depth-derived hand pose |
| `components/hand_open_close` | offline-tested handoff | ROS-independent anatomical `OPEN`/`CLOSED` rule from 21 hand keypoints; proposed ROS adaptation is not yet tested |
| `components/blood_detection` | RF-DETR + Cutie | Fused RF-DETR Seg-Small + Cutie 2D Blood mask/blue overlay; optional centroid depth |
| `components/coordinator_ws` | current | `DETECT_TOOL`, `DETECT_HAND`, `DETECT_BLOOD`, `STOP` lifecycle selector |

This folder deliberately does **not** include checkpoints, MCAP, H5, large
recorded datasets, Python virtual environments, ROS `build/install/log`
folders, or the Blood `third_party` clones (`rfdetr`, `cutie`). They must be
supplied locally on the destination PC. The documented 22-second fixture
under `components/hand_open_close/test_data` is the only small recorded-video
exception. This repo ships the empty folder
`components/blood_detection/third_party/` only; clone the two trees into it
as described below.

## Required external checkpoints

Tool weights are selectable without removing the existing Small model:
`cam4_rfdetr_seg_small_regular_resume_best.pth`, `medium_best.pth`,
`large_best.pth`, or `checkpoint_selected_external_0825_holdout_conf030.pth`.
Download Tool checkpoints from
[this Drive folder](https://drive.google.com/drive/folders/1E42Cpgg8CbFRtnA8DuFbYeBT5IWx_G_h).
Copy the `.pth` files onto the destination PC and set `TOOL_MODEL_SIZE` plus
the matching `TOOL_CHECKPOINT_SMALL`, `TOOL_CHECKPOINT_MEDIUM`, and
`TOOL_CHECKPOINT_LARGE`, and `TOOL_CHECKPOINT_XLARGE` paths in
`config/system.env`. The legacy
`TOOL_CHECKPOINT` variable remains a Small fallback.

Small preserves its validated BGR input and class-agnostic NMS contract.
Medium, Large, and XLarge use RF-DETR's RGB NumPy input contract and no extra
NMS. XLarge is the default; if measured latency is too high, step down to
Large and then Medium. For example, select XLarge explicitly with:

```bash
TOOL_MODEL_SIZE=xlarge bash scripts/run_tool_v16.sh cam_4
```

Tool postprocessing supports a camera-calibrated workspace polygon and
class-independent mask/bbox association with temporal class smoothing. CAM4
is the Mayo-stand camera and CAM3 is the August tray camera. Dataset and live
camera views select separate ROI profile YAML files with `TOOL_ROI_PROFILE`.
The bundled `cam4_20260814_mayo` profile is provisional and must not be reused
for a different live view without calibration. CAM3 ROI stays disabled until
CAM3 August image data is received and calibrated.

Blood uses a fused **RF-DETR Seg-Small + Cutie** pipeline. This repo does not
vendor those sources. Create the empty folder (already present after clone)
and clone into it:

```bash
mkdir -p components/blood_detection/third_party
git clone https://github.com/roboflow/rf-detr.git components/blood_detection/third_party/rfdetr
git clone https://github.com/hkchengrex/Cutie.git components/blood_detection/third_party/cutie
```

Deploy tree: RF-DETR **1.10.0.dev**. Exact commit/tag: **TBD**.
Do this before `components/blood_detection/setup_env.sh`.

Download the two live weights from Google Drive (**TBD**; fill in after upload)
into `$HOME/models/` or `components/blood_detection/pretrained/`:

- `detr_blood.pth` — Drive URL **TBD**
- `cutie_blood.pth` — Drive URL **TBD**

Set `BLOOD_PYTHON`, `BLOOD_CHECKPOINT`, and `BLOOD_CUTIE_CHECKPOINT` in
`config/system.env`. Do not point Blood at `RFDETR_PYTHON` or the old
single-file `blood_detection.pth` overlay. See
[components/blood_detection/README.md](components/blood_detection/README.md).

The previous detector-only
[RF-DETR Seg-Small checkpoint](https://drive.google.com/file/d/1Srkw_3K3Feb7FyTy7kNv-eCF0Ev-W773/view)
is not the live fused overlay.

## First setup on another PC

Commands below assume the repository root after clone:

```bash
git clone https://github.com/hanwae-py/hand-blood-tools.git
cd hand-blood-tools
```

1. Install Ubuntu 24.04, ROS 2 Jazzy, NVIDIA CUDA, and the Python
   dependencies described by the Hand and Tool component documentation.
   The deploy scripts target a native Ubuntu PC, not WSL.
2. Create three Python 3.12 environments: Hand, Tool, and Blood.
   Do not combine them: Hand pins `torch==2.11.0`, Tool pins
   `torch==2.7.0+cu118` and `rfdetr==1.8.3`, Blood pins `torch==2.6.0+cu124`
   with cloned RF-DETR 1.10.0.dev + Cutie. Use conda or venv.

   Clone Blood third-party trees into the empty folder first:

   ```bash
   mkdir -p components/blood_detection/third_party
   git clone https://github.com/roboflow/rf-detr.git components/blood_detection/third_party/rfdetr
   git clone https://github.com/hkchengrex/Cutie.git components/blood_detection/third_party/cutie
   ```

   Conda:

   ```bash
   conda env create -f components/hand_keypoints_ros/environment.yml
   conda env create -f components/tool_runtime_v1_6/algorithm/environment/environment.yml
   conda run -n rfdetr pip install -e components/tool_runtime_v1_6/algorithm
   conda env create -f components/blood_detection/environment.yml
   conda run -n blood bash components/blood_detection/setup_env.sh
   ```

   Point `HAND_PYTHON`, `RFDETR_PYTHON`, and `BLOOD_PYTHON` at
   `$(conda info --base)/envs/hand/bin/python`,
   `$(conda info --base)/envs/rfdetr/bin/python`, and
   `$(conda info --base)/envs/blood/bin/python`.

   If the name `blood` is already used by the BloodDetection training env,
   create `blood-ros` instead and set `BLOOD_PYTHON` to that interpreter.

   venv (created inside this clone):

   ```bash
   python3.12 -m venv .venv-hand
   .venv-hand/bin/pip install -r components/hand_keypoints_ros/requirements.txt

   python3.12 -m venv .venv-rfdetr
   .venv-rfdetr/bin/pip install -r components/tool_runtime_v1_6/algorithm/environment/requirements-reference-cu118.txt
   .venv-rfdetr/bin/pip install -e components/tool_runtime_v1_6/algorithm

   python3.12 -m venv .venv-blood
   source .venv-blood/bin/activate
   bash components/blood_detection/setup_env.sh
   deactivate
   ```

   Point `HAND_PYTHON`, `RFDETR_PYTHON`, and `BLOOD_PYTHON` at
   `$PWD/.venv-hand/bin/python`, `$PWD/.venv-rfdetr/bin/python`, and
   `$PWD/.venv-blood/bin/python`.

   Do not `conda activate` or `source` the venv in the ROS `colcon` build
   shell. The run scripts only need those interpreter paths.
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

VIPLab publishes `/synced/cam_4` 1:1. QoS is reliable / volatile / KEEP_LAST 20.
`config/system.env` names those source topics. One ingress process subscribes
to them and fans out locally. Tool, Hand, and Blood read
`/perception/ingress/cam_4/...`, not `/synced` directly.

| Role | VIPLab topic | Local worker topic |
|---|---|---|
| CAM4 RGB | `/synced/cam_4/color/image_raw/compressed` | `/perception/ingress/cam_4/color/image_raw/compressed` |
| CAM4 color CameraInfo | `/synced/cam_4/color/camera_info` | `/perception/ingress/cam_4/color/camera_info` |
| CAM4 depth | `/synced/cam_4/depth/image_rect_raw/compressedDepth` | `/perception/ingress/cam_4/depth/image_rect_raw/compressedDepth` |
| CAM4 depth CameraInfo | `/synced/cam_4/depth/camera_info` | `/perception/ingress/cam_4/depth/camera_info` |

| Mode | What starts | Configuration needed |
|---|---|---|
| Real camera | Camera publisher already running on ROS | `config/system.env` camera topic names/QoS/domain |
| Offline RGB/H5 video | Hand fake-camera publisher | RGB AVI, depth H5, calibration JSON paths |
| MCAP | `ros2 bag play` publishes the recorded topics | `MCAP_PATH` |

`config/input_profiles/` contains templates for each mode. Copy values from
the selected profile into `config/system.env`; never edit component source just
to change an input path or topic.

Offline RGB/H5 caveat: the current fake-camera publisher emits raw images.
Hand supports that input directly. Tool v1.6 requires compressed RGB and
`compressedDepth`, so use MCAP replay for a complete Tool/Hand/Blood test, or
add a raw-to-compressed RGB-D bridge before running Tool v1.6 on AVI/H5.

## Depth

All three nodes can consume the same RealSense metric depth stream.
The live/MCAP depth topic is VIPLab `/synced/cam_4/depth/image_rect_raw/compressedDepth`
(`sensor_msgs/CompressedImage`, `16UC1` `compressedDepth`). Each uint16 unit is millimetres
and is converted with `0.001 m/unit`. This is **not** normalized monocular depth.

| Node | RGB without depth | When matching depth is present |
|---|---|---|
| Tool | Always runs RF-DETR and publishes 2D at the longitudinal-axis midpoint | Samples metric depth at that UV; with camera infos also runs planar 4-DOF pose |
| Blood | Always runs fused RF-DETR + Cutie and publishes 2D masks | Samples metric depth at the mask centroid (`centroid_depth_m`) in the RGB frame (same HxW, or depth-to-color registration). No suction pose |
| Hand | 2D MediaPipe keypoints | Back-projects keypoints with RealSense metric depth in the RGB frame (same HxW, or depth-to-color registration) |

Tool defaults: `confidence_threshold` 0.30 and `require_depth:=false`. Small
uses class-agnostic bounding-box NMS (IoU 0.80); larger variants use no
additional NMS. Missing depth skips metric fields and still publishes 2D. Set
`require_depth:=true` to drop frames that have no usable depth. Tool also publishes a pose-axis overlay on
`/perception/cam_4/tool/pose_overlay/compressed` when a pose result exists.

Hand's run script sets `depth_source:=real`, so the deploy path is the
RealSense `compressedDepth` topic, not Depth-Anything V2. `depth_source:=mono`
(or `auto` when no depth publisher is visible) is a relative monocular
fallback only. `run_hand_cam4.sh` also sets
`depth_alignment_validated:=false`, so Hand keeps 2D and leaves 3D invalid
until alignment is explicitly approved. In that pending state it uses an
RGB-sized NaN depth map, so RGB UVs are never clipped into a smaller native
depth image. After approval, native depth that is not already RGB-sized is
registered with `config/cam4_depth_to_color.yaml` (the same CAM4 extrinsics
as Tool).

Tool pose additionally needs a confirmed `depth_scale_m_per_unit`, RGB-depth
extrinsics, and support-plane parameters. The reference config uses
`0.001 m/unit` with `depth_scale_verified: false` until the camera provider
confirms the device scale.

## Main commands

One command starts the whole stack: CAM3 and CAM4 ingress, CAM3 Tool, the
concurrent CAM4 Tool, Hand, and Blood workers, and the final Debug overlay.

```bash
bash scripts/run.sh
```

Ingress is the only VIPLab `/synced` subscriber, so `run.sh` exits without
starting anything when an ingress process is already running.
`config/systemd/user/` holds the persistent user units that run the same set;
install them with `scripts/install_perception_user_units.sh`.

```bash
bash scripts/stop_local_perception.sh
```

That stops the local processes only. Units started by systemd have
`Restart=always`, so stop the target instead:
`systemctl --user stop taskplanner-perception-stack.target`.

### One worker at a time

For debugging a single algorithm. Ingress must already be running, because each
worker reads only `/perception/ingress/<cam>` topics.

```bash
bash scripts/run_perception_ingress.sh both   # or cam_3, or cam_4
bash scripts/run_tool_v16.sh cam_4            # or cam_3
bash scripts/run_hand_cam4.sh cam_4
bash scripts/run_blood_cam4.sh flir           # integrated default
bash scripts/run_blood_cam4.sh cam_4          # single-worker debug
bash scripts/run_final_overlay.sh
```

`scripts/list_topics.sh` lists topics that currently have a publisher, and
`scripts/view_overlay.sh cam_4 tool hand` opens the JPEG overlays.

## Important limits

- Tool v1.6 pose is `PLANAR_4DOF_WITH_NORMAL_PRIOR`, not unrestricted full 6D.
- Class-agnostic NMS can suppress overlapping boxes even when their class IDs
  differ.
- Workspace ROI coordinates are camera-installation specific. Recalibrate
  after moving the camera, Mayo stand, or tray. Live Tool runs default to
  `TOOL_ROI_PROFILE=none`; explicitly select the calibrated live profile.
- Valid Tool pose and Hand 3D both need confirmed metric scale and RGB-depth
  alignment; the current run defaults leave those flags false.
- Blood currently publishes fused 2D masks plus an optional centroid depth sample.
  A 3D robot suction pose needs a separate depth-based target-selection step.
  Blood shares the GPU with Tool/Hand when they run together; it does not share
  their Python environments.
