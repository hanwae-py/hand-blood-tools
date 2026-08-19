# Hand keypoints + palm 6D pose pipeline

Single-camera hand tracking that outputs, per frame: 21 metric 3D hand
joints (in the camera's optical frame, metres) + a 6-DoF palm pose
(translation + rotation) + left/right handedness.

2D detection is **MediaPipe Hand Landmarker**, running on GPU (TFLite/OpenGL
delegate, falls back to CPU automatically if no GPU delegate is available).
Depth comes from whichever source is actually available for a given clip:

- **Real depth** (e.g. a RealSense-style aligned depth stream) — used
  automatically whenever the pipeline can find one.
- **Monocular depth fallback** ([Depth-Anything V2, metric-indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf))
  — used automatically when no real depth is available, so the same
  script also works from RGB-only footage.

Every output JSON records which depth source was actually used
(`"depth_source"` field), so results are never ambiguous about how the
3D was obtained.

---

## Directory layout

```
hand_keypoints/
├── README.md
├── requirements.txt
├── scripts/
│   ├── hand_keypoints_core.py       ← shared detection/geometry logic (imported by BOTH consumers below)
│   └── run_hand_keypoints.py        ← offline CLI: reads video files, thin wrapper around core.py
├── ros2_ws/                         ← ROS2 (Jazzy) integration — see "ROS2 (Jazzy) integration" below
│   ├── Dockerfile
│   └── src/
│       ├── hand_keypoint_interfaces/  ← ament_cmake: custom .msg types only (HandKeypoints, Hand, PalmPose6D, Point2D)
│       └── hand_keypoint_ros/         ← ament_python: hand_detection_node, fake_camera_publisher
├── data/                            ← one example clip (see "Example data" below)
│   ├── rgb/gnu_0704_rgb_03.avi
│   ├── depth/gnu_0704_depth_03.avi        (depth as a viewable video — NOT used by the pipeline)
│   ├── depth_raw/gnu_0704_depth_raw_03.h5 (the actual depth input: aligned, per-pixel, uint16 mm)
│   └── calibration/gnu_0704_calibration_03.json
└── results/
    ├── example_output/              ← real depth, GPU, mono-depth fallback demo (see "Example data")
    │   ├── keypoints.json
    │   ├── overlay.gif               (15s preview of overlay.mp4 — full mp4 not committed, see below)
    │   └── depth_valid.csv
    └── example_output_cpu_realdepth/ ← same clip, real depth, --cpu-only (no GPU used at all)
        ├── keypoints.json
        ├── overlay.gif
        └── depth_valid.csv
```

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`torch` needs a build that matches your CUDA version — if `pip install torch==2.11.0` doesn't pull a CUDA build for your system, install it
from the selector at https://pytorch.org/get-started/locally/ instead.
Everything was developed/tested with CUDA 12.8 on an RTX PRO 6000.

The first run downloads the MediaPipe hand-landmarker `.task` file
(~7 MB, cached at `~/.cache/mediapipe/`) and, only if the monocular
fallback is actually used, the Depth-Anything V2 weights from
HuggingFace (cached at `~/.cache/huggingface/`). If real depth is found,
Depth-Anything V2 is never loaded — no download, no GPU memory used for it.

## Usage

**Real depth, auto-detected** — the script looks for
`<rgb's parent dir>/depth_raw/<same file name, "_rgb_" → "_depth_raw_">.h5`.
If found, it's used automatically:

```bash
python scripts/run_hand_keypoints.py \
  --rgb   data/rgb/gnu_0704_rgb_03.avi \
  --calib data/calibration/gnu_0704_calibration_03.json \
  --out   results/example_output
```

**RGB-only clip, no real depth available** — same command, nothing to
change. If no depth h5 is found next to the clip, the script logs
`depth source: MONOCULAR (Depth-Anything V2)` and proceeds automatically.

**Force monocular depth even if real depth exists** (e.g. to compare
both on the same clip):

```bash
python scripts/run_hand_keypoints.py --rgb ... --calib ... --out ... --force-mono-depth
```

**Explicit depth file** (skips auto-detection, useful for a different
directory layout):

```bash
python scripts/run_hand_keypoints.py --rgb ... --calib ... --depth path/to/depth.h5 --out ...
```

**CPU only, no GPU at all** (e.g. no GPU available, or verifying it
still runs without one):

```bash
python scripts/run_hand_keypoints.py --rgb ... --calib ... --out ... --cpu-only
```

Forces MediaPipe onto its CPU delegate directly (doesn't even attempt
the GPU delegate) and, if the mono-depth fallback is in use, forces that
model onto CPU too. Real depth mode has no other GPU dependency, so
`--cpu-only` + real depth is a fully GPU-free run end to end.

Other flags: `--cam-key` (which camera in the calibration JSON, default
`cam_4`), `--max-hands`, `--stride` (process every Nth frame),
`--flip-handedness` (see caveat below), `--region-x-min/max`,
`--region-y-min/max` (only keep hands whose 2D centroid falls in a
sub-rectangle of the frame, e.g. to isolate one person in a multi-person
scene).

### Robot handoff mode (`--robot-position`) — extra, opt-in

For the surgical-tool-handoff use case: keep only the **single hand
nearest the robot**, per frame, instead of every detected hand. This is
purely additive — omit the flag and you get the exact original
multi-hand behaviour (verified: identical output, this flag defaults to
off and short-circuits to a no-op).

```bash
python scripts/run_hand_keypoints.py \
  --rgb ... --calib ... --out ... \
  --robot-position top-left    # one of: top-left, top-right, bottom-left, bottom-right
```

`--robot-position` describes where the **robot** physically sits, not a
region of the frame. Each frame, every detected hand's 2D pixel centroid
is compared to a target pixel corner, and only the closest hand is kept
(both in the output JSON and drawn in the overlay).

**Corner-mapping caveat:** this camera is not mirrored like a selfie
camera (the same reason `--flip-handedness` exists — see below), so a
robot at physical position "top-left" is nearest to the hand that shows
up in the frame's **top-right** corner, not its top-left. The script
mirrors left/right (keeps top/bottom as-is) when converting
`--robot-position` to a target pixel, matching this lab's
already-validated `top-right`-quadrant convention from the earlier
single-hand pipeline. Only `top-left -> frame top-right` has actually
been confirmed against real footage; the other three follow the same
mirror by inference and should be checked against `overlay.mp4` (the
text overlay prints `robot=<pos> -> nearest <frame-corner>`) before
being trusted operationally. If your camera *is* mirrored, or the robot
sits somewhere the four corners don't describe well, adjust
`robot_position_target_px()` in the script directly.

## Calibration JSON format

The script reads camera intrinsics from:

```json
{ "camera_info": { "/synced/<cam_key>/color/camera_info": { "k": [fx, 0, cx, 0, fy, cy, 0, 0, 1] } } }
```

`--cam-key` selects which camera's intrinsics to use (default `cam_4`).
See `data/calibration/gnu_0704_calibration_03.json` for a real example.

## Real depth HDF5 format

A single dataset named `depth`, shape `(n_frames, H, W)`, dtype
`uint16`, values in millimetres, `0` = invalid/no return, already
pixel-aligned to the RGB stream. See `data/depth_raw/gnu_0704_depth_raw_03.h5`.

## Output — `keypoints.json`

```jsonc
{
  "video": "gnu_0704_rgb_03.avi",
  "n_frames": 2462,
  "fps": 15.0,
  "resolution": [1280, 720],
  "camera_used": "cam_4",
  "camera_intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
  "coordinate_frame": "cam_4 optical frame. Units: metres.",
  "depth_source": "realsense_h5_aligned_mm",   // or "monocular_Depth-Anything-V2-Metric-Indoor-Small-hf"
  "robot_handoff_mode": null,   // or {"robot_position": "top-left", "target_frame_corner": "top-right",
                                 //     "target_pixel": [1280.0, 0.0], "selection": "single nearest hand per frame..."}
                                 // — set only when --robot-position was passed
  "joint_names": ["wrist", "thumb_CMC", ..., "pinky_TIP"],   // 21 names, MediaPipe joint order
  "palm_6d_notes": { "formula_version": "v2 (2026-07-21 revision)", ... },
  "perf": {"infer_fps": 43.51, "source_fps": 15.0, "wall_time_s": 56.6},
  "frames": [
    {
      "frame_idx": 0, "t_s": 0.0,
      "hands": [
        {
          "hand_index": 0,
          "handedness": {"label": "Left", "score": 0.78},
          "joints_3d": [[x, y, z], ...],       // 21 x 3, metres, camera frame
          "joints_2d": [[u, v], ...],          // 21 x 2, pixels
          "kp_scores": [1.0, ...],             // MediaPipe doesn't expose per-joint confidence
          "kp_valid_depth": [true, ...],       // per-joint: did depth lookup succeed
          "palm_6d": {
            "translation": [x, y, z],          // midpoint(wrist, middle_MCP)
            "rotation_matrix": [[...],[...],[...]],  // columns: X=wrist->middle_MCP, Y=across palm, Z=palm normal
            "rotation_quat_wxyz": [w, x, y, z] // Hamilton convention
          }
        }
      ]
    }
  ]
}
```

`palm_6d` is `null` for a hand if any of joints `{wrist, index_MCP, middle_MCP, pinky_MCP}` lacks valid depth — the v2 formula needs all four.

## Measured performance (gnu_0704_rgb_03, full 2639/2462-frame clip)

| Depth source                       | Compute                                                                                  | infer_fps |   vs 15 fps source |
| ---------------------------------- | ---------------------------------------------------------------------------------------- | --------: | -----------------: |
| Real depth (h5, no ML depth model) | GPU (RTX PRO 6000, uncontended)                                                          |    ~42-44 | ~2.8-2.9x realtime |
| Real depth (h5, no ML depth model) | **CPU only** (`--cpu-only`), AMD EPYC 9554 (64c, dev server)                     |     15.73 |    ~1.05x realtime |
| Real depth (h5, no ML depth model) | **CPU only** (`--cpu-only`), Intel i5-14400F (10c/16t, 2.5GHz, Windows) local pc |     21.34 |    ~1.42x realtime |
| Monocular (Depth-Anything V2)      | GPU (RTX PRO 6000, uncontended)                                                          |    ~19-21 | ~1.3-1.4x realtime |

`infer_fps` is the **full pipeline**: depth lookup/inference + MediaPipe
detection + backprojection + palm-6D math + overlay drawing + file I/O,
not model-only throughput. Real depth is faster than monocular simply
because it skips running a depth neural net entirely — it's a direct
array lookup.
GPU numbers were measured with the pipeline running alone on an
otherwise idle GPU; running two instances concurrently on a
shared/loaded machine will reduce those figures (CPU-bound stages —
depth pre/post-processing, video I/O — contend for the same cores).

Interestingly, the 10-core/16-thread consumer i5 (21.34 fps) beat the
64-core server EPYC (15.73 fps) on the same `--cpu-only` real-depth
run — MediaPipe's CPU path doesn't scale with core count the way the
EPYC's core advantage would suggest; single/few-thread performance and
clock speed matter more here.

## Known caveats

- **Selfie-view handedness.** MediaPipe assumes a selfie/mirrored
  camera. A non-mirrored top-down or side-view camera may report L/R
  swapped — use `--flip-handedness` if so. Only the text label is
  affected; the 3D geometry (`joints_3d`, `palm_6d`) is unchanged either way.
- **Monocular depth is approximate.** Depth-Anything V2 was trained on
  general indoor scenes, not hand-scale sub-cm geometry — per-joint Z
  variation within a hand is mostly noise in that mode. Only the
  hand-centroid Z is reliable. Real depth does not have this limitation.
- **Real-depth stream can be shorter than the RGB stream** (sensor drop
  frames, sync boundary). The pipeline stops at the last frame with a
  matching depth entry and logs `no depth for frame N, stopping` — this
  is expected, not an error.
- **`--region-*` filters happen after full-frame detection**, i.e. they
  don't speed anything up — every hand in the frame is still detected
  and depth-sampled, only recording is filtered. Use them for
  multi-person scenes where you only want one person's hand(s) in the
  output.

## Example data

`data/` ships **one** example clip (`gnu_0704_rgb_03`) with its real
depth stream, so the pipeline can be run and verified end-to-end without
any other setup.

`results/example_output/` was deliberately generated with
`--force-mono-depth` (`depth_source: "monocular_Depth-Anything-V2-Metric-Indoor-Small-hf"`, **19.35 fps**,
136.4s for the full 2639-frame clip) to demonstrate the RGB-only
fallback path, even though this clip's real depth is right there in
`data/depth_raw/`. To see the real-depth result instead (faster, more
accurate), just drop the flag:

```bash
python scripts/run_hand_keypoints.py \
  --rgb   data/rgb/gnu_0704_rgb_03.avi \
  --calib data/calibration/gnu_0704_calibration_03.json \
  --out   results/example_output
```

— which reproduces the **43.51 fps** / `realsense_h5_aligned_mm` result
shown earlier in this README (deterministic given the same model weights).

`results/example_output_cpu_realdepth/` is the same real-depth run with
`--cpu-only` added — **15.73 fps**, no GPU touched at all (see the
performance table above). Reproduce with:

```bash
python scripts/run_hand_keypoints.py \
  --rgb   data/rgb/gnu_0704_rgb_03.avi \
  --calib data/calibration/gnu_0704_calibration_03.json \
  --out   results/example_output_cpu_realdepth \
  --cpu-only
```

**Note on repo size:** the three input files here are 281-553 MB each,
over GitHub's 100 MB per-file limit — they're tracked with
[Git LFS](https://git-lfs.com/) (see `.gitattributes`). Cloning this
repo requires `git lfs install` beforehand, otherwise `data/rgb/`,
`data/depth/`, and `data/depth_raw/` will come down as LFS pointer
files instead of real video/HDF5 data.

## ROS2 (Jazzy) integration

This is the **detection node only** — subscribe to camera topics, run
the same detection pipeline as `run_hand_keypoints.py` above, publish
results. Coordinate-frame transforms and robot control (the rest of a
camera → detection → TF → robot flow) are a separate, downstream node
and out of scope here.

### Why a separate Docker image

ROS2 Jazzy officially targets **Ubuntu 24.04**; this dev environment
(and the rest of this repo) runs Ubuntu 22.04. Rather than fighting
that mismatch, `ros2_ws/Dockerfile` builds a self-contained Ubuntu
24.04 image with ROS2 Jazzy + the same ML stack as `requirements.txt`
(mediapipe, torch, transformers, h5py — see the Dockerfile for the one
difference: `numpy`/`opencv-python` are left unpinned there, because
`mediapipe==0.10.18` requires `numpy<2`, which conflicts with the
`opencv-python==4.13.0.92` pin in `requirements.txt` for a plain venv).

`scripts/hand_keypoints_core.py` is the single source of truth for the
detection math — the Dockerfile copies it into the ROS package
(`hand_keypoint_ros/hand_keypoint_ros/core.py`) at image-build time,
so the CLI script and the ROS node can never silently drift apart.

### Package structure

Two packages, split by ROS convention — one for the message *types*,
one for the node *logic* — so any other node (e.g. a robot-control node
someone else writes) can depend on just the types without pulling in
mediapipe/torch/opencv:

- **`hand_keypoint_interfaces`** (`ament_cmake`) — only `.msg` files,
  no code: `HandKeypoints` (one frame's result), `Hand` (one detected
  hand), `PalmPose6D`, `Point2D`. See the `.msg` files themselves for
  the full field-by-field documentation.
- **`hand_keypoint_ros`** (`ament_python`) — `hand_detection_node`,
  `fake_camera_publisher`, the `core.py` copy, and the demo launch file.
  Depends on `hand_keypoint_interfaces`.

`hand/keypoints` carries **both** 2D and 3D together in one typed
message (`joints_2d` and `joints_3d` on the same `Hand`) — this
intentionally merges what an earlier planning diagram split into a
separate "2D detection" section and a "2D→3D conversion" section; no
extra conversion node is needed downstream. See `HandKeypoints.msg`'s
header comment for the reasoning.

### Build

```bash
cd hand_keypoints   # build context needs both scripts/ and ros2_ws/
docker build -f ros2_ws/Dockerfile -t hand_keypoint_ros .
```

### Native Windows/WSL2 manual (Ubuntu 24.04)

Use this procedure on the local Windows PC after the repository, ROS2
Jazzy, virtual environment, dependencies, and workspace have been built.
Every new WSL terminal must source ROS, the Python virtual environment,
and the built workspace overlay.

#### Terminal 1: start the pipeline

Open PowerShell or Windows Terminal:

```powershell
wsl -d Ubuntu
```

Then, inside Ubuntu:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
source ~/hand_keypoints_ros/ros2_ws/install/setup.bash

cd ~/hand_keypoints_ros/ros2_ws
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py \
  preload_depth:=false \
  depth_source:=real \
  loop:=true
```

Expected confirmation lines include:

```text
streaming real depth frame-by-frame ... background chunk prefetch ... no full preload
renderer: D3D12 (NVIDIA GeForce RTX 3060)
Created TensorFlow Lite delegate for GPU
depth source: REAL DEPTH
```

`fake_camera_publisher` is Node 1 for this test: it publishes recorded
RGB, depth, and calibration as camera topics. `hand_detection_node` is
Node 2: it subscribes to those topics, runs MediaPipe/depth processing,
and publishes `/surgery/perception/cam4/hand_keypoints`.

#### Terminal 2: inspect topics and measure FPS

Leave Terminal 1 running. Open another PowerShell or Windows Terminal:

```powershell
wsl -d Ubuntu
```

Then source the environment in this new Ubuntu terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
source ~/hand_keypoints_ros/ros2_ws/install/setup.bash
```

List the live ROS topics:

```bash
ros2 topic list
```

Measure Node 1's frame timer using the lightweight calibration topic:

```bash
ros2 topic hz /camera/color/camera_info --window 50
```

This is the preferred source-rate check because `CameraInfo` is published
once per RGB/depth frame but is much smaller than a 1280x720 image. It
should be close to 15 Hz.

The raw RGB rate can also be checked:

```bash
ros2 topic hz /camera/color/image_raw --window 50
```

However, the Python CLI must deserialize another complete raw image
stream. On this local test it added enough load to perturb the pipeline,
so do not leave this command running while measuring Node 2.

Press `Ctrl+C` to stop the RGB measurement, then measure the complete
Node 2 result rate by itself:

```bash
ros2 topic hz /surgery/perception/cam4/hand_keypoints --window 50
```

Or read the detector's internal rate without adding an image subscriber:

```bash
ros2 topic echo /surgery/perception/handkeypoint/diagnostics/json --once
```

Look for `processed_hz_1s` in the JSON.

Interpretation:

- Source near 15 Hz and `/surgery/perception/cam4/hand_keypoints` near
  15 Hz: end-to-end real-time.
- Source near 15 Hz but `/surgery/perception/cam4/hand_keypoints` below
  15 Hz: Node 2 or its ROS
  input/output path is the bottleneck.
- Source below 15 Hz: fake-camera replay/publishing is the bottleneck.

Display one typed result message:

```bash
ros2 topic echo /surgery/perception/cam4/hand_keypoints --once
```

Stop the measurement with `Ctrl+C`, then return to Terminal 1 and press
`Ctrl+C` to stop both launched nodes. For a single playback, use
`loop:=false`; publication stops at EOF, but the current launch still
requires Ctrl+C to terminate its processes.

#### Rebuild after editing Python or launch code

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
cd ~/hand_keypoints_ros/ros2_ws
python -m colcon build --symlink-install --packages-select hand_keypoint_ros
source install/setup.bash
```

Use `python -m colcon`, not plain `colcon`, so generated node scripts use
the virtual-environment interpreter containing MediaPipe and h5py.

### Try it without any real camera hardware

A `fake_camera_publisher` node replays this repo's bundled example clip
(`data/`) as ROS topics with the exact names `hand_detection_node`
expects out of the box, so the whole thing can be verified end-to-end
first:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/data:/repo/data:ro" \
  hand_keypoint_ros bash -c "
    source /opt/ros/jazzy/setup.bash && source /ros_ws/install/setup.bash
    ros2 launch hand_keypoint_ros fake_camera_demo.launch.py loop:=false
  "
```

No GPU on this machine: add `-e CUDA_VISIBLE_DEVICES=` to the `docker
run` line, or launch with `cpu_only:=true` (see below) to also skip
attempting MediaPipe's GPU delegate.

`loop:=false` is recommended for a first test (see the looping caveat
below) — the publisher runs through the clip once and stops instead of
looping indefinitely.

In another terminal (`docker exec -it <container> bash`, then `source
/opt/ros/jazzy/setup.bash && source /ros_ws/install/setup.bash` — **both**
are needed, the custom message type lives in the workspace overlay, not
base ROS):

```bash
ros2 topic hz /surgery/perception/cam4/hand_keypoints          # confirm it is publishing
ros2 topic echo /surgery/perception/cam4/hand_keypoints --once # see one typed frame
ros2 interface show hand_keypoint_interfaces/msg/HandKeypoints   # full schema
```

Launch file arguments (all optional, `name:=value`):

| Argument | Default | Meaning |
| --- | --- | --- |
| `rgb_path`, `calib_path`, `depth_h5_path` | this repo's example clip | swap in a different clip |
| `depth_source` | `auto` | `auto` \| `real` \| `mono` — forces the detection node's depth source. Use `mono` to exercise the RGB-only fallback even when real depth is available (ROS2 launch can't take an empty-string arg, hence a dedicated flag instead of clearing `depth_h5_path`) |
| `robot_position` | `` (disabled) | `top-left` \| `top-right` \| `bottom-left` \| `bottom-right` — see the robot handoff caveat above; same corner-mapping logic, same "only top-left validated" caveat |
| `cpu_only` | `false` | forces CPU for MediaPipe + any mono-depth model |
| `rate_hz` | `15.0` | fake_camera_publisher's playback rate |
| `loop` | `true` | loop the clip at end-of-stream. `loop:=false` gives a quick, deterministic one-shot run instead — useful for a fast smoke test |
| `preload_depth` | `true` | `true`: load the complete HDF5 depth array into RAM for maximum replay speed. `false`: publish frame by frame using a bounded, background chunk-prefetch buffer (about 118 MB for the bundled file) |

Low-memory, frame-by-frame replay on native WSL:

```bash
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py \
  preload_depth:=false \
  loop:=false
```

`loop:=false` stops publication at end-of-clip, but currently does not
terminate the two launched ROS processes. The terminal remains active
until `Ctrl+C`; automatic launch shutdown at EOF is still open work.

### Run against a real camera

`hand_detection_node` subscribes to standard RealSense-style topics by
default — remap if your driver uses different names:

```bash
ros2 run hand_keypoint_ros hand_detection_node --ros-args \
  -r /camera/color/image_raw:=/your/color/topic \
  -r /camera/aligned_depth_to_color/image_raw:=/your/depth/topic \
  -r /camera/color/camera_info:=/your/camera_info/topic \
  -p depth_source:=auto \
  -p robot_position:=top-left
```

`depth_source:=auto` polls the ROS graph for ~3s at startup for a
publisher on the depth topic; if the camera driver hasn't come up yet,
it falls back to mono depth and logs a warning — restart the node (or
pass `depth_source:=real` explicitly) once the driver is confirmed up.

### Topics

| Topic | Type | Direction | Notes |
| --- | --- | --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image` | sub | bgr8 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | sub | 16UC1, millimetres, RealSense-aligned-depth convention. Only subscribed when `depth_source` resolves to `real` |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | sub | `k[0,4,2,5]` = fx,fy,cx,cy |
| `hand/keypoints` | `hand_keypoint_interfaces/HandKeypoints` | pub | one frame's hands, typed — `header` (stamp + frame_id), `depth_source`, `hands[]` (each with `joints_2d`, `joints_3d`, `handedness`, `palm_6d`). See `hand_keypoint_interfaces/msg/*.msg` for the full field-by-field schema, or `ros2 interface show hand_keypoint_interfaces/msg/HandKeypoints` |
| `hand/overlay_image` | `sensor_msgs/Image` | pub | annotated debug view; disable with `publish_overlay:=false` |
| `hand/target_pose` | `geometry_msgs/PoseStamped` | pub | **only when `robot_position` is set** — the selected hand's palm 6D, ready for a downstream TF/robot node |

### Parameters (`hand_detection_node`)

`color_topic`, `depth_topic`, `camera_info_topic`, `depth_source`
(`auto`\|`real`\|`mono`), `depth_model`, `max_hands`, `cpu_only`,
`flip_handedness`, `region_x_min/max`, `region_y_min/max`,
`robot_position`, `publish_overlay`, `sync_slop_sec` — same meanings as
the equivalent `run_hand_keypoints.py` CLI flags (see "Usage" above).

### Test-fixture fixes and replay modes

While capturing a demo recording (below) to confirm the pipeline
actually works, not just that it compiles, two real issues turned up in
`fake_camera_publisher`:

1. **Naive per-frame HDF5 reads.** The HDF5 dataset is gzip-compressed
   in `(32, 45, 80)` chunks. Indexing one frame at a time repeatedly
   decompresses data belonging to the same 32-frame chunk and previously
   made replay fall badly behind its configured rate. Full preload remains
   the default (`preload_depth:=true`) for maximum replay speed. A new
   low-memory mode (`preload_depth:=false`) reads chunk-aligned 32-frame
   blocks and prefetches the next block on a background worker while still
   publishing exactly one RGB/depth/calibration set per timer tick. It uses
   about 118 MB for two 59 MB blocks instead of materializing the complete
   4.54 GB depth array.
2. **A depth-source race this preload fix exposed.** Publishers were
   created *after* the (now up-front) depth preload, so the depth topic
   wasn't advertised in time for `hand_detection_node`'s ~3s
   `depth_source=auto` check — it consistently (mis)concluded no real
   depth was available and silently fell back to monocular depth every
   run. **Fixed:** create publishers immediately, before the preload;
   `__init__` still can't return (so the timer can't fire) until
   preloading finishes, so this only fixes discovery timing, not data
   flow.

The HDF5 file itself occupies about 574 MB on disk; full decompression is
about 4.54 GB. On the local WSL machine, three isolated aligned 32-frame
reads took 0.123-0.130 seconds each (about 247-261 depth frames/s). This
shows that chunk-aligned access is fast enough in isolation, but it does
not by itself prove end-to-end ROS throughput; see the local measurement
below.

`hand_detection_node` still runs the same `process_frame()` call as the
CLI pipeline. End-to-end rate also includes video decoding, ROS image
conversion/serialization, synchronization, and subscribers.

### Native WSL code-change history (2026-08-11)

- Added `core.py` to the native ROS package path; the Docker build had
  previously supplied this copy only while building the image.
- Made demo data-path discovery portable: use the checkout's `data/`
  directory under native WSL and retain `/repo/data` as the Docker fallback.
- Made Ctrl+C teardown idempotent by handling
  `ExternalShutdownException`, using `rclpy.try_shutdown()`, and guarding
  node destruction.
- Suppressed MediaPipe GPU delegate's repetitive native `tensor.cc:410`
  output only around the detection call. Set
  `HAND_KEYPOINTS_MEDIAPIPE_LOGS=1` to restore it for debugging.
- Added the `preload_depth` launch/node parameter and the bounded,
  chunk-aligned background prefetch implementation described above.
- Native builds must be run from the project virtual environment with
  `python -m colcon build --symlink-install`; plain `colcon build` can
  regenerate console scripts with `/usr/bin/python3`, which cannot see
  the venv-only MediaPipe/h5py installations.

### Proof it actually works: a captured recording

`results/ros2_demo_capture/overlay_capture.gif` — real output, not a
mockup: `fake_camera_publisher` replaying the bundled example clip with
real depth, `hand_detection_node` correctly reporting `REAL DEPTH`,
tracking the hand holding the scalpel with correct handedness
(`Right 0.99`). Captured by subscribing to `/hand/overlay_image` for
~35s and writing it to video — see `results/ros2_demo_capture/` (the
capture script itself isn't part of the shipped package, it was a
one-off verification tool).

### Measured ROS2 pipeline rate

Two different measurements — don't confuse them:

**At the source clips' actual rate** (`rate_hz=15.0`, the default —
this is what a real camera driver replaying/producing footage at 15 fps
would look like), measured via `ros2 topic hz`:

| Topic | Rate |
| --- | ---: |
| `/camera/color/image_raw` (source, fake_camera_publisher) | ~15.0 Hz |
| `hand/keypoints` (full pipeline output, hand_detection_node) | ~14.9-15.0 Hz |

`hand_detection_node` keeps up with the source exactly — but this
number alone doesn't tell you the node's real ceiling, only that 15 Hz
isn't enough to find it.

**Real ceiling**, found by uncapping the source (`rate_hz:=100.0`, so
`fake_camera_publisher` publishes as fast as it can — ~80 Hz in
practice — and `hand_detection_node` is measured at whatever rate it
settles to on its own):

| Compute | `hand/keypoints` ceiling | Offline CLI equivalent (from the table above) |
| --- | ---: | ---: |
| GPU (RTX PRO 6000) | **~44 Hz** | ~42-44 fps |
| CPU only (`cpu_only:=true`, same EPYC 9554) | **~17 Hz** | 15.73 fps |

Both essentially match the offline CLI benchmarks — ROS/DDS message
passing adds negligible overhead here; the GPU/CPU compute itself is
still the real bottleneck in both cases, same as the standalone script.

#### Local WSL2 low-memory replay measurement (RTX 3060, 2026-08-11)

Configuration:

- Windows 11 + WSL2 Ubuntu 24.04 + ROS2 Jazzy
- Intel Core i5-14400F and NVIDIA GeForce RTX 3060 12 GB
- Real HDF5 depth, `preload_depth:=false`, `rate_hz:=15.0`
- MediaPipe GPU delegate confirmed through EGL/D3D12 on the RTX 3060

Observed during the first instrumented run (retained here as a measurement
pitfall, not as a valid pipeline benchmark):

| Measurement | Observed rate/result |
| --- | ---: |
| HDF5 aligned 32-frame block read, isolated | ~247-261 frames/s |
| `/camera/color/camera_info` (same timer as RGB/depth) | normally ~14.96-15.00 Hz |
| `/camera/color/image_raw` via Python `ros2 topic hz` | cumulative average rose to 10.18 Hz |
| old `/hand/keypoints`, 50-message moving window | ~7.47-8.60 Hz; latest 8.46 Hz |
| Benchmark validity | **invalid for judging inference speed** (the monitor perturbed the source) |

One 2.117-second source-timer interruption appeared while the large raw
RGB topic was simultaneously being monitored. At the time, the fake camera
also used the default RELIABLE publisher QoS. The extra Python subscriber
had to deserialize 1280x720 images and could apply delivery backpressure,
slowing the source itself. A single 2.117-second stall inside a 50-message
window reduces an otherwise 15 Hz stream to roughly 9.1 Hz
(`49 / (49/15 + 2.117)`), which explains most of the reported 8-9 Hz.

The fake-camera publishers now use sensor-style BEST_EFFORT/VOLATILE QoS,
matching the detector subscriptions and preventing a slow monitor from
requiring reliable delivery. The 8.46 Hz value must not be reported as the
RTX 3060 inference rate or as proof that the pipeline is not real-time.

For a clean test, launch with real depth explicitly, do not monitor the raw
image at the same time, and measure only the current lightweight output:

```bash
ros2 topic hz /surgery/perception/cam4/hand_keypoints --window 50
```

The node also reports a non-invasive internal one-second processing rate as
`processed_hz_1s` on
`/surgery/perception/handkeypoint/diagnostics/json`.

The corrected rerun confirmed D3D12/RTX 3060 and the MediaPipe GPU delegate.
With a 15 Hz source, CameraInfo recovered to 15.00 Hz, while hand-keypoint
output measured about 11.35 Hz between pauses and about 9.22 Hz after one
2.388-second pause entered the 100-message window. Internal diagnostics at
that point reported 8.92 Hz, 52.1 ms for the last frame, and zero errors. An
aggressive `rate_hz:=100` test reached about 25.4-30.9 Hz output, showing that
GPU inference capacity exceeds 15 Hz. Full depth preload did not remove the
periodic pause, so the remaining bottleneck is in the fake-camera/ROS
synchronization path rather than HDF5 prefetch or GPU inference capacity.

A subsequent exact repeat used `preload_depth:=false`, `depth_source:=real`,
and `rate_hz:=15.0`, with no raw-image monitor. It confirmed a stable
14.999-15.001 Hz source and approximately **9.3-10.2 Hz** hand-keypoint output.
Internal diagnostics sampled 11.03 Hz, 49.3 ms for the last processed frame,
and zero errors. This is the current authoritative recorded-data result:
frame-by-frame publication without full HDF5 preload is **not yet real-time**
against the 15 FPS requirement (~0.62-0.68x). The bounded ~118 MB chunk buffer
is used only because the on-disk HDF5 dataset is gzip chunked; the complete
4.54 GB decompressed depth array is not loaded.

#### Local WSL2 live camera measurement (RTX 3060, 2026-08-11)

The detector was also tested without the fake camera or HDF5 replay. It
subscribed directly to the live ROS camera's frame-by-frame streams:

| Measurement | Result |
| --- | ---: |
| Live RGB | 848x480 `rgb8` |
| Live depth | 848x480 `16UC1` |
| Camera source rate | **~32 Hz** |
| `/surgery/perception/cam4/hand_keypoints` | **~27.4-28.4 Hz** |
| Relative to a 15 FPS requirement | **~1.86x real-time** |

The run confirmed the D3D12 RTX 3060 renderer, MediaPipe GPU delegate,
real-depth mode, and zero processing errors. This is a valid live throughput
measurement and shows that Node 2 is fast enough for a 15 FPS real-time input.

It is not yet a validation of 3D accuracy. The available depth topic was
`/camera/camera/depth/image_rect_raw` in `camera_depth_optical_frame`, while
RGB was in `camera_color_optical_frame`. The detector expects depth already
aligned to RGB and uses the color intrinsics for 3D lookup. Enable the camera
driver's aligned-depth output and use a topic such as
`/camera/camera/aligned_depth_to_color/image_raw` before treating live 3D hand
coordinates as geometrically valid.

Do not replace the earlier RTX PRO 6000/server figures with this result;
they describe different hardware and replay conditions.
