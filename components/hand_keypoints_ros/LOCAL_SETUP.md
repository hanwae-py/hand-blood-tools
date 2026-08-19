# Local machine setup — read this first

This file exists so a fresh agent (or a person) picking this repo up on
a *different* machine — e.g. a Windows/WSL desktop, not the lab server
this was developed on — has one place to start instead of reconstructing
context from `git log`. Read this, then go to `README.md` for the full
pipeline/ROS2 reference.

## Where this stands right now

Everything below is done, tested, and pushed to
`https://github.com/hanwae-py/hand_keypoints_ros` (private repo):

- **Offline CLI pipeline** (`scripts/run_hand_keypoints.py`) — MediaPipe
  (GPU or CPU) + real depth (HDF5) or Depth-Anything V2 mono-depth
  fallback + palm-6D pose. Measured: ~43 fps GPU/real-depth, ~20 fps
  GPU/mono-depth, ~15.7-21.3 fps CPU-only (varies by CPU — see the
  README's performance table, includes a measurement on an
  i5-14400F).
- **ROS2 (Jazzy) integration** (`ros2_ws/`) — two packages, matching
  the team's planned structure:
  - `hand_keypoint_interfaces` — custom `.msg` types (`HandKeypoints`,
    `Hand`, `PalmPose6D`, `Point2D`). No code, no ML deps.
  - `hand_keypoint_ros` — `hand_detection_node` (the actual detector,
    subscribes to camera topics, publishes typed messages) and
    `fake_camera_publisher` (replays the bundled example clip as ROS
    topics, for testing without real camera hardware).
- **Verified end-to-end**, not just "it compiles": captured real
  output (`results/ros2_demo_capture/overlay_capture.gif`), measured
  real throughput ceilings (~44 Hz GPU, ~17 Hz CPU-only — matching the
  offline CLI numbers, confirming ROS overhead is negligible).
- Two real bugs were found and fixed while verifying this (see
  `README.md`'s "Test fixture: two bugs found and fixed" section) —
  worth reading if `fake_camera_publisher` ever looks like it's
  stalled or silently using the wrong depth source again.

**Not yet done / open questions:**
- Never tested against a *real* camera driver, only the
  `fake_camera_publisher` replay fixture.
- WSL2 GPU passthrough for MediaPipe's GPU delegate has not been
  verified on Windows — the CPU-only numbers on the i5-14400F have,
  the GPU path on that machine has not.
- The team hasn't yet decided whether to keep `hand/keypoints`
  carrying both 2D and 3D together (current implementation) vs.
  splitting it the way an earlier planning diagram implied (separate
  2D-detection and 2D→3D-conversion sections/nodes). Current code
  assumes "keep it combined" — flag this if the team decides
  otherwise.

## Getting the repo (Windows / WSL2)

This repo uses **Git LFS** for the example data (`data/rgb/`,
`data/depth/`, `data/depth_raw/` — ~1.25 GB total). Install LFS
*before* cloning, or you'll get tiny pointer files instead of real
video/HDF5 data:

```powershell
git lfs install
git clone https://github.com/hanwae-py/hand_keypoints_ros.git
```

If you already cloned without LFS installed, fix it in place:

```powershell
git lfs install
git lfs pull
```

## Building and testing the ROS2 side locally

ROS2 Jazzy targets Ubuntu 24.04. If your WSL2 distro isn't 24.04,
Docker is the path of least resistance (same approach used on the lab
server) — see `README.md`'s "ROS2 (Jazzy) integration" → "Build"
section for the exact commands. Short version, from a shell with
Docker available (Docker Desktop + WSL2 backend, or a native WSL2
Ubuntu 24.04 with Docker installed):

```bash
cd hand_keypoints
docker build -f ros2_ws/Dockerfile -t hand_keypoint_ros .

docker run --rm -it --gpus all \
  -v "$(pwd)/data:/repo/data:ro" \
  hand_keypoint_ros bash -c "
    source /opt/ros/jazzy/setup.bash && source /ros_ws/install/setup.bash
    ros2 launch hand_keypoint_ros fake_camera_demo.launch.py loop:=false
  "
```

Drop `--gpus all` (and the GPU delegate will fall back to CPU
automatically) if this machine has no GPU, or pass `cpu_only:=true` on
the launch line to force it explicitly. In another terminal:

```bash
ros2 topic echo /hand/keypoints --once
```

If that gives you a typed message with real numbers in `joints_3d` /
`palm_6d`, the local setup is confirmed working end-to-end.

## Full reference

Everything else — CLI flags, calibration JSON format, output schema,
ROS topics/parameters table, the robot-handoff mode and its
corner-mapping caveat, performance tables, known caveats — is in
`README.md`. This file is only the "orient yourself first" entry
point.
