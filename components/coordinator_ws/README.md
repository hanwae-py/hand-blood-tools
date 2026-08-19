# Surgical Task Coordinator

Take-turn sequencer for the surgical robot. In normal coordinated operation it
decides which **one** of the three perception algorithms may run at a time and
drives the robot operations between them. A separate manual Tool + Hand
parallel stress test is also documented below; it is not the normal task flow.

| Algorithm | Owner | Local location and current status |
| --- | --- | --- |
| Surgical tool detection (RF-DETR) | 최비결 | `~/surgical_robot/rfdetr_perception_ros` — transferred model bundle plus locally integrated real ROS lifecycle node; tested |
| Hand keypoints + palm 6D (MediaPipe) | Han Nwae Nyein | `~/hand_keypoints_ros` — real ROS lifecycle node with RGB, real depth and CameraInfo; tested |
| Blood detection (RF-DETR with different weights/config) | third member | Real checkpoint/config not yet on this machine; coordinator currently uses a lifecycle stub |

## What this package is, and is not

It **is** the orchestration layer: it listens for a task command, activates
exactly one detector, waits for that detector's target result, sends the robot a
long-running action when the required target pose exists, and hands the turn to
the next algorithm. The current Tool wrapper publishes image-space semantics
but not a robot-ready `PoseStamped`, so the complete Tool -> robot grasp path is
not yet implemented.

It **is not** a detector and never becomes one. It imports none of the three
algorithms, opens no camera, and loads no model. Every interaction happens over
the ROS 2 lifecycle interface plus `geometry_msgs/PoseStamped` results, so each
team keeps its own repository, its own virtual environment and its own
conflicting dependency versions.

That separation is the point. Do not add detection code to this package.

## Why lifecycle nodes instead of an `enabled` flag

All three algorithms share one RTX 3060 (12 GB). A boolean "enabled" flag makes
a node ignore camera frames, but **its model stays resident in VRAM** — which
solves the latency problem and not the memory problem.

ROS 2 lifecycle gives two distinct "off" levels, and the coordinator chooses
between them with the `release_gpu_between_modes` parameter:

| State | Model in VRAM | Processing frames | Re-activation cost |
| --- | --- | --- | --- |
| `unconfigured` | no | no | full model load (seconds) |
| `inactive` | **yes** | no | instant |
| `active` | yes | yes | — |

- `release_gpu_between_modes:=true` — detectors are cleaned all the
  way to `unconfigured` between turns. Safe when the three models do not co-fit.
- `release_gpu_between_modes:=false` (default) — detectors only drop to `inactive`, so
  turns switch instantly. Use this **only** after confirming with
  `watch -n 1 nvidia-smi` that all three models fit at once.

## State machine

```text
IDLE
 │
 ├── REQUEST_TOOL
 │       ↓
 │   TOOL_DETECTION      tool detector ACTIVE, everyone else off
 │       ↓ tool pose received
 │   ROBOT_GRASP_TOOL    GraspTool action
 │       ↓ grasp succeeded
 │   HAND_DETECTION      hand detector ACTIVE, everyone else off
 │       ↓ hand pose received
 │   ROBOT_HANDOVER      HandoverTool action
 │       ↓ handover succeeded
 │   IDLE
 │
 ├── SUCK_BLOOD
 │       ↓
 │   BLOOD_DETECTION     blood detector ACTIVE, everyone else off
 │       ↓ blood pose received
 │   ROBOT_SUCTION       SuctionBlood action
 │       ↓ suction succeeded
 │   IDLE
 │
 └── ABORT ─────────────────────────────────────────────→ IDLE
```

Hand detection deliberately runs **after** the grasp, not before: where the
surgeon's hand was when the tool was requested is irrelevant — what matters is
where it is when the robot is actually ready to hand over.

Any failure enters `FAULT`, is published with a reason, then falls back to
`IDLE` so the next command is still accepted. On every exit path — success,
failure, crash or abort — all three detectors are deactivated.

## Interfaces

### Commands in — `/surgery/task/command` (`std_msgs/String`)

| Command | Effect |
| --- | --- |
| `REQUEST_TOOL` | tool → grasp → hand → handover, using `default_tool_name` |
| `REQUEST_TOOL:scalpel` | same, naming the tool |
| `SUCK_BLOOD` | blood → suction |
| `ABORT` | cancel the running flow and the in-flight robot goal, return to `IDLE` |

`std_msgs/String` is a deliberate placeholder: the voice team has not defined
their message yet, and a plain string is what a human can publish from a
terminal today. When their typed message exists, only `_on_command()` in
[`task_coordinator.py`](src/surgical_task_coordinator/surgical_task_coordinator/task_coordinator.py)
has to change.

### State out — `/surgery/task/state` (`surgical_task_interfaces/TaskState`)

Published on every transition plus a 1 Hz heartbeat. Carries the state name,
which detector is active (`''` when none), a human-readable detail string, and
when the state was entered.

### Robot actions out

| Action | Type |
| --- | --- |
| `/surgery/robot/grasp_tool` | `surgical_task_interfaces/action/GraspTool` |
| `/surgery/robot/handover_tool` | `surgical_task_interfaces/action/HandoverTool` |
| `/surgery/robot/suction_blood` | `surgical_task_interfaces/action/SuctionBlood` |

Actions rather than services or topics because these take seconds, can fail, and
must be cancellable when the surgeon aborts.

## What a detector must provide

This is the contract each of the three algorithms has to satisfy. It is
demonstrated end-to-end by
[`stub_detector.py`](src/surgical_task_coordinator/surgical_task_coordinator/stub_detector.py) —
read that file as the reference implementation.

1. Be a **ROS 2 lifecycle node** with a node name the coordinator is configured
   with (`tool_detection_node`, `hand_detection_node`, `blood_detection_node`).
2. Load the model in `on_configure()`, and **actually free it** in
   `on_cleanup()` — drop the references and call `torch.cuda.empty_cache()`.
   `nvidia-smi` is the acceptance test, not the absence of a Python reference.
3. Consume camera frames only while `ACTIVE`.
4. Publish `geometry_msgs/PoseStamped` on your result topic when you find your
   target, and keep publishing while you still see it.
5. Publish `std_msgs/String` health on its agreed health topic (currently `/surgery/perception/rfdetr/health` for Tool and `/surgery/perception/handkeypoint/health` for Hand).

Everything else is yours: which camera topic you read, what your model is, and
what richer messages you publish for other consumers.

`PoseStamped` is used uniformly so this package needs no build dependency on any
detector's private message package. Publish your rich typed result as well — the
coordinator simply does not need it.

### Default result topics

| Detector | Topic |
| --- | --- |
| tool | `/surgery/perception/cam4/tool_target_pose` |
| hand | `/surgery/perception/cam4/hand_target_pose` |
| blood | `/surgery/perception/cam4/blood_target_pose` |

The **hand** target topic is real and already published by `hand_keypoint_ros`.
The Blood target-pose name is still a coordinator proposal used by the stub. Surgical Tool Component v1.3 now publishes `/surgery/perception/cam4/tool_poses`, but these are per-instance camera-frame pose states rather than the old single `tool_target_pose` proposal. They are not robot-ready until the CAM4 support-plane calibration is supplied and each entry reports valid position/orientation. The real Blood contract is still unavailable.

### Verified detector output interfaces (2026-08-12)

The current real **Hand Detection** node publishes these five outputs:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/surgery/perception/cam4/hand_keypoints` | `hand_keypoint_interfaces/msg/HandKeypoints` | 2D keypoints, real-depth 3D keypoints, handedness, and palm pose |
| `/surgery/perception/cam4/hand_target_pose` | `geometry_msgs/msg/PoseStamped` | Robot-ready palm target when a valid palm pose exists |
| `/surgery/images/cam4/hand_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG visualization of detected hands |
| `/surgery/perception/handkeypoint/health` | `std_msgs/msg/String` | JSON lifecycle, readiness, and input-freshness status |
| `/surgery/perception/handkeypoint/diagnostics/json` | `std_msgs/msg/String` | JSON processing-rate, latency, frame, and error counters |

The current local real **Surgical Tool Component v1.3.0-rc1** adapter publishes these six outputs:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/surgery/perception/cam4/tool_poses` | `surgical_perception_msgs/msg/ToolPoseArray` | Canonical per-instance pose state and validity; currently INVALID until the calibrated CAM4 support plane is supplied |
| `/surgery/perception/cam4/observations` | `surgical_perception_msgs/msg/ToolObservation2DArray` | Canonical class, bbox, lossless COCO RLE mask, and observation-point evidence |
| `/surgery/images/cam4/detection_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | JPEG visualization of tool detections |
| `/surgery/perception/rfdetr/health` | `std_msgs/msg/String` | JSON model and input health status |
| `/surgery/perception/rfdetr/diagnostics/json` | `std_msgs/msg/String` | JSON performance and diagnostic information |
| `/surgery/perception/cam4/semantics/json` | `std_msgs/msg/String` | Temporary compatibility JSON; canonical consumers should use the typed observation and pose topics |

`/surgery/images/cam4/detected/compressed` and `/surgery/perception/cam4/mayo_tool_observations` remain removed/deprecated and are not published by v1.3.
### Visualize Tool and Hand overlay images

Keep the detector test running. Open a separate sourced WSL terminal for each viewer.

Tool-detection overlay:

```bash
source /opt/ros/jazzy/setup.bash
source ~/surgical_robot/coordinator_ws/install/setup.bash

ros2 run image_view image_view --ros-args \
  -p image:=/surgery/images/cam4/detection_overlay \
  -p image_transport:=compressed
```

Hand-detection overlay:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros/ros2_ws/install/setup.bash

ros2 run image_view image_view --ros-args \
  -p image:=/surgery/images/cam4/hand_overlay \
  -p image_transport:=compressed
```

Before opening a viewer, confirm that its compressed topic is publishing:

```bash
ros2 topic hz /surgery/images/cam4/detection_overlay/compressed
ros2 topic hz /surgery/images/cam4/hand_overlay/compressed
```

Press `Ctrl+C` after checking each rate. With `image_transport:=compressed`, pass the base topic to the `image` parameter; do not append `/compressed`. Use `-p image:=...`, not `-r image:=...`, with the installed Jazzy `image_view` executable. During signal-driven take-turn operation, only the currently active detector updates its overlay. During the parallel test, open both commands in separate terminals to watch Tool and Hand results at the same time.
## Relationship to the ARPA-H interface contract

The ARPA-H interface table is the **inter-organization** contract: what PNU
Computer Vision Lab publishes for PNU Visual Intelligence & Cognition Lab to
consume. Message types there live in the shared `surgical_msgs` package
(for example `surgical_msgs/msg/ToolObservation`).

Everything in `surgical_task_interfaces` is **PNU-internal orchestration** and
is deliberately not part of that table. Do not propose `TaskState` or the three
actions as inter-organization interfaces unless the other organization actually
needs to drive the robot.

The five current Hand output rows are documented locally in
`~/surgical_robot/docs/Hand_Pose.csv` (with the user's filled copy in
`HandPoseFilledByMe.csv`). They still need team confirmation/merge into the
shared master workbook. `hand_keypoint_interfaces/msg/HandKeypoints` is a
private package; any external subscriber must receive and build that package,
or the message must move into the team's shared interface package.

## Build

```bash
source /opt/ros/jazzy/setup.bash
cd ~/surgical_robot/coordinator_ws
colcon build --symlink-install
source install/setup.bash
```

This coordinator package has ROS dependencies and the local
`surgical_task_interfaces` package, but no MediaPipe, RF-DETR or Torch runtime
dependency. It therefore does **not** need either detector's ML virtual
environment; source Jazzy and build the complete coordinator workspace with
`colcon` as shown above.

## Run the stub demo

Brings up the coordinator, a fake robot, and three stub detectors. No real
model, no GPU, starts in seconds. Verifies sequencing, lifecycle switching and
the abort path.

```bash
ros2 launch surgical_task_coordinator coordinator_stub_demo.launch.py
```

In a second terminal:

```bash
# watch whose turn it is
ros2 topic echo /surgery/task/state

ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'REQUEST_TOOL:scalpel'}"
ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'SUCK_BLOOD'}"
ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'ABORT'}"

# confirm the invariant: at most one detector active, ever
ros2 lifecycle get /tool_detection_node
ros2 lifecycle get /hand_detection_node
ros2 lifecycle get /blood_detection_node
```

**Use `-w 1`.** Plain `ros2 topic pub --once` starts a brand-new publisher and
exits as soon as it has published, which is frequently *before* DDS discovery
has matched the coordinator's subscription — the message is then silently
dropped and nothing happens. This bit during testing: an `ABORT` vanished with
no log line at all. `-w 1` makes the publisher wait for one matched subscriber
first. (If nothing is subscribed, `-w 1` blocks forever — that is the tradeoff.)

Inject a robot failure to exercise the `FAULT` path:

```bash
ros2 param set /fake_robot_node fail_next true
ros2 topic pub --once -w 1 /surgery/task/command std_msgs/msg/String "{data: 'REQUEST_TOOL'}"
```

## Run with the real hand detector

Start the coordinator without its stub hand detector:

```bash
ros2 launch surgical_task_coordinator coordinator_stub_demo.launch.py use_stub_hand:=false
```

Then, in the hand repository's environment, start the real node with
`autostart:=false` so the **coordinator** owns the turn-taking rather than the
node activating itself:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
source ~/hand_keypoints_ros/ros2_ws/install/setup.bash
ros2 run hand_keypoint_ros hand_detection_node --ros-args -p autostart:=false
```

It also needs a camera. Use that repo's `fake_camera_demo.launch.py`, or run
`fake_camera_publisher` on its own.

## Integration status and remaining member checklist

Tool Detection is now copied locally, has its own virtual environment, runs as
a real lifecycle node, and has passed sequential and parallel ROS tests with
Hand Detection. Its remaining integration work is a robot-ready 3D target pose
and reconciliation of the two CSV-only Tool interfaces.

Blood Detection remains pending:

- [ ] Obtain the real RF-DETR checkpoint, ontology and configuration.
- [ ] Reuse/parameterize the RF-DETR lifecycle wrapper as
      `blood_detection_node` in its own compatible environment.
- [ ] Confirm `on_cleanup()` releases its GPU allocation with `nvidia-smi`.
- [ ] Agree and implement the real blood target/result interfaces.
- [ ] Replace the Blood stub and repeat sequential, signal-driven, and resource
      tests.

## RF-DETR sharing and three-algorithm test decision (2026-08-12)

Tool detection and blood detection use the same RF-DETR implementation with
different weights/configuration. Put one shared repository at
`~/surgical_robot/rfdetr_perception_ros/`, not inside the hand repository or
coordinator workspace. Initially copy/clone the complete server directory
`cam4_seg8_local_20260810` and verify `SHA256SUMS`. The ROS wrapper should
run `tool_detection_node` and `blood_detection_node` as two parameterized
lifecycle-node instances from the same code.

Incremental progress now completed: the all-stub coordinator flow, real Hand,
real Tool -> Hand sequencing, signal-driven switching, and optional parallel
Tool + Hand execution have been tested. The remaining detector replacement is
real Blood when its RF-DETR weights/config arrive. `SUCK_BLOOD` remains
testable with the lifecycle stub.

Tool and Hand use the same camera but normally run sequentially: Tool Detection
before grasp, Hand Detection after grasp. Their optional parallel stress test
confirmed that both models coexist on the RTX 3060 without out-of-memory at an
observed total allocation of 4362 MiB. Parallel publication was below the 15 FPS
source rate, however, and the real Blood model has not been added. The available Tool + Hand + Blood-stub workflow now uses `release_gpu_between_modes:=false`; repeat the VRAM check after the real Blood checkpoint is integrated.

**Parallel measured result: Tool output 11.583 Hz; Hand output 9.923 Hz (15 FPS source, functional pass but not real-time).**

### Local RF-DETR transfer status (2026-08-12)

Copied from
`C:\Users\user\Documents\arpa-h\August\tools detection\cam4_seg8_local_20260810`
to `~/surgical_robot/rfdetr_perception_ros/`. All five entries in
`SHA256SUMS` passed: checkpoint, inference script, environment config,
ontology and validation image. The RF-DETR environment must remain separate
from the hand-detection venv because their pinned Torch/dependency versions
differ.

### Transferred RF-DETR bundle and local ROS integration (2026-08-12)

The original `rfdetr_perception_ros` transfer contained the model, configs,
documentation, one validation image and `scripts/standalone_inference.py`, but
did not include the teammate's separate ROS bridge or image publisher. A local
real lifecycle wrapper, `scripts/tool_detection_ros_node.py`, has since been
implemented and tested. It subscribes to the same frame-by-frame fake-camera
RGB stream as Hand Detection and publishes real RF-DETR semantics, overlay,
health and diagnostics topics. Sequential Tool -> Hand operation and parallel
Tool + Hand operation have both passed functional downstream-reception tests.

### Surgical Tool Component v1.3.0-rc1 migration (2026-08-12)

The checksum-verified delivery is preserved separately at `~/surgical_robot/tool_detection_component_v1_3_rc1`; the old `rfdetr_perception_ros` folder remains unchanged for rollback. The v1.3 Python algorithm was installed into the existing Tool virtual environment, and its `surgical_perception_msgs` v0.2.0 package was copied into `coordinator_ws/src/` and built successfully.

The host lifecycle adapter is `scripts/tool_detection_v13_ros_node.py`. It consumes the shared frame-by-frame RGB, aligned depth, and CameraInfo topics, publishes canonical typed `ToolObservation2DArray` and `ToolPoseArray` messages, retains the overlay/health/diagnostics outputs, and provides the old semantics JSON only as a temporary compatibility topic. Both fixed-order and signal-driven launchers now use v1.3; the old wrapper remains available only for rollback.

Validation passed for the component static contract, pose contract, ROS mapping, Python syntax, shell syntax, ROS message build, and interface introspection. The final real sequential run received five typed observations and a five-entry Tool pose array (`observation_id=cam4:0`), then switched to Hand and received one depth-backed hand with 10 valid 3D keypoints and one palm pose. The downstream receiver reported success.

Important limitation: the delivery contains `null` CAM4 support-plane template values. Therefore the adapter correctly publishes `valid_poses=0`, `POSE_MODE_INVALID`, and `SUPPORT_PLANE_NOT_CONFIGURED` rather than inventing 3D Tool poses. Detection, masks, typed transport, overlay, lifecycle switching, and downstream reception are verified; metric Tool pose is pending Bigyeol's calibrated support-plane normal, offset, and configuration version.
### RF-DETR benchmark status (2026-08-12)

The transferred bundle itself contained no valid streaming benchmark. Its
metadata's unoptimized, non-warmed-up 409.37 ms single-image latency is marked
`latency_is_benchmark_eligible: false` and must not be reported as pipeline FPS.
A local warmed-up video benchmark, `scripts/benchmark_video.py`, was subsequently
added and run on the RTX 3060. Its verified 300-frame result is documented below:
40.93 model-only FPS and 36.06 decode-plus-inference FPS. That standalone result
excludes ROS transport, overlay rendering and downstream processing.

### RF-DETR local single-image sanity test (2026-08-12)

The transferred surgical-tool RF-DETR bundle was verified locally in WSL using its own Python 3.12 virtual environment and the supplied validation image.

- Checkpoint: `rfdetr_perception_ros/models/checkpoint_best_total.pth`
- Input: `test_data/cam4_validation_sample.jpg` (1280 x 720)
- Confidence threshold: `0.5`
- Result: **9 detected instances** (matches the expected sanity result)
- Saved result: `rfdetr_perception_ros/results/local_single_frame_sanity.json`
- Reported latency: **491.93 ms**, with `--no-optimize`

This confirms that the transferred model, dependencies, checkpoint, and inference script work on the local WSL machine. The 491.93 ms value is a cold, unoptimized, single-image measurement; it is **not** a valid streaming FPS benchmark. Run the optimized check next, followed by a warmed-up repeated-frame/video benchmark.

### RF-DETR video FPS benchmark

`rfdetr_perception_ros/scripts/benchmark_video.py` reads each AVI frame on demand; it does not preload the video. It performs warm-up frames before timing and writes a JSON result containing model-only and complete pipeline rates.

From `~/surgical_robot/rfdetr_perception_ros`, activate its own environment:

```bash
source .venv/bin/activate
```

Quick 300-frame test on the 30-second clip:

```bash
python scripts/benchmark_video.py \
  --task cam4_seg8 \
  --checkpoint models/checkpoint_best_total.pth \
  --video test_data/videos/0618_2_cam4_rgb_11m11s_to_11m41s.avi \
  --output results/benchmark_0618_300frames.json \
  --threshold 0.5 \
  --warmup-frames 10 \
  --max-frames 300
```

For a full-video test, use the same command with `--max-frames 0`. Repeat with `0704_pnu_4_cam_4_rgb_all_tools.avi` and a different output filename.

Interpretation:

- `inference_fps`: RF-DETR model prediction only.
- `end_to_end_fps`: sequential video decoding plus RF-DETR inference; use this for the local frame-by-frame pipeline result.
- `source.fps`: FPS stored in the source video.
- `realtime_ratio`: `end_to_end_fps / source.fps`.
- `keeps_up_with_source`: true only when `end_to_end_fps >= source.fps`.

This benchmark excludes ROS transmission, overlay drawing, and downstream robot processing.

#### Measured local result: 0618 clip, 300 frames (2026-08-12)

| Device | Source | Measured frames | Inference FPS | End-to-end FPS | Real-time ratio | Keeps up |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVIDIA GeForce RTX 3060 | 15.0 FPS, 1280x720 | 300 (+10 warm-up) | 40.93 | 36.06 | 2.40x | Yes |

The optimized benchmark read the AVI frame-by-frame without preloading. `end_to_end_fps` includes video decoding and RF-DETR inference, but excludes ROS transport, overlay rendering, and downstream processing. The result is saved in `rfdetr_perception_ros/results/benchmark_0618_300frames.json`.

### Hand pipeline real-time status and remaining fix

The local WSL2 live-camera test already demonstrated sufficient hand-detection throughput on the RTX 3060:

| Measurement | Result |
| --- | ---: |
| Live camera source | ~32 Hz |
| Hand-keypoint output | **~27.4-28.4 Hz** |
| Relative to the 15 FPS requirement | **~1.86x real-time** |

Therefore, hand detection does not currently require an inference-speed fix. The remaining production requirement is **geometric correctness**: the depth stream must be aligned to the RGB/color frame, use color-camera intrinsics, and carry synchronized timestamps. The available live test used separate color and depth optical frames, so it validated throughput but not the accuracy of the resulting 3D hand coordinates.

The lower ~9.3-10.2 Hz result belongs to the gzip-HDF5 fake-camera replay and its synchronization/recorded-data pauses; it must not be reported as the live detector's throughput. Disable `publish_overlay` in production unless the debug image is needed, to reduce CPU work and ROS bandwidth.

### Real tool + real hand take-turn test on the matched 0618 recording

Run:

```bash
cd ~/surgical_robot/coordinator_ws
bash scripts/run_real_tool_hand_take_turn_test.sh
```

The script uses the matched files under `~/surgical_robot/test_data/cam4_0618`: RGB AVI, raw depth HDF5, and calibration JSON. It publishes RGB, real depth, and CameraInfo at 15 Hz without loading the complete HDF5 into RAM. It then:

1. configures and activates the real v1.3 Tool node;
2. confirms a downstream subscriber receives both `/surgery/perception/cam4/observations` and `/surgery/perception/cam4/tool_poses`;
3. deactivates and cleans the Tool node to release GPU memory;
4. configures and activates the real Hand node with `depth_source:=real`;
5. confirms a downstream subscriber receives `/surgery/perception/cam4/hand_keypoints` with valid depth-backed 3D data and a palm pose;
6. reports success only after one receiver has received all three typed outputs.

Expected receiver lines are `RECEIVED TOOL POSE ARRAY v1.3`, `RECEIVED TOOL RESULT v1.3`, `RECEIVED HAND RESULT`, and `SUCCESS: downstream node received both real detector outputs`. Tool detection/masks are real; Tool pose entries remain explicitly invalid until the missing calibrated support-plane parameters are supplied. Hand output includes HDF5-depth-based 3D keypoints and palm pose.

### Signal-driven perception switching test

The fixed-order script proves tool-to-hand sequencing, but the production-like signal behavior is tested separately:

```bash
cd ~/surgical_robot/coordinator_ws
bash scripts/run_signal_driven_perception_test.sh
```

Keep that terminal running. In a second WSL terminal, source ROS and the coordinator workspace, then send one command at a time:

```bash
source /opt/ros/jazzy/setup.bash
source ~/surgical_robot/coordinator_ws/install/setup.bash

ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String "{data: 'DETECT_TOOL'}"
ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String "{data: 'DETECT_HAND'}"
ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String "{data: 'DETECT_BLOOD'}"
ros2 topic pub --once -w 1 /surgery/perception/mode_command std_msgs/msg/String "{data: 'STOP'}"
```

Wait for the previous command's ACTIVE/result line before sending the next. At startup, the coordinator configures Tool, Hand, and Blood sequentially and leaves them INACTIVE with their models resident. Every new mode signal then deactivates the current detector without cleanup and activates the already-loaded requested detector. Tool and hand are real models using the matched 0618 recording; blood is a lifecycle stub until the real blood checkpoint/config arrive. State is published as JSON on `/surgery/perception/mode_state`.

The downstream receiver now accepts tool success only when at least one instance exists. Hand success requires at least one hand, at least one valid depth-backed 3D keypoint, and a valid palm pose; a zero-hand message no longer produces a false success.

#### Measured signal-driven result (2026-08-12)

The available full sequence passed using the 15 FPS frame-by-frame fake-camera replay with both overlays enabled: real RF-DETR Tool, real MediaPipe/real-depth Hand, and the Blood lifecycle stub.

| Phase | Command to ACTIVE | Command to first result | Output rate |
| --- | ---: | ---: | ---: |
| `DETECT_TOOL` | **6.93 s** | **10.52 s** | **7.92 Hz** |
| Tool -> `DETECT_HAND` | **2.83 s** | **17.14 s** first message; **19.23 s** first nonzero hand | **8.33 Hz** |
| Hand -> `DETECT_BLOOD` stub | **2.17 s** | **3.65 s** | Not meaningful for a stub |

Tool became inactive within 0.38 s and completed model cleanup within 0.44 s after the Hand command. Hand became inactive within 0.04 s and completed model cleanup within 0.12 s after the Blood command. The long Hand first-result delay occurred after Hand was already ACTIVE, in the recorded RGB/HDF5 replay/synchronization path; it was not model-loading time.

This verifies command delivery, lifecycle transitions, GPU cleanup, activation, and result publication for every detector currently available. Tool and Hand were below the 15 FPS source rate in this overlay-enabled run. A complete three-real-detector test remains pending until the real Blood checkpoint and configuration are available.


#### Startup-preloaded switching result (2026-08-12)

The signal-driven launcher now starts the coordinator with `preload_models_on_startup:=true` and `release_gpu_between_modes:=false`. Tool, Hand, and the Blood stub are configured sequentially once at startup and remain `INACTIVE` with their models resident until commanded.

Cold preload took **14.77 s** in the measured full run; a repeat preload-only run took **11.60 s**. This cost is paid once before the system reports `all detector models preloaded` and accepts the timed command sequence.

| Phase | Command to ACTIVE | Command to first result |
| --- | ---: | ---: |
| `DETECT_TOOL` | **0.10 s** | **3.74 s** |
| Tool -> `DETECT_HAND` | **0.12 s** | **14.73 s** first message; **15.76 s** first nonzero hand |
| Hand -> `DETECT_BLOOD` stub | **0.12 s** | **1.60 s** |

Tool became inactive **0.02 s** after the Hand command, and Hand became inactive **0.04 s** after the Blood command. Neither detector was cleaned up during switching; both logs explicitly reported that the model remained loaded. Compared with the earlier cleanup/reload run, command-to-ACTIVE improved from 6.93 s to 0.10 s for Tool, 2.83 s to 0.12 s for Hand, and 2.17 s to 0.12 s for the Blood stub.

A preload-only snapshot showed **3016/12288 MiB** total RTX 3060 memory usage and no out-of-memory error. This is whole-GPU usage, not model-exclusive usage. The real Blood model is still unavailable, so VRAM must be measured again after it replaces the stub.

Preloading solved model-switch activation delay. It did **not** solve Hand's first-result delay: Hand was ACTIVE after 0.12 s but waited another ~14.6 s in the recorded RGB/HDF5 synchronization path before publishing. Continuous Tool/Hand output also remains below the 15 FPS source rate; those are separate synchronization and per-frame performance tasks.

### Parallel Tool + Hand stress test (2026-08-12)

Parallel test results:

- Tool output: **11.583 Hz**
- Hand output: **9.923 Hz**
- Source: **15 FPS**
- GPU memory: **4362/12288 MiB**
- GPU utilization observed: **1–35%**
- No GPU out-of-memory error
- Conclusion: both algorithms work concurrently, but this run was not 15 FPS real-time.

The real Tool and Hand lifecycle nodes were manually configured and activated
at the same time while consuming the same fake-camera RGB frames. Hand also
consumed matching frame-by-frame HDF5 depth and CameraInfo. A downstream
receiver obtained both real outputs, including a Hand message with one valid
palm pose; no GPU out-of-memory error occurred.

**Parallel result: Tool 11.583 Hz + Hand 9.923 Hz on the same RTX 3060.**

| Measurement | Observed result |
| --- | ---: |
| Tool semantics output | **11.583 Hz** |
| Hand keypoints output | **9.923 Hz** |
| Source rate | **15 FPS** |
| RTX 3060 memory | **4362 / 12288 MiB** |
| Observed GPU utilization | **1-35%** |

This is a functional parallel pass but **not** a 15 FPS real-time pass. These
values are ROS result-topic publication rates measured with `ros2 topic hz`.
The exact commands and raw test interpretation are recorded in
`~/surgical_robot/test_history.md`.


#### PNU-4 v1.3 ROS performance and mask-encoding fix (2026-08-12)

The first PNU-4 v1.3 run published only about 0.125 Hz. Live diagnostics separated `last_inference_ms=193.21` from `last_total_ms=7854.82`, proving that GPU inference was not the eight-second bottleneck. The delivered reference ROS mapper encoded every 1280x720 instance mask with a Python pixel loop; with roughly 9-12 tools, COCO-RLE conversion dominated the complete frame time.

The host adapter now replaces only that encoder with the equivalent compiled `pycocotools` COCO-RLE implementation while preserving the v1.3 message format. Validation still passes. On the same PNU-4 15 FPS frame-by-frame RGB/HDF5 replay, `/surgery/perception/cam4/observations` improved to approximately **8.03 Hz**. This is about a 64x improvement over 0.125 Hz, but it is still below the 15 FPS source rate and is not yet a real-time pass.

A live overlay message was independently decoded successfully (`jpeg`, 1280x720, 167137 bytes, non-black pixel distribution). A black `image_view` window was therefore a viewer discovery/environment issue, not a bad overlay publisher.

The complete post-v1.3 integration record—including adapter/interface changes, sequential and parallel tests, signal-driven switching, startup preloading, performance measurements, reproduction commands, and remaining limitations—is in `~/surgical_robot/tool_v1_3_integration_test_results.md`.
## Operation manual

For the complete prepared-PC operating procedure, use `~/surgical_robot/OPERATION_MANUAL.md`. It covers builds, startup preloading, mode commands, output verification, FPS/GPU measurements, overlays, shutdown, and troubleshooting.

For a beginner explanation of the architecture, important runtime files, lifecycle flow, five Hand interfaces, and six Tool interfaces, read `~/surgical_robot/BEGINNER_SYSTEM_GUIDE.md`.

## Known gaps

- Tool and Hand use real detector nodes. Blood is still a lifecycle stub until
  its RF-DETR checkpoint/configuration is supplied.
- The local Tool node does not yet publish `tool_target_pose`,
  `/surgery/images/cam4/detected/compressed`, or
  `/surgery/perception/cam4/mayo_tool_observations`; those remain proposed or
  contract-only interfaces.
- The robot node is a stub that moves nowhere. The three action definitions are
  a proposal for the robot-control team, not something they have accepted.
- The voice team's command message does not exist; commands are typed strings.
- The shared frame-by-frame fake-camera RGB source has been confirmed for real
  Tool and Hand nodes; Hand additionally consumes real depth and CameraInfo.
  Real camera-driver integration and Blood input remain untested.
- No TF. Every pose is published in the camera optical frame; converting to the
  robot base frame is section 3-5 work and belongs to the robot team.
