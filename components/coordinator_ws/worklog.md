# Surgical Task Coordinator Work Log

Date: 2026-08-11
Local machine: Intel Core i5-14400F, NVIDIA GeForce RTX 3060 12 GB
Environment: Windows 11 + WSL2 Ubuntu 24.04 + ROS 2 Jazzy

## Objective

Build the take-turn sequencer for the surgical robot. Three perception
algorithms written by three people must share one desktop and one GPU, and must
never run at the same time:

```text
tool detection (RF-DETR)          최비결                 not on this machine yet
hand keypoints (MediaPipe)        Han Nwae Nyein         ~/hand_keypoints_ros
blood detection                   third member           not on this machine yet
```

Only the hand algorithm exists locally, so the other two — and the robot — are
represented by stubs. The point of this pass is to make the **sequencing**
runnable and verifiable before the other algorithms arrive, and to pin down what
those algorithms will have to provide.

## Decisions

### Separate workspace, not a package inside the hand repository

The coordinator is a team-level artifact that 최비결 and the blood-detection
member also depend on. Putting it inside `hand_keypoints_ros` would force them
to clone a hand-detection repository to get it.

```text
~/surgical_robot/
├── coordinator_ws/                 this workspace
├── hand_keypoints_ros/             (currently still at ~/hand_keypoints_ros)
├── surgical_tool_detection_ros/    to be cloned from 최비결
└── blood_detection_ros/            to be cloned
```

Each algorithm keeps its own repository and its own virtual environment; their
package versions conflict.

### Lifecycle nodes, not an `enabled` boolean

This was the main design decision. The obvious approach — give each detector an
`enabled` flag so disabled ones ignore camera frames — does not solve the
problem that actually binds. A disabled node's model **stays resident in VRAM**.
With MediaPipe + Depth-Anything V2 + RF-DETR + a blood model on a 12 GB card,
memory is the constraint, not CPU.

ROS 2 lifecycle separates the two cases properly:

| State | Model in VRAM | Processing frames | Re-activation |
| --- | --- | --- | --- |
| `unconfigured` | no | no | full model load |
| `inactive` | yes | no | instant |
| `active` | yes | yes | — |

The coordinator's `release_gpu_between_tasks` parameter picks between cleaning
detectors to `unconfigured` (frees VRAM, costs a reload per turn) and stopping
at `inactive` (instant turns, costs VRAM). Default is `true` — the safe one.
Flip it to `false` only after `nvidia-smi` confirms all three models co-fit.

### `PoseStamped` as the common detector result

The coordinator accepts `geometry_msgs/PoseStamped` from all three detectors, so
it needs no build dependency on any detector's private message package. Each
detector still publishes its own rich typed message for other consumers; the
coordinator just does not need it.

### Actions for robot operations

Grasp, handover and suction take seconds, can fail, and must be cancellable on
abort. Topics give no completion signal and services cannot be cancelled, so
these are actions.

## What was built

```text
coordinator_ws/src/
├── surgical_task_interfaces/          (ament_cmake)
│   ├── msg/TaskState.msg
│   └── action/{GraspTool,HandoverTool,SuctionBlood}.action
└── surgical_task_coordinator/         (ament_python)
    ├── task_coordinator.py            the state machine
    ├── lifecycle_detector.py          drives one detector's lifecycle
    ├── stub_detector.py               stands in for a missing algorithm
    ├── fake_robot_node.py             three action servers, moves nowhere
    └── launch/coordinator_stub_demo.launch.py
```

### Threading

The state machine runs on its own worker thread and blocks on service calls,
action results and detector poses, while a `MultiThreadedExecutor` spins in the
main thread. Every client uses a `ReentrantCallbackGroup`. A single-threaded or
mutually-exclusive setup deadlocks the moment the worker waits on something the
executor still has to deliver.

The blocking helpers deliberately poll `future.done()` rather than calling
`spin_until_future_complete`, because spinning from the worker thread while the
executor spins in the main thread is re-entrant.

### Stale-pose guard

Detectors keep publishing right up until they are deactivated, so the previous
round's pose is still on the wire when the next round starts. `_PoseWaiter` is
cleared **before** the detector is activated; without that the coordinator would
routinely "succeed" instantly on last round's data.

## Changes to hand_keypoint_ros

`hand_detection_node` was converted to a lifecycle node and moved onto the
ARPA-H contract topic names:

| Before | After |
| --- | --- |
| `hand/keypoints` | `/surgery/perception/cam4/hand_keypoints` |
| `hand/overlay_image` (`sensor_msgs/Image`) | `/surgery/images/cam4/hand_overlay/compressed` (`CompressedImage`) |
| `hand/target_pose`, only when `robot_position` set | `/surgery/perception/cam4/hand_target_pose`, always |
| — | `/surgery/perception/handkeypoint/health` |
| — | `/surgery/perception/handkeypoint/diagnostics/json` |

Model loading moved from `__init__` into `on_configure()`; `on_cleanup()` closes
the MediaPipe landmarker, drops the Depth-Anything references and calls
`torch.cuda.empty_cache()`.

`autostart` (default `true`) makes the node configure and activate itself on
startup, so running it standalone behaves exactly as before. The coordinator
launches it with `autostart:=false`.

Two side effects worth noting:

- `depth_source:=auto` detection now runs at configure time rather than at
  process start. When the coordinator activates the node long after the camera
  came up, auto-detection is more reliable than it used to be — the verification
  run below picked `REAL DEPTH`, where earlier runs had fallen back to mono.
- The target pose is now published whenever a hand has a valid `palm_6d`, not
  only in `robot_position` mode. Without `robot_position` the first such hand is
  chosen.

## Verification

### Coordinator flow, stub detectors

```bash
ros2 launch surgical_task_coordinator coordinator_stub_demo.launch.py
ros2 topic pub --once /surgery/task/command std_msgs/msg/String "{data: 'REQUEST_TOOL:scalpel'}"
```

Observed sequence, trimmed:

```text
[TOOL_DETECTION] locating tool "scalpel" (activating tool_detection_node)
[tool_detection_node] ACTIVE: searching for target
[tool_detection_node] INACTIVE: stopped processing, model still loaded
[tool_detection_node] cleaned up: model released
tool target: (0.100, 0.050, 0.400) in frame "cam4_color_optical_frame"
[ROBOT_GRASP_TOOL] grasping "scalpel"   → 7% → 40% → 73%
[HAND_DETECTION] locating surgeon's hand (activating hand_detection_node)
[hand_detection_node] ACTIVE: searching for target
[hand_detection_node] cleaned up: model released
hand target: (-0.080, 0.020, 0.550) in frame "cam4_color_optical_frame"
[ROBOT_HANDOVER] handing over "scalpel" → 7% → 40% → 73%
[IDLE] "scalpel" handed to the surgeon
```

The invariant holds: the tool detector reached `cleaned up` **before** the hand
detector reached `ACTIVE`. The two never overlap.

### Real hand detector, unchanged standalone behaviour

```bash
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py loop:=true
```

- 46 `published hand keypoints` messages over a 70 s run
- `autostart:=true -- self-configuring and activating`
- `depth source: REAL DEPTH`
- `configured.` then `ACTIVE: processing camera frames`
- Both processes reported `finished cleanly` on Ctrl-C

### Not yet verified

- `SUCK_BLOOD` and `ABORT` were implemented but not exercised in a run.
- The real hand node has not yet been driven **by the coordinator**
  (`use_stub_hand:=false` + `autostart:=false`); only the stub path was run.
- No measurement of whether `on_cleanup()` genuinely returns VRAM on the real
  hand node. `nvidia-smi` before/after a cleanup is the outstanding test, and it
  is the one that decides the `release_gpu_between_tasks` default.

## Fixed along the way

`rclpy.shutdown()` in both hand nodes' `finally` blocks raised
`RCLError: rcl_shutdown already called` on every Ctrl-C, so both processes
exited 1 and `ros2 launch` reported "process has died" on a perfectly normal
shutdown. In Jazzy the default signal handler has already shut the context down
by then. Fixed by catching `ExternalShutdownException` and using
`rclpy.try_shutdown()`. Node construction was also moved inside the `try`, since
a Ctrl-C during the multi-second model load or depth preload escaped `main()` as
a raw `KeyboardInterrupt`.

MediaPipe's GPU delegate logs `tensor.cc:410 Tensors are designed for single
writes` several times per frame, burying all other output. It is spurious — the
landmarker is only ever driven from one thread. `GLOG_minloglevel` does **not**
suppress it (MediaPipe 0.10.18 logs via absl, not glog); it is now silenced at
the file-descriptor level around the `detect_for_video()` call only, in
`core.py`. Set `HAND_KEYPOINTS_MEDIAPIPE_LOGS=1` to see it again.

## RF-DETR and concurrency update (2026-08-12)

Professor guidance: tool and blood detection share one RF-DETR algorithm and
differ only by weights/configuration. Use one codebase at
`~/surgical_robot/rfdetr_perception_ros/` with two parameterized lifecycle
node instances. The supplied tool directory currently lives at
`/home/miruware/PNU_CVLAB/choivy/sam3/experiments/arpa_h_rfdetr/transfer/cam4_seg8_local_20260810`.
Transfer its checkpoint, configs, ontology, test image, docs, metadata and
`SHA256SUMS` together.

Blood code is still missing. The current acceptance plan is real hand + real
tool when transferred + blood stub + fake robot, while all command branches
remain covered. Replace the blood stub after its parameters arrive.

Although tool and hand consume the same camera, the required workflow remains
sequential: tool locate → robot grasp → hand locate → robot handover.
Concurrent tool+hand execution is an optional stress test only and does not
replace the take-turn test.

## RF-DETR local transfer completed (2026-08-12)

The tool bundle was copied from the Windows folder
`C:\Users\user\Documents\arpa-h\August\tools detection\cam4_seg8_local_20260810`
to `~/surgical_robot/rfdetr_perception_ros/`. The checkpoint, standalone
inference script, environment YAML, tool ontology and sample image all passed
`sha256sum -c SHA256SUMS`.

The bundle requests Python 3.12 with Torch 2.7.0+cu118 and RF-DETR 1.8.3.
These dependencies will use a dedicated RF-DETR environment and will not be
installed into `~/hand_keypoints_ros_ws/.venv`.

## RF-DETR bundle inspection (2026-08-12)

Inspection confirmed that the transferred tool bundle is inference-only. The
only executable is `scripts/standalone_inference.py`, which requires one
`--image`; there are no ROS imports, camera subscriptions, publishers,
lifecycle nodes, launch files, video replay code or fake-camera publisher.

The teammate's previously demonstrated `rfdetr_perception_bridge` and
`cam4_image_publisher` belong to a separate ROS project that was not included.
That ROS code must be transferred, or an equivalent lifecycle wrapper/test
publisher must be implemented, before coordinator integration.

## RF-DETR performance evidence inspection (2026-08-12)

No valid FPS measurement was included in the transferred bundle. The only
timing is 409.37 ms for one `--no-optimize` single-image sanity run, and the
metadata explicitly marks it as not benchmark eligible. The documented real
benchmark depends on `benchmark_local_realtime.py` and a CAM4 video, neither
of which was transferred. Request those files or implement an equivalent
warm-up-excluded FP16 benchmark before reporting RF-DETR pipeline FPS.

## 2026-08-12 - RF-DETR local sanity inference passed

- Ran `scripts/standalone_inference.py` in the RF-DETR repository's Python 3.12 virtual environment.
- Used task `cam4_seg8`, the supplied fine-tuned checkpoint, validation image, threshold `0.5`, and `--no-optimize`.
- The model loaded successfully and detected **9 instances**, matching the supplied bundle's expected result.
- Detailed output was saved to `rfdetr_perception_ros/results/local_single_frame_sanity.json`.
- The script reported **491.93 ms** for this unoptimized single-image run.
- The deprecation and DINOv2 architecture messages were warnings, not inference failures.
- This run verifies installation/model correctness only. It does not measure ROS streaming throughput or valid real-time FPS.
- Next: repeat without `--no-optimize`, then obtain or create a warm-up/repeated-frame benchmark and integrate the teammate's ROS bridge.

## 2026-08-12 - Added RF-DETR frame-by-frame video benchmark

- Added `rfdetr_perception_ros/scripts/benchmark_video.py` because the transferred `standalone_inference.py` accepts only one image and cannot measure video FPS.
- The benchmark reads AVI frames sequentially without preloading the video.
- Added configurable warm-up and measured-frame limits.
- It records model-only `inference_fps`, complete decode-plus-inference `end_to_end_fps`, source FPS, real-time ratio, GPU/CPU device, and detection counts in JSON.
- Syntax and CLI help were verified in the RF-DETR Python 3.12 virtual environment.
- No performance number has been recorded yet; a benchmark command must finish before an FPS claim is made.

## 2026-08-12 - RF-DETR optimized video benchmark result

- Tested `0618_2_cam4_rgb_11m11s_to_11m41s.avi` at 1280x720 and 15 FPS.
- Used the RTX 3060, optimized FP16 inference, 10 warm-up frames, and 300 measured frames.
- Model-only inference: **40.93 FPS**.
- Sequential decode-plus-inference: **36.06 FPS**.
- Real-time ratio: **2.40x** relative to the 15 FPS source; `keeps_up_with_source=true`.
- Average detections: 4.01 per measured frame.
- Result JSON: `rfdetr_perception_ros/results/benchmark_0618_300frames.json`.
- The tracing/deprecation output occurred during model optimization and did not indicate a failure.
- This result does not yet include ROS transmission, overlay drawing, or coordinator overhead.

## 2026-08-12 - Hand-pipeline real-time conclusion clarified

- Confirmed that the live WSL2/RTX 3060 hand pipeline already meets the 15 FPS throughput target.
- Live input was approximately 32 Hz and hand-keypoint output was approximately **27.4-28.4 Hz**, or **~1.86x real-time**.
- No inference-speed fix is presently required for Node 2.
- The remaining blocking issue is 3D correctness: use depth aligned to the RGB/color frame, the color CameraInfo intrinsics, and synchronized timestamps before accepting live 3D keypoints.
- The ~9.3-10.2 Hz HDF5 replay result is specific to the fake-camera recorded-data/synchronization path and is not the live detector throughput.
- Production runs should use `publish_overlay:=false` when visualization is unnecessary to reduce CPU and network overhead.

## 2026-08-12 - Blood stub coordinator flow verified

- Published `SUCK_BLOOD` on `/surgery/task/command` with a matching coordinator subscriber.
- The coordinator transitioned through `BLOOD_DETECTION` and `ROBOT_SUCTION`, then returned to `IDLE` after the fake suction action completed.
- The blood lifecycle stub configured, activated, published a fake target `(0.020, 0.120, 0.450)` in `cam4_color_optical_frame`, deactivated, and cleaned up its fake model.
- The flow completed successfully twice in the captured launch log.
- An empty `/surgery/task/state` echo window did not indicate failure: it was started after the short-lived state messages had already been published and was then stopped with Ctrl-C.
- This verifies coordinator sequencing only; the real blood RF-DETR model is still unavailable.

## 2026-08-12 - Added real tool/hand sequential result-receipt test

- Added a lifecycle RF-DETR tool ROS node at `rfdetr_perception_ros/scripts/tool_detection_ros_node.py`.
- It subscribes to shared RGB, loads/releases its model through ROS lifecycle transitions, and publishes compact semantics JSON, COCO RLE masks, overlay JPEG, health, and diagnostics.
- Added `coordinator_ws/scripts/perception_result_receiver.py`, a downstream node that confirms receipt of both real tool JSON and typed hand-keypoint output without dumping large masks.
- Added `coordinator_ws/scripts/run_real_tool_hand_take_turn_test.sh` to automate one shared-video publisher, tool turn, GPU cleanup, hand turn, and downstream verification.
- Updated the test to use the matched 0618 RGB AVI, raw HDF5 depth, and calibration under `~/surgical_robot/test_data/cam4_0618`.
- Hand detection is forced to real depth and HDF5 full preload is disabled.
- Replaced a hanging `ros2 lifecycle get` discovery probe with `ros2 service list --no-daemon` for WSL compatibility.
- Python and shell syntax/startup smoke checks passed. The full real-model sequential run remains to be completed by the one-command test.
- The tool node publishes real image-space detections but no invented 3D pose; aligned-depth 2D-to-3D tool projection remains separate work.

## 2026-08-12 - Fixed stale ROS CLI daemon in lifecycle transitions

- The first matched-dataset run reached the tool turn but failed before model loading because `ros2 lifecycle set` attempted to contact a stale ROS CLI daemon and timed out.
- Discovery had already been changed to `ros2 service list --no-daemon`, but the configure/activate/deactivate/cleanup commands still used the daemon.
- Updated every lifecycle transition to use `--no-daemon --spin-time 1.0` with bounded timeouts.
- Cleanup transitions now also use daemon-free calls with short timeouts, preventing a failed test from hanging during its exit trap.
- Background launch wrappers now use `exec`, and the result receiver PID is tracked directly, so cleanup does not leave child detector processes behind.
- Removed the known orphaned processes from both failed runs and revalidated the shell script syntax.

## 2026-08-12 - Fixed real tool-to-hand test stopping after tool receipt

- The real RF-DETR lifecycle node configured and activated successfully on the RTX 3060.
- The downstream receiver successfully received a real tool result containing 5 instances.
- The script nevertheless stopped before the hand turn because a second, redundant `ros2 topic echo --once` check used the stale WSL ROS CLI daemon and timed out after 60 seconds.
- Removed both daemon-dependent topic-echo stage checks.
- The script now advances only when the already-running downstream receiver logs `RECEIVED TOOL RESULT` and then `RECEIVED HAND RESULT`.
- Revalidated Bash syntax and confirmed no remaining `topic echo` command exists in the automated test.

## 2026-08-12 - Fixed transient node discovery during tool deactivation

- A rerun successfully configured and activated real RF-DETR on the RTX 3060.
- The downstream receiver received a real tool result with 4 instances.
- The next deactivation command intermittently returned `Node not found` because a one-second daemon-free discovery window was too short under WSL/DDS.
- The script exited before the hand turn; the context-invalid publish message was a shutdown consequence, not an inference failure.
- Updated the lifecycle transition helper to use a three-second direct discovery window and retry only the transient `Node not found` condition up to ten times.
- Syntax validation passed and the remaining stale helper process from the original run was removed.

## 2026-08-12 - Added signal-driven perception mode switching

- The latest fixed-order run successfully configured/activated real RF-DETR, delivered real tool results, deactivated/cleaned RF-DETR, configured/activated the real GPU hand node, and delivered hand topic messages.
- Its receiver initially declared success on a zero-hand message. A later frame reported one hand, but the original success condition was too weak.
- Updated the receiver: tool requires at least one instance; hand requires at least one detected hand, valid depth-backed 3D keypoints, and a palm pose.
- Added `perception_mode_coordinator`, driven by `/surgery/perception/mode_command`.
- Supported signals: `DETECT_TOOL`, `DETECT_HAND`, `DETECT_BLOOD`, and `STOP` (with short aliases).
- A new signal stops/cleans the current lifecycle detector before loading and activating the requested detector.
- Added `scripts/run_signal_driven_perception_test.sh`: matched 0618 RGB/HDF5/calibration, real tool, real hand, blood stub, state coordinator, and downstream receiver.
- Rebuilt the ROS package successfully.
- A no-signal startup smoke test confirmed all nodes coexist initially in IDLE/UNCONFIGURED state. Added a ROS-context guard found by the forced-timeout cleanup test.
- Full interactive signal-order validation remains pending. Blood remains a stub because its real weights/config are unavailable.

## Open questions for the team

1. **Tool and blood result contracts.** Topic name, message type, and what
   "detection succeeded" means for each. The coordinator currently assumes
   `PoseStamped` on `/surgery/perception/cam4/{tool,blood}_target_pose`; those
   names are placeholders invented here, not agreed.
2. **Camera source.** Do all three algorithms read the same camera topic? The
   contract table mentions both `cam4` and `flir`.
3. **Hand-keypoint rows in the ARPA-H table.** They are missing entirely.
   Proposed rows are listed in the README. 최비결's message says the existing
   entries were filled in arbitrarily, so the whole CVLAB block needs review.
4. **`HandKeypoints` message ownership.** It currently lives in the private
   `hand_keypoint_interfaces`. If the consuming organization must deserialize
   it, it belongs in the shared `surgical_msgs` instead.
5. **Robot action definitions.** `GraspTool` / `HandoverTool` / `SuctionBlood`
   are a proposal to the robot-control team, not something they have accepted.
6. **Voice command message.** Currently `std_msgs/String` on
   `/surgery/task/command`. Needs replacing with the voice team's type.

## Next steps

1. Run `nvidia-smi` across a real cleanup to settle `release_gpu_between_tasks`.
2. Drive the real hand node from the coordinator end-to-end.
3. Exercise `SUCK_BLOOD` and `ABORT`.
4. Get the RF-DETR checkpoint and script from 최비결 as a git repository rather
   than a directory copy — the weights likely need Git LFS or `scp`.
5. Take the two contract questions above to the team before anyone implements
   against the placeholder names.

## 2026-08-12 — Full README status audit

Audited the complete coordinator README against the current Tool and Hand node sources, local interface CSVs, completed tests, `test_history.md`, and existing worklogs. Updated the detector availability table, coordinator scope wording, Hand contract status, ROS build-dependency wording, Tool/Blood integration checklist, original RF-DETR bundle limitation, benchmark status, shared-camera status, parallel Tool + Hand results, and known gaps. Current summary: Tool and Hand are real and tested; Blood remains a lifecycle stub; parallel Tool + Hand passed functionally at 11.583 Hz and 9.923 Hz respectively but did not meet the 15 FPS source rate; real camera-driver/TF/robot integration remains pending.
## 2026-08-12 — Measured signal-driven Tool -> Hand -> Blood-stub run

Completed the available signal-driven sequence using the 15 FPS frame-by-frame fake-camera replay with overlays enabled. Real Tool measured 7.92 Hz; command-to-active was 6.93 s and command-to-first-result was 10.52 s. Tool cleanup completed 0.44 s after `DETECT_HAND`; real Hand became active after 2.83 s, published its first message after 17.14 s and first nonzero-hand message after 19.23 s, and measured 8.33 Hz. Hand cleanup completed 0.12 s after `DETECT_BLOOD`; the Blood stub became active after 2.17 s and published its first fake target after 3.65 s. The available real Tool/real Hand/stub Blood switching test passed; a three-real-detector test remains blocked on the real Blood checkpoint/configuration.
## 2026-08-12 — Migrated coordinator to Surgical Tool Component v1.3.0-rc1

- Copied delivery verified separately at `~/surgical_robot/tool_detection_component_v1_3_rc1`; all `SHA256SUMS` entries passed. The previous `rfdetr_perception_ros` implementation remains untouched for rollback.
- Installed `pnu-surgical-tool==1.0.0rc1` into the existing Tool virtual environment with `--no-deps`.
- Added `surgical_perception_msgs` v0.2.0 to `coordinator_ws/src/` and successfully built all three workspace packages.
- Added `scripts/tool_detection_v13_ros_node.py`, a lifecycle host adapter for frame-by-frame RGB, exact-stamp aligned depth, and CameraInfo input.
- Canonical outputs are now typed `/surgery/perception/cam4/observations` and `/surgery/perception/cam4/tool_poses`; overlay, health, diagnostics, and a compatibility semantics JSON are also published.
- Updated both fixed-order and signal-driven launchers to use v1.3. Updated the downstream receiver to require the typed observation array, typed pose array, and a usable real Hand result.
- Added `scripts/validate_tool_v13_integration.sh`. Static contract, pose contract, ROS mapping, interface build/introspection, Python compilation, and shell syntax checks all passed.
- Final sequential runtime test passed: receiver obtained 5 v1.3 Tool observations, a 5-entry ToolPoseArray with the same `cam4:0` observation ID, then 1 real hand with 10 valid depth-backed keypoints and 1 palm pose.
- The delivery contains no calibrated CAM4 support-plane values. Tool poses therefore correctly report `valid_poses=0`/INVALID. Detection, masks, messages, overlay, lifecycle switching, GPU cleanup, and downstream reception are verified; metric Tool pose validation remains blocked on the real support-plane normal, offset, and config version.
## 2026-08-12 — Fixed v1.3 COCO-RLE output bottleneck on PNU-4

- Initial live diagnostics: 0.125 Hz, 193.21 ms GPU inference, and 7854.82 ms complete processing time with about 11 instances.
- Root cause: the delivered reference mapper used a Python per-pixel loop to create compressed COCO RLE for every full-resolution mask.
- The host adapter now substitutes the equivalent compiled `pycocotools` encoder; the delivered component files remain unchanged.
- Full component/interface validation passed after the change.
- PNU-4 typed observation output improved to approximately 8.03 Hz on the 15 FPS replay (about 64x faster than 0.125 Hz), but remains below real time.
- Verified the overlay publisher independently: valid 1280x720 JPEG, 167137 bytes, normal non-black pixel statistics. The observed black viewer had no active subscription and was not a publisher/image-content failure.

## 2026-08-12 — Added dedicated Tool v1.3 parallel test report

- Added `~/surgical_robot/tool_v1_3_integration_test_results.md` as the consolidated report for Surgical Tool Component v1.3.0-rc1.
- Recorded the PNU-4 frame-by-frame RGB/real-depth test setup, six Tool output interfaces, functional results, RLE optimization history, GPU observations, and remaining support-plane limitation.
- Recorded the latest parallel rates: Tool observations **8.028 Hz** and Hand keypoints **8.905 Hz** from a 15 FPS source. Both detectors ran concurrently without GPU out-of-memory, but the run did not meet 15 FPS real time.
## 2026-08-12 — Implemented startup model preloading

- Added `LifecycleDetector.configure()` so models can be loaded once and left INACTIVE before the first command.
- Added coordinator parameters `preload_models_on_startup` (default true) and `release_gpu_between_modes` (default false).
- Updated the signal-driven launcher to preload real Tool, real Hand, and the Blood stub sequentially, then switch by deactivate/activate without cleanup.
- Rebuilt all three coordinator workspace packages successfully.
- Full test passed: typed v1.3 Tool observations, real-depth Hand messages, and Blood-stub pose were received.
- Cold preload was 14.77 s; repeat preload was 11.60 s. Command-to-ACTIVE was 0.10 s Tool, 0.12 s Hand, and 0.12 s Blood stub.
- First result was 3.74 s Tool, 14.73 s Hand message / 15.76 s nonzero Hand, and 1.60 s Blood stub.
- Preloaded GPU snapshot: 3016/12288 MiB total, no OOM. Real Blood is still absent and must be rechecked when integrated.
- Remaining issue: Hand activation is now immediate, but recorded RGB/HDF5 synchronization still delays its first output by about 14.6 s.
## 2026-08-12 — Fixed preload coordinator startup regression

- A user rerun exposed two truncated Python identifiers in `perception_mode_coordinator.py`: `MultiThreadedExecuto` and `get_paramete`.
- Restored `MultiThreadedExecutor` and `get_parameter`, rebuilt all coordinator packages, and added an import-level smoke test that reproduces the original failure path.
- Final startup verification passed: Tool, Hand, and Blood stub all reached INACTIVE/model-resident state and the coordinator published `all detector models preloaded; waiting for command` with no traceback.
## 2026-08-12 — Expanded and renamed the Tool v1.3 results report

- Renamed `~/surgical_robot/tool_v1_3_parallel_test_results.md` to `~/surgical_robot/tool_v1_3_integration_test_results.md` because the document now covers all work after Tool v1.3 adoption, not only parallel execution.
- Added the v1.3 adapter and typed-interface changes, COCO-RLE optimization, sequential and parallel results, cleanup/reload baseline, startup-preload implementation and timing, latest 0.019 s Hand ACTIVE confirmation, complete reproduction commands, interpretation, and remaining work.
- Updated all README/worklog references to the new report name.
## 2026-08-12 — Added combined-system operation manual

- Added `~/surgical_robot/OPERATION_MANUAL.md` for the prepared WSL workstation and fresh-clone asset requirements.
- Documented build steps, startup model preloading, Tool/Hand/Blood-stub commands, lifecycle and interface checks, FPS and GPU measurement, overlay viewing, correct switching-latency calculation, shutdown, troubleshooting, and current limitations.
- Linked the manual from the combined repository README and coordinator README.
## 2026-08-13 — Added beginner combined-system guide

- Added `~/surgical_robot/BEGINNER_SYSTEM_GUIDE.md` as a plain-language explanation of ROS nodes/topics, camera input, lifecycle preloading, command flow, and the runtime files that matter.
- Documented all five Hand output interfaces and all six Tool v1.3 output interfaces, including message contents, publication conditions, validity checks, intended consumers, and when each interface is needed.
- Clarified that algorithm topic names remain stable across standalone, sequential, parallel, and signal-driven modes unless explicitly overridden; inactive/conditional publication behavior is separate from interface existence.
- Linked the beginner guide from the combined README, coordinator README, and operation manual.
## 2026-08-13 - Documented Tool and Hand overlay visualization

- Added the exact `image_view` commands for Tool and Hand overlays to the coordinator README and test history.
- Documented the matching `/compressed` topic-rate checks.
- Clarified that Jazzy `image_view` uses `-p image:=<base_topic>` with `-p image_transport:=compressed`; the earlier `-r image:=...` form can leave the viewer unsubscribed or black.
- Clarified that take-turn mode updates one overlay at a time, while the parallel test supports two simultaneous viewer windows.
