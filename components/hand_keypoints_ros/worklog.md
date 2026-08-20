# Hand Keypoints ROS 2 Work Log

Date: 2026-08-11
Local machine: Intel Core i5-14400F, NVIDIA GeForce RTX 3060 12 GB
Environment: Windows 11 + WSL2 Ubuntu 24.04 + ROS 2 Jazzy

## Objective

Run and verify the hand-keypoint pipeline as ROS 2 nodes on the local Windows PC through WSL2. A fake camera replays recorded RGB, aligned-depth, and calibration data because the physical camera and its ROS driver are not currently available.

## Repository and environment

- Repository: `https://github.com/hanwae-py/hand_keypoints_ros`
- WSL repository: `~/hand_keypoints_ros`
- Python virtual environment: `~/hand_keypoints_ros_ws/.venv`
- ROS workspace: `~/hand_keypoints_ros/ros2_ws`
- ROS distribution: Jazzy
- Ubuntu version: 24.04

The virtual environment must be activated before building. Use `python -m colcon`, not plain `colcon`, so generated ROS console scripts use the virtual-environment Python containing MediaPipe, h5py, and the other ML dependencies.

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
cd ~/hand_keypoints_ros/ros2_ws
python -m colcon build --symlink-install
source install/setup.bash
```

Expected detector shebang:

```bash
head -1 install/hand_keypoint_ros/lib/hand_keypoint_ros/hand_detection_node
```

```text
#!/usr/bin/env python3
```

## GPU verification

Native Windows MediaPipe Python could not use its GPU delegate and previously fell back to CPU. Under WSL2, GPU passthrough works:

- EGL initialized successfully.
- OpenGL renderer reported `D3D12 (NVIDIA GeForce RTX 3060)`.
- TensorFlow Lite created its GPU delegate.
- MediaPipe reported `hand landmarker ready (GPU delegate)`.

This confirms that the ROS detector is able to use the RTX 3060 through WSL2.

## ROS 2 test flow

```text
Recorded RGB/depth/calibration
        -> fake_camera_publisher
        -> ROS camera topics
        -> hand_detection_node
        -> /surgery/perception/cam4/hand_keypoints
```

The fake camera is only a test replacement for the future physical camera node. The detection node should not need to change when the real camera publishes compatible topics and messages.

Launch continuously:

```bash
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py loop:=true
```

Launch for one playback only:

```bash
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py loop:=false
```

Low-memory frame-by-frame HDF5 replay:

```bash
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py \
  preload_depth:=false \
  loop:=false
```

In this mode the fake camera publishes one synchronized RGB frame, depth
frame, and `CameraInfo` message per timer tick. Calibration values are
loaded once because they are constant, but `CameraInfo` is published for
every frame.

Inspect one result message from another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
source ~/hand_keypoints_ros/ros2_ws/install/setup.bash
ros2 topic echo /surgery/perception/cam4/hand_keypoints --once
```

Stop a looping launch with `Ctrl+C`. Currently, `loop:=false` cancels the
fake publisher's timer after one playback but does not terminate the ROS
launch, so that mode also needs `Ctrl+C` after
`source clip finished, stopping publisher` appears.

## Problems found and resolutions

### Missing Python dependencies after rebuilding

Running plain `colcon build` generated scripts using `/usr/bin/python3`, which could not import packages installed in the virtual environment, including MediaPipe and h5py.

Resolution: activate the virtual environment and always build with `python -m colcon build --symlink-install`. Manually changing files under `install/` is only temporary because rebuilding regenerates them.

### Missing shared core module in a native WSL build

The Docker build copied `scripts/hand_keypoints_core.py` into the ROS package, but the initial native build did not. This caused `ModuleNotFoundError: hand_keypoint_ros.core`.

Resolution: ensure the shared core module is included in the ROS package before building natively.

### Docker-only data path

The fake-camera launch initially looked for calibration data under `/repo/data`, which exists in the Docker image but not in the native WSL checkout. It therefore crashed with `FileNotFoundError` and the detector fell back to monocular depth.

Resolution: use the repository data directory under `~/hand_keypoints_ros/data` for native WSL execution, while retaining `/repo/data` as the Docker fallback.

### MediaPipe `tensor.cc:410` messages

MediaPipe's GPU delegate repeatedly printed messages about multiple tensor writes. Testing confirmed that detection continued and `/surgery/perception/cam4/hand_keypoints` was published successfully. These were noisy native MediaPipe logs rather than a pipeline failure. Native log suppression was limited to the MediaPipe detection call; setting `HAND_KEYPOINTS_MEDIAPIPE_LOGS=1` restores those logs for debugging.

### Shutdown error on Ctrl+C

ROS 2 Jazzy's signal handler had already shut down the context, after which unconditional shutdown code attempted to shut it down again. Normal Ctrl+C termination was consequently reported as a process failure.

Resolution: handle `ExternalShutdownException`, use `rclpy.try_shutdown()`, and guard node destruction. Consecutive launch tests then ended cleanly.

### `All-NaN slice encountered`

This warning means that no valid depth value existed in the sampled depth patch for some detected landmarks. It is non-fatal and processing continues. It may require separate handling later if downstream consumers require every 3D point to be valid.

### Full HDF5 preload versus frame-by-frame replay

The bundled real-depth dataset has shape `(2462, 720, 1280)`, dtype
`uint16`, gzip level 4, and HDF5 chunk shape `(32, 45, 80)`. It occupies
about 574 MB on disk but expands to about 4.54 GB in RAM.

The original fast replay path loads the complete decompressed array into
RAM. A first low-memory implementation indexed one HDF5 frame per timer
tick, but this repeatedly decompressed the same multi-frame chunks and was
not representative or efficient.

Resolution: add a `preload_depth` parameter. The default `true` preserves
the maximum-speed full preload. With `preload_depth:=false`, the publisher
reads aligned 32-frame blocks, keeps the current and next blocks in a
background double buffer, and still publishes only one frame per tick.
The buffer is about 118 MB rather than 4.54 GB.

Isolated local benchmarks for three aligned 32-frame reads were:

| Frames | Read time | Equivalent depth-read rate |
| --- | ---: | ---: |
| 0-31 | 0.130 s | 247.1 frames/s |
| 32-63 | 0.123 s | 260.9 frames/s |
| 64-95 | 0.127 s | 252.9 frames/s |

These figures measure only HDF5 reads, not the complete ROS pipeline.

## Code changes made during local WSL integration

- Added the shared `core.py` module to the native ROS package path.
- Changed fake-demo data discovery from Docker-only `/repo/data` to the
  native repository `data/` directory with `/repo/data` retained as a
  Docker fallback.
- Fixed clean ROS2 Jazzy shutdown handling using
  `ExternalShutdownException`, `rclpy.try_shutdown()`, and guarded node
  destruction.
- Limited suppression of MediaPipe GPU delegate `tensor.cc:410` spam to
  the native detection call; `HAND_KEYPOINTS_MEDIAPIPE_LOGS=1` restores it.
- Added `preload_depth` and chunk-aligned background HDF5 prefetch.
- Rebuilt with the venv interpreter using
  `python -m colcon build --symlink-install`.

The package rebuilt successfully. The remaining setuptools message about
dash-separated `script-dir` is a deprecation warning, not a build failure.

## Local ROS rate measurements

Test configuration:

- `preload_depth:=false`
- `loop:=false`
- fake-camera target rate: 15.0 Hz
- MediaPipe: WSL2 GPU delegate on RTX 3060
- depth source: real HDF5 depth; no Depth-Anything model

Observed results (diagnostic run; not a valid inference benchmark):

| Measurement | Result |
| --- | ---: |
| `/camera/color/camera_info` | normally 14.96-15.00 Hz |
| `/camera/color/image_raw` through Python `ros2 topic hz` | cumulative average reached 10.18 Hz |
| old `/hand/keypoints` (50-message window) | 7.47-8.60 Hz; latest 8.46 Hz |
| Benchmark status | **Invalid for judging inference speed** |

### 8.46 Hz investigation and fix history (2026-08-11)

1. The fake camera was configured to replay RGB, real HDF5 depth, and
   CameraInfo at 15 Hz with the MediaPipe GPU delegate active on the local
   RTX 3060.
2. `ros2 topic hz /camera/color/image_raw` was started to check the source
   rate. This added another Python subscriber that had to receive and
   deserialize every 1280x720 raw image.
3. While that heavy monitor was running, the cumulative raw-image rate reached
   only 10.18 Hz and one 2.117-second publishing interruption was recorded.
4. `/hand/keypoints` was then observed at 7.47-8.60 Hz, with 8.46 Hz as the
   latest 50-message-window value. It was initially recorded as a failure to
   meet the 15 FPS real-time target.
5. Inspection found that the fake camera used ROS's default RELIABLE QoS.
   Consequently, the added raw-image subscriber could apply reliable-delivery
   backpressure and slow Node 1 itself. The test was measuring a disturbed
   replay pipeline, not the detector's clean GPU inference rate.

One 2.117-second publishing interruption was observed while
`ros2 topic hz` was also subscribing to the large raw RGB stream. The fake
camera was using default RELIABLE QoS, so that extra Python subscriber could
deserialize the image stream and apply backpressure to the source timer. In a
50-message window, that stall alone changes an ideal 15 Hz rate to about
9.1 Hz (`49 / (49/15 + 2.117)`). Therefore, the observed 8.46 Hz is a
measurement artifact and must not be used as the RTX 3060 inference FPS.

Fix applied:

- fake camera RGB, depth, and CameraInfo publishers changed to
  BEST_EFFORT/VOLATILE sensor QoS;
- detector diagnostics now publish `processed_hz_1s`, measured internally
  without adding a raw-image subscriber;
- performance documentation marks 8.46 Hz as invalid rather than
  “not real-time.”

Fix verification result:

| Check | Result |
| --- | --- |
| Python syntax check | Passed |
| `colcon build --symlink-install --packages-select hand_keypoint_ros` | Passed |
| Fake-camera publisher QoS | BEST_EFFORT / VOLATILE / KEEP_LAST depth 5 |
| Detector internal rate field | `processed_hz_1s` added to diagnostics JSON |
| Corrected clean RTX 3060 measurement | Completed; results below |

The fix result does **not** mean that 8.46 Hz became a particular new FPS.
It means the measurement path was corrected and the invalid 8.46 Hz verdict
was withdrawn.

A clean repeat should force real depth, stop the raw-image monitor, and
measure only:

```bash
ros2 topic hz /surgery/perception/cam4/hand_keypoints --window 50
```

Alternatively, inspect the node's internal rate without subscribing to the
raw image stream:

```bash
ros2 topic echo \
  /surgery/perception/handkeypoint/diagnostics/json \
  --once
```

### Corrected clean RTX 3060 measurement (2026-08-11)

The rerun forced `GALLIUM_DRIVER=d3d12`, confirmed
`renderer: D3D12 (NVIDIA GeForce RTX 3060)`, confirmed the MediaPipe GPU
delegate, forced `depth_source:=real`, and monitored no raw-image topic.

| Test | Observed result |
| --- | ---: |
| Requested fake-camera source | 15.0 Hz |
| Lightweight CameraInfo source check | **15.00 Hz** after startup |
| Hand-keypoint output between pauses | **~11.35 Hz** |
| Hand-keypoint output after one 2.388 s pause entered the 100-message window | **~9.22 Hz** |
| Internal diagnostic at the same time | **8.92 Hz**, last frame 52.1 ms, 0 errors |
| High-input test (`rate_hz:=100`) | **~25.4-30.9 Hz output** |

This corrected 15 Hz replay test is **not yet end-to-end real-time** because
the output does not sustain 15 Hz. However, the high-input test proves that
the GPU detector can process faster than 15 Hz. The remaining limitation is
in the recorded fake-camera/ROS synchronization path and its periodic pauses,
not the RTX 3060 inference ceiling.

A full-depth preload comparison did not fix the pauses: it produced roughly
8.3-10.2 Hz and contained a 2.607-second interruption. Therefore, the bounded
HDF5 prefetch implementation is not the main cause. The next optimization
work should instrument Node 1 publish intervals, synchronized callback arrival
times, and per-frame Node 2 processing latency to locate the periodic pause.

### Exact no-preload HDF5 frame-by-frame rerun (2026-08-11)

This is the replay test that most closely represents RGB and depth messages
arriving incrementally. It did **not** preload the complete HDF5 file:

```bash
ros2 launch hand_keypoint_ros fake_camera_demo.launch.py \
  preload_depth:=false \
  depth_source:=real \
  rate_hz:=15.0 \
  loop:=true
```

The HDF5 file remained on disk and the publisher emitted one RGB frame, one
depth frame, and one CameraInfo message per timer tick. Because the dataset is
gzip chunked, the implementation maintains a bounded two-chunk background
buffer (about 118 MB); it does not materialize the 4.54 GB depth array.

| Measurement | Result |
| --- | ---: |
| Full HDF5 preload | **Disabled** |
| Depth mode | Real HDF5 depth, frame-by-frame publication |
| MediaPipe | RTX 3060 GPU delegate confirmed |
| Source rate | **14.999-15.001 Hz** |
| Hand-keypoint output | **~9.3-10.2 Hz** |
| Internal one-second sample | **11.03 Hz** |
| Last-frame processing time | **49.3 ms** |
| Processing errors | **0** |
| 15 FPS real-time status | **Not yet real-time** (~0.62-0.68x) |

This latest controlled result supersedes the earlier ambiguous 8.46 Hz run.
It proves that Node 1 publishes at the requested 15 Hz without full preload,
while the combined ROS synchronization/detection/output path does not yet
sustain 15 result messages per second.

### Live frame-by-frame camera measurement (2026-08-11)

The detector was then run directly against the live ROS camera topics, with no
fake-camera node and no HDF5 file:

| Input/output | Topic | Measurement |
| --- | --- | ---: |
| RGB | `/camera/camera/color/image_raw` | 848x480, `rgb8` |
| Depth | `/camera/camera/depth/image_rect_raw` | 848x480, `16UC1` |
| Calibration | `/camera/camera/color/camera_info` | **~32 Hz** |
| Hand-keypoint output | `/surgery/perception/cam4/hand_keypoints` | **~27.4-28.4 Hz** |

The run confirmed `D3D12 (NVIDIA GeForce RTX 3060)`, the MediaPipe GPU
delegate, real-depth mode, 670 processed frames, and zero processing errors.
At about 28 Hz, the live pipeline has roughly **1.86x** the throughput required
for a 15 FPS source. This also confirms that the earlier 9-11 Hz behavior was
specific to the recorded fake-camera replay/synchronization path rather than
the detector's GPU capacity.

Important correctness limitation: this run validates live transport and
processing speed, but not the 3D coordinates. RGB uses
`camera_color_optical_frame`, while the available depth uses
`camera_depth_optical_frame`. The current detector assumes depth pixels are
already aligned to RGB pixels and applies color intrinsics. The real camera
publisher must enable aligned depth and provide a topic such as
`/camera/camera/aligned_depth_to_color/image_raw` before the resulting 3D hand
keypoints can be considered geometrically valid.

## Current result

The fake-camera ROS test is running end-to-end. The detector continuously
publishes `/surgery/perception/cam4/hand_keypoints`, with logs such as:

```text
published hand/keypoints: 0 hands
published hand/keypoints: 1 hands
published hand/keypoints: 2 hands
published hand/keypoints: 3 hands
```

Counts vary by frame because visibility and detector confidence change throughout the video. With `loop:=true`, the recording repeats indefinitely; this is expected behavior.

The local WSL GPU path and end-to-end ROS message flow are working. The old
8.46 Hz observation remains invalid. The exact no-full-preload HDF5
frame-by-frame replay supplies 15.00 Hz but currently produces approximately
9.3-10.2 Hz, so that required recorded-data test is not yet real-time. A
separate physical-camera throughput test reached 27.4-28.4 Hz, but its depth
was not RGB-aligned and therefore it does not replace the HDF5 replay result or
validate live 3D accuracy.

## Remaining work

- Confirm a clean rebuild still produces virtual-environment Python shebangs.
- Add automatic launch termination after `loop:=false` reaches EOF; at
  present it stops publication but requires Ctrl+C to exit both processes.
- Connect and test the real camera driver when the hardware is available.
- Confirm the real camera publishes compatible RGB, aligned-depth, and camera-info topics.
- Enable RGB-aligned depth on the real camera publisher and repeat the live
  test to validate 3D coordinate accuracy, not only throughput.
- Optionally instrument the fake-camera replay path if it still needs to
  reproduce the live-camera rate for offline testing.
- Decide whether combined 2D/3D output remains one node or is split into separate detection and 2D-to-3D nodes.
