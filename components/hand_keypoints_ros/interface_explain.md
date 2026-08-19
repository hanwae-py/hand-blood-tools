# Hand Detection ROS 2 Output Interfaces

## Short answer

The five `/surgery/...` topics below are outputs published by
`hand_detection_node`. The additional
`/hand_detection_node/transition_event` topic is generated automatically by
ROS 2 because the detector is a lifecycle node. It is not a hand-detection
result interface for downstream algorithms.

## Application output topics

| Topic | ROS 2 type | Current QoS | Purpose | When it publishes |
| --- | --- | --- | --- | --- |
| `/surgery/perception/cam4/hand_keypoints` | `hand_keypoint_interfaces/msg/HandKeypoints` | RELIABLE, VOLATILE, KEEP_LAST depth 10 | Main result for the next processing/robot node. Contains all detected hands, 21 pixel keypoints, depth-derived 3D keypoints in metres, handedness, validity flags, and palm 6D pose. | Once for every processed frame while the lifecycle node is active, including frames with zero hands. |
| `/surgery/perception/cam4/hand_target_pose` | `geometry_msgs/msg/PoseStamped` | RELIABLE, VOLATILE, KEEP_LAST depth 10 | Simplified robot handover target: position and orientation of one valid palm. | Only when at least one detected hand has a valid palm 6D pose. |
| `/surgery/images/cam4/hand_overlay/compressed` | `sensor_msgs/msg/CompressedImage` | RELIABLE, VOLATILE, KEEP_LAST depth 1 | JPEG preview with hand detections/keypoints drawn on the RGB image. Intended for a person, UI, debugging, or recording—not as the main robot input. | Every processed frame while `publish_overlay:=true`. |
| `/surgery/perception/handkeypoint/health` | `std_msgs/msg/String` | RELIABLE, VOLATILE, KEEP_LAST depth 10 | JSON readiness/status report: lifecycle state, `ok`/`degraded`, depth source, and seconds since the last frame. | Approximately 1 Hz in every lifecycle state. |
| `/surgery/perception/handkeypoint/diagnostics/json` | `std_msgs/msg/String` | RELIABLE, VOLATILE, KEEP_LAST depth 10 | JSON diagnostics: processed-frame count, last hand count, processing time, measured rate, error count, depth source, maximum hands, and CPU-only flag. | Approximately 1 Hz in every lifecycle state. |

The three detection-result publishers (`hand_keypoints`, `hand_target_pose`,
and overlay) are lifecycle publishers. They publish only while the detector is
`active`. Health and diagnostics remain available even when the detector is
inactive so the coordinator can inspect its status.

## Main `HandKeypoints` content

One message represents one RGB/depth camera timestamp:

- `header`: original camera timestamp and camera frame ID.
- `depth_source`: `real` for aligned sensor depth or `mono` for the
  Depth-Anything V2 fallback.
- `hands`: zero or more detected hands.

Each hand contains:

- handedness label and confidence;
- 21 `joints_2d` pixel coordinates `(u, v)`;
- 21 `joints_3d` points `(x, y, z)` in metres in the camera optical frame;
- per-keypoint score and valid-depth flag;
- optional palm 6D translation, quaternion, and rotation matrix.

The downstream node must check `kp_valid_depth` before using an individual 3D
joint and `has_palm_6d` before using the palm pose.

## Automatic lifecycle topic

| Topic | Type | Meaning |
| --- | --- | --- |
| `/hand_detection_node/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | Automatically reports lifecycle changes such as unconfigured → inactive → active. It is useful for orchestration/debugging but is not a hand result. |

## How to start the test pipeline

Open WSL Terminal 1:

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

Keep Terminal 1 running. Confirmation should include:

```text
depth source: REAL DEPTH
Created TensorFlow Lite delegate for GPU.
ACTIVE: processing camera frames
published hand keypoints: ... hands
```

## How to discover and verify the interfaces

Open WSL Terminal 2 and source the same environment:

```bash
source /opt/ros/jazzy/setup.bash
source ~/hand_keypoints_ros_ws/.venv/bin/activate
source ~/hand_keypoints_ros/ros2_ws/install/setup.bash
```

Use `--no-daemon` for discovery because a daemon started previously with a
different ROS domain can show a stale or empty topic list:

```bash
ros2 topic list --no-daemon | grep -E 'hand|surgery'
```

Expected output:

```text
/hand_detection_node/transition_event
/surgery/images/cam4/hand_overlay/compressed
/surgery/perception/cam4/hand_keypoints
/surgery/perception/cam4/hand_target_pose
/surgery/perception/handkeypoint/diagnostics/json
/surgery/perception/handkeypoint/health
```

### Check topic types

```bash
ros2 topic list --show-types --no-daemon | grep -E 'hand|surgery'
```

For one topic, also check the publisher count and actual QoS:

```bash
ros2 topic info \
  /surgery/perception/cam4/hand_keypoints \
  --verbose --no-daemon
```

The expected publisher count is at least `1` while Terminal 1 is running.

### Inspect the custom message definition

```bash
ros2 interface show hand_keypoint_interfaces/msg/HandKeypoints
ros2 interface show hand_keypoint_interfaces/msg/Hand
ros2 interface show hand_keypoint_interfaces/msg/PalmPose6D
```

### Receive one main result

```bash
ros2 topic echo \
  /surgery/perception/cam4/hand_keypoints \
  --once --no-daemon
```

Success means a typed message is received. `hands: []` is still a valid result;
it means no hand was detected in that particular frame.

### Receive the robot target pose

```bash
ros2 topic echo \
  /surgery/perception/cam4/hand_target_pose \
  --once --no-daemon
```

This command can wait when no hand has valid depth/palm pose. That is expected.

### Check health and diagnostics

```bash
ros2 topic echo \
  /surgery/perception/handkeypoint/health \
  std_msgs/msg/String \
  --once --no-daemon --full-length

ros2 topic echo \
  /surgery/perception/handkeypoint/diagnostics/json \
  std_msgs/msg/String \
  --once --no-daemon --full-length
```

Healthy operation should report lifecycle state `active`, status `ok`, a
recent frame, increasing `frames_processed`, and no increasing error count.

### View the overlay

```bash
ros2 run rqt_image_view rqt_image_view
```

Select:

```text
/surgery/images/cam4/hand_overlay/compressed
```

### Measure the output rate

```bash
ros2 topic hz \
  /surgery/perception/cam4/hand_keypoints \
  --window 100
```

Do not simultaneously measure the large raw RGB topic; that additional image
subscriber can perturb the replay test. The diagnostics JSON also provides the
internal `processed_hz_1s` field.

## Test success criteria

The output-interface test passes when:

1. All five `/surgery/...` topics are discovered.
2. `hand_keypoints` has the expected custom type and at least one publisher.
3. A `HandKeypoints` message can be received.
4. Health reports `active` and `ok`.
5. Diagnostics reports increasing frames with no increasing errors.
6. The overlay can be displayed when `publish_overlay:=true`.

Topic discovery alone confirms only that publishers exist. Receiving and
checking messages confirms that the interfaces actually carry usable data.
