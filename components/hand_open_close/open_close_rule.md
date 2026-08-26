# Hand Open/Close Anatomical Rule

Date: 2026-08-26  
Project: Surgical tool handover perception  
Input representation: MediaPipe 21-point hand landmarks

## 1. Objective

The objective is to recognize two hand states from the surgical-room camera:

- `OPEN`: the surgeon presents an open hand, potentially requesting a tool.
- `CLOSED`: the surgeon closes the hand or grasps a tool presented by the
  robot.

The open/close result must be fast enough for a live ROS pipeline and stable
under the overhead camera view, gloves, hand rotation, nearby people, surgical
tools, and partial occlusion.

This document describes the current rule, compares it with the previously tested
methods, and explains why the current method is the best available method for
this project at this stage.

## 2. Hand landmark convention

The system uses MediaPipe's standard 21 hand landmarks:

| Finger | Landmark chain |
|---|---|
| Wrist | `0` |
| Thumb | `1, 2, 3, 4` |
| Index | `5, 6, 7, 8` |
| Middle | `9, 10, 11, 12` |
| Ring | `13, 14, 15, 16` |
| Pinky | `17, 18, 19, 20` |

The fingertips are `4, 8, 12, 16, 20`. The current classifier consumes the
2D pixel coordinates from `joints_2d`. Depth is not required for the binary
open/close decision.

## 3. Current method: anatomical ordering rules

The current method does not count how many points overlap and does not use a
generic pretrained gesture label. It evaluates whether the detected landmark
topology is anatomically consistent with an open hand.

The hand is `OPEN` only when both rules below pass. If either rule is violated,
the result is `CLOSED`.

### 3.1 Rule A: a finger cannot cross into a second-neighbor lane

A finger may touch, overlap, or enter the image-space lane of its immediate
neighbor. This allows normal open-hand poses in which adjacent fingers are
close together.

However, the finger's two distal landmarks—the DIP/IP and fingertip—must not
reach or cross the lane of a second-neighbor finger.

Examples:

- Thumb points `3,4` may enter or overlap the index lane `5–8`, but they must
  not reach the middle lane `9–12`.
- Index points `7,8` may enter the middle lane, but they must not reach the ring
  lane.
- Middle points `11,12` may enter the index or ring lane, but they must not
  reach the thumb or pinky lane.
- Ring and pinky use the same immediate-neighbor/second-neighbor principle.

The lateral direction is calculated from index MCP `5` toward pinky MCP `17`.
This creates a palm-local lateral axis rather than using the image's horizontal
axis. Consequently, rotating the hand in the image does not change the rule.

Finger lanes are represented by these anchors:

```text
Thumb MCP  = 2
Index MCP  = 5
Middle MCP = 9
Ring MCP   = 13
Pinky MCP  = 17
```

The default second-neighbor margin is `0.03 × palm width`. Palm width is the
distance between landmarks `5` and `17`.

### 3.2 Rule B: a fingertip cannot fold back to its second joint

For every finger, the fingertip must remain distally beyond its second joint.
The following folds are prohibited:

```text
Thumb:  tip 4  cannot fold to/past MCP 2
Index:  tip 8  cannot fold to/past PIP 6
Middle: tip 12 cannot fold to/past PIP 10
Ring:   tip 16 cannot fold to/past PIP 14
Pinky:  tip 20 cannot fold to/past PIP 18
```

For each finger, a distal axis is constructed from its base toward its
immediate joint. The tip and second joint are projected onto that axis. If the
tip is not sufficiently farther from the base than the second joint, that
finger is considered proximally folded.

The required distal gap is `0.05 × palm width` by default.

### 3.3 Final decision

```text
No second-neighbor crossing AND no proximal fold → OPEN
One or more anatomical violations                  → CLOSED
```

There is currently no `PARTIAL`, `UNKNOWN`, holding timer, or trigger state in
this classifier. It produces one binary state for every detected hand.

### 3.4 Overlay diagnostics

The result overlay uses:

- Green skeleton and label: `OPEN`
- Red skeleton and label: `CLOSED`
- Magenta diagnostic: distal point reached a second-neighbor lane
- Cyan diagnostic: fingertip folded back to its second joint
- `cross=N`: number of second-neighbor crossing violations
- `fold=N`: number of proximal-fold violations

These diagnostics are important because they show exactly why a hand was
classified as closed. The decision is inspectable rather than hidden inside a
model score.

## 4. Runtime efficiency

The anatomical decision was benchmarked using 379 saved hand observations.
The benchmark excludes MediaPipe, depth estimation, JSON loading, video
decoding, and overlay drawing.

| Measurement | Result |
|---|---:|
| Median time for 379 hands | 16.0373 ms |
| Median time per hand | 42.3149 microseconds |
| Equivalent throughput | 23,632 decisions/second |

At 15 FPS, one frame has a 66.67 ms time budget. A 0.042 ms decision consumes
approximately 0.06% of that budget for one hand. Therefore, the open/close rule
is easily suitable for real-time computation.

The complete GPU test pipeline—including MediaPipe and forced Depth Anything
V2—processed the latest clip at 28.11 FPS, faster than the 15 FPS source.
Depth Anything is not needed for this 2D classification and may be omitted in
the live open/close path. Real depth can still run when 3D hand position or palm
pose is required.

## 5. Methods previously tested

### 5.1 Manual finger-angle and fingertip-reach classifier

Method:

- Calculated PIP/DIP angles.
- Calculated fingertip reach relative to the palm.
- Voted across the fingers.
- Produced `OPEN`, `CLOSED`, or `UNKNOWN`.

Automated synthetic tests passed, but the real-video labels were noisy and not
usable. Surgical gloves, overhead viewpoint, foreshortening, tools, and
occlusion changed the 2D angles enough to break the fixed thresholds.

Decision: rejected.

### 5.2 Pretrained MediaPipe Gesture Recognizer

Method:

- Used Google's official pretrained `gesture_recognizer.task`.
- Read the canned `Open_Palm` and `Closed_Fist` classes.
- Initially required confidence `≥ 0.75` and a 500 ms stable hold.

On the first full recording, the pretrained model produced:

```text
Raw Open_Palm: 119
Raw Closed_Fist: 0
Open_Palm frames ≥ 0.75: 3
Stable events: 0
```

The generic pretrained model did not transfer well to the surgical domain. It
was not trained specifically for surgical gloves, the top-down camera,
instrument grasping, or the observed occlusions.

Decision: integration worked, but recognition quality was not acceptable.

### 5.3 Forced pretrained binary Open_Palm/Closed_Fist

Method:

- Restricted MediaPipe's pretrained gesture classifier to only `Open_Palm`
  and `Closed_Fist`.
- Removed the `None` rejection class.
- Forced every detected hand into one of the two classes.

This removed `None`, but it converted uncertainty into incorrect labels. Many
open palms were labeled closed, and many closed hands were labeled open. Some
forced decisions had scores near `0.01`, showing that the model had almost no
support for the selected class.

Decision: rejected. Removing an uncertainty class did not improve recognition.

### 5.4 Fingertip overlap-count classifier

Method evolution:

1. Initially required all five fingertips to overlap for `CLOSED`.
2. Then changed to any overlap meaning `CLOSED`.
3. Then allowed one or two overlaps and used three overlaps as the closed
   threshold.
4. Used a smaller distance threshold for thumb-index pairs.

This approach was fast and understandable, but the number of overlaps was not
an anatomically reliable state description. An open hand can naturally have
several nearby points due to perspective or grouped fingers. Conversely, a
closed hand does not always create the expected number of 2D overlaps.

Decision: superseded by point-specific anatomical rules.

## 6. Comparison

| Method | Main advantage | Main problem | Status |
|---|---|---|---|
| Finger angles/reach | Simple and fast | Fixed angles were noisy under surgical viewpoints | Rejected |
| Pretrained Gesture Recognizer | Ready-made learned model | Poor domain transfer to gloves and overhead video | Rejected for this data |
| Forced pretrained binary | Always returned open/closed | Forced low-confidence errors | Rejected |
| Overlap count | Explainable and fast | Count did not encode which anatomical relation was wrong | Superseded |
| Anatomical ordering rules | Fast, interpretable, point-specific, rotation/scale independent | Still depends on correct MediaPipe landmarks | Current best |

## 7. Why the anatomical method is the best option for now

### 7.1 It matches the observed hand structure

The method does not assume that nearby points automatically mean a fist.
Immediate neighboring fingers may overlap. A violation occurs only when a
finger travels anatomically too far across the hand or folds proximally beyond
the expected joint.

### 7.2 It is interpretable

Every closed result has a visible reason: a second-neighbor crossing or a
proximal fold. Thresholds and landmark indices are explicit. This makes errors
easier to diagnose and refine using the team's own footage.

### 7.3 It is independent of image rotation and scale

The classifier uses a palm-local coordinate system and divides margins by palm
width. A rotated hand, a different pixel resolution, or moderate changes in
camera distance should not require new pixel thresholds.

### 7.4 It allows normal adjacent-finger contact

Unlike the overlap-count method, the current method explicitly allows a finger
to touch or enter its immediate neighbor's lane. This reflects normal open-palm
poses and reduces false `CLOSED` decisions caused by tightly grouped fingers.

### 7.5 It is computationally negligible

At approximately 42 microseconds per hand, this logic adds effectively no
latency to the MediaPipe pipeline. Perception speed is determined by hand
landmark detection, not this decision rule.

### 7.6 It performs better qualitatively on current footage

Visual inspection of sampled overlays showed straight, ordered fingers staying
green while curled or anatomically crossed fingers turned red. This was more
consistent with visible hand posture than the earlier angle, pretrained, and
overlap-count experiments.

## 8. Test results with the current method

| Test clip | Frames | Hand observations | Open | Closed |
|---|---:|---:|---:|---:|
| `cam4_10m10s-10m50s.mp4` | 600 | 733 | 439 | 294 |
| `cam4_05m38s-06m00s.mp4` | 330 | 405 | 304 | 101 |
| `cam4_12m22s-12m43s.mp4` | 315 | 379 | 131 | 248 |

Seven focused anatomical tests pass. Across the preserved geometry and trigger
tests, 19 tests pass.

## 9. Current limitations

The current method is the best tested option for now, but it is not proof of
robot-trigger reliability.

- Incorrect MediaPipe landmarks will produce an incorrect state.
- Heavy occlusion can distort finger ordering.
- A closed hand viewed from an unusual angle may not show a proximal fold in
  2D.
- The clip counts are predictions, not accuracy measurements, because the
  frames do not yet have hand-labeled ground truth.
- A binary output cannot express true ambiguity.

Before connecting the result to robot motion, the team should label staged
open, closed, grasping, neutral, and occluded intervals from the deployment
camera. Evaluation should measure false triggers, missed triggers, and
detection latency at the event level.

## 10. Portable bundle and offline evidence

Implementation:

- `anatomical_rule_classifier.py`
- `tests/test_anatomical_rule_classifier.py`
- `requirements.txt`
- `README.md`

The bundle includes one approved 22-second clip, its extracted keypoints, and
the current generated results as a reproducible regression fixture under
`test_data/`. These generated results are not human-annotated accuracy ground
truth. Large datasets, generated overlays, cached models, and the experimental
worklog are intentionally excluded.

## 11. Proposed ROS 2 adaptation (not yet tested)

The pure keypoint classifier and its offline overlays have been tested. A ROS
node wrapping this classifier has **not** been implemented, built, or tested in
this repository yet. Everything in this section is an integration proposal for
the receiving ROS developer or Codex session and must be validated there.

The classifier deliberately has no ROS dependency. The receiving developer
should read this rule, inspect their current message and topic contracts, then
adapt the small wrapper to their environment rather than assuming the example
names below are already deployed.

### 11.1 Observed ROS contract at the time of handoff

At the time this bundle was prepared, the repository's ROS 2 Jazzy hand
pipeline defined the following required input:

```text
Topic: hand/keypoints
Type:  hand_keypoint_interfaces/msg/HandKeypoints
```

Each `HandKeypoints.hands[]` entry contains:

```text
int32 hand_index
bool has_handedness
string handedness_label
Point2D[21] joints_2d
```

`Point2D` contains pixel coordinates `u` and `v`. The indices follow the
standard MediaPipe ordering documented earlier in this report.

The classifier uses only `joints_2d`. It does not depend on `depth_source`,
`joints_3d`, `kp_valid_depth`, or `palm_6d`. Therefore, the same open/close node
works whether the upstream hand node uses real aligned depth, Depth Anything,
or no 3D consumer at all.

### 11.2 Proposed architecture

Add a small downstream ROS node instead of inserting the algorithm into the
MediaPipe callback:

```text
camera/depth
    ↓
existing hand_detection_node
    ↓  hand/keypoints (HandKeypoints)
new open_close_node
    ↓  hand/open_close (HandOpenCloseArray)
robot coordinator / logger / visualization
```

Reasons for a separate node:

- The existing hand detection interface remains backward compatible.
- Open/close rules can be changed without touching MediaPipe or depth code.
- The classifier can be tested with recorded `hand/keypoints` messages.
- The processing cost is only about 42 microseconds per hand.
- A failure in the new node does not stop keypoint and palm-pose publication.

If the deployment repository has a strict single-process requirement, the same
function can instead be called immediately after `process_frame()` returns
`row_hands`. The separate-node design is recommended unless that constraint
exists.

### 11.3 Add typed messages

Do not publish an unstructured JSON string for the robot-facing interface.
Add these messages to the existing `hand_keypoint_interfaces/msg/` package.

`HandOpenClose.msg`:

```text
uint8 UNKNOWN=0
uint8 OPEN=1
uint8 CLOSED=2

int32 hand_index
bool has_handedness
string handedness_label
float32 handedness_score

uint8 state
string state_label
uint16 lateral_crossing_count
uint16 proximal_fold_count
float32 palm_width_px
```

`HandOpenCloseArray.msg`:

```text
std_msgs/Header header
HandOpenClose[] hands
```

Add both files to `rosidl_generate_interfaces()` in
`hand_keypoint_interfaces/CMakeLists.txt`:

```cmake
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Point2D.msg"
  "msg/PalmPose6D.msg"
  "msg/Hand.msg"
  "msg/HandKeypoints.msg"
  "msg/HandOpenClose.msg"
  "msg/HandOpenCloseArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)
```

No new interface dependency is required because `std_msgs` is already present.

### 11.4 Copy the classifier into the ROS Python package

Copy this bundle's:

```text
components/hand_open_close/anatomical_rule_classifier.py
```

to:

```text
ros2_ws/src/hand_keypoint_ros/hand_keypoint_ros/anatomical_rule_classifier.py
```

The classifier has one runtime dependency: NumPy. It has no OpenCV, MediaPipe,
Torch, depth, ROS, or model-file dependency.

Ensure the runtime package declares NumPy. For an Ubuntu/ROS installation, add
this to `hand_keypoint_ros/package.xml` if it is not already declared:

```xml
<exec_depend>python3-numpy</exec_depend>
```

### 11.5 Implement `open_close_node.py`

Create:

```text
ros2_ws/src/hand_keypoint_ros/hand_keypoint_ros/open_close_node.py
```

Reference implementation:

```python
import numpy as np
import rclpy
from rclpy.node import Node

from hand_keypoint_interfaces.msg import (
    HandKeypoints,
    HandOpenClose,
    HandOpenCloseArray,
)
from hand_keypoint_ros.anatomical_rule_classifier import (
    AnatomicalRuleConfig,
    classify_anatomical_rules,
)


class OpenCloseNode(Node):
    def __init__(self):
        super().__init__('open_close_node')
        self.declare_parameter('input_topic', 'hand/keypoints')
        self.declare_parameter('output_topic', 'hand/open_close')
        self.declare_parameter('second_neighbor_margin_ratio', 0.03)
        self.declare_parameter('minimum_tip_beyond_second_ratio', 0.05)

        get = self.get_parameter
        self.config = AnatomicalRuleConfig(
            second_neighbor_margin_ratio=float(
                get('second_neighbor_margin_ratio').value),
            minimum_tip_beyond_second_ratio=float(
                get('minimum_tip_beyond_second_ratio').value),
        )
        self.publisher = self.create_publisher(
            HandOpenCloseArray, get('output_topic').value, 10)
        self.subscription = self.create_subscription(
            HandKeypoints, get('input_topic').value, self._on_keypoints, 10)

    def _on_keypoints(self, msg):
        output = HandOpenCloseArray()
        output.header = msg.header

        for hand in msg.hands:
            if len(hand.joints_2d) != 21:
                self.get_logger().warn(
                    f'hand {hand.hand_index}: expected 21 joints_2d, '
                    f'got {len(hand.joints_2d)}')
                continue

            points = np.asarray(
                [[point.u, point.v] for point in hand.joints_2d],
                dtype=np.float64,
            )
            result = classify_anatomical_rules(points, self.config)

            state = HandOpenClose()
            state.hand_index = hand.hand_index
            state.has_handedness = hand.has_handedness
            state.handedness_label = hand.handedness_label
            state.handedness_score = hand.handedness_score
            state.state_label = result['state']
            state.state = (
                HandOpenClose.OPEN
                if result['state'] == 'OPEN'
                else HandOpenClose.CLOSED
            )
            state.lateral_crossing_count = len(result['lateral_crossings'])
            state.proximal_fold_count = len(result['proximal_folds'])
            state.palm_width_px = result['palm_width_px']
            output.hands.append(state)

        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = OpenCloseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

The callback publishes an empty `hands` array when no hand is detected. It
preserves the upstream header, allowing consumers to correlate the result with
the exact camera frame and keypoint message.

### 11.6 Register the executable

Add the following entry to `hand_keypoint_ros/setup.py`:

```python
entry_points={
    'console_scripts': [
        'hand_detection_node = hand_keypoint_ros.hand_detection_node:main',
        'fake_camera_publisher = hand_keypoint_ros.fake_camera_publisher:main',
        'open_close_node = hand_keypoint_ros.open_close_node:main',
    ],
},
```

### 11.7 Build and run

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install \
  --packages-select hand_keypoint_interfaces hand_keypoint_ros
source install/setup.bash

ros2 run hand_keypoint_ros open_close_node --ros-args \
  -p input_topic:=hand/keypoints \
  -p output_topic:=hand/open_close
```

If the existing system places the hand node in a namespace, use fully resolved
topic names or launch-time remappings rather than editing source code.

Inspect the result:

```bash
ros2 topic info /hand/open_close --verbose
ros2 topic hz /hand/open_close
ros2 topic echo /hand/open_close
```

Expected behavior:

- Output rate follows `hand/keypoints`.
- Zero detected hands produce an empty array, not a stale state.
- Every valid input hand produces exactly one `OPEN` or `CLOSED` item.
- `header.stamp` and `header.frame_id` match the input message.
- The node does not subscribe to RGB or depth and does not load an ML model.

### 11.8 Testing requirements

Copy the portable unit test into the ROS package's test directory:

```text
components/hand_open_close/tests/test_anatomical_rule_classifier.py
→ ros2_ws/src/hand_keypoint_ros/test/test_anatomical_rule_classifier.py
```

Adjust only the import path after copying:

```python
from hand_keypoint_ros.anatomical_rule_classifier import (
    classify_anatomical_rules,
)
```

Run:

```bash
colcon test --packages-select hand_keypoint_ros
colcon test-result --verbose
```

Also perform a ROS smoke test with a recorded camera/bag or the existing fake
camera publisher. Verify message rate, header preservation, hand count, and
absence of exceptions for zero, one, and multiple hands.

### 11.9 Robot integration boundary

`hand/open_close` should be treated as a perception result, not a direct motor
command. The robot coordinator should decide when an `OPEN` or `CLOSED` state is
relevant based on its task state and selected hand. For example, `CLOSED` may
only be meaningful while the robot is already in `HANDOVER`.

Do not remove the existing `hand/keypoints`, `hand/target_pose`, health, or
diagnostic outputs. The new topic is additive.

## 12. Files to share with the ROS team

### Required

Share the complete `components/hand_open_close` directory. Its essential files
are:

1. `open_close_rule.md` — algorithm explanation, evidence, and complete
   ROS integration instructions.
2. `anatomical_rule_classifier.py` — the runtime algorithm.
3. `tests/test_anatomical_rule_classifier.py` — self-contained portable unit
   tests.
4. `README.md` and `requirements.txt` — usage and dependency information.

Large model caches, Python environments, Depth Anything weights, and generated
JSON/MP4/GIF files are not needed to integrate this classifier. The open/close
runtime file needs only NumPy and 21 two-dimensional landmarks.

## 13. Ready-to-paste task for Codex in the ROS repository

Give the receiving Codex session the required files and this instruction:

```text
Integrate the supplied anatomical hand open/close classifier into this ROS 2
Jazzy repository. Read open_close_rule.md completely before editing.

Use the existing hand_keypoint_interfaces/msg/HandKeypoints topic as input and
consume each hand's 21 joints_2d points. Add new typed HandOpenClose and
HandOpenCloseArray messages without changing the existing Hand or HandKeypoints
messages. Add a separate hand_keypoint_ros/open_close_node.py that publishes
hand/open_close, preserves the input header, handles zero/multiple hands, and
exposes the two classifier margins as ROS parameters. Register the executable,
declare NumPy as a runtime dependency, update rosidl_generate_interfaces, copy
and adapt the supplied unit test, build with colcon, run tests, and perform a
topic-level smoke test. Do not add depth or MediaPipe inference to the new node;
the classifier uses only the keypoints already published by the existing hand
node. Keep robot actuation outside this perception node.
```
