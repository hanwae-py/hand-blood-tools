# Hand open/close anatomical rule

Portable, ROS-independent classification of a MediaPipe hand as `OPEN` or
`CLOSED` from its 21 two-dimensional landmarks.

This folder is an algorithm handoff bundle. The classifier and unit tests are
usable now; the ROS wrapper described in `open_close_rule.md` is a proposal and
has not yet been implemented or tested.

## Contents

- `anatomical_rule_classifier.py`: dependency-light runtime classifier.
- `tests/test_anatomical_rule_classifier.py`: synthetic topology tests.
- `open_close_rule.md`: rule definition, tested alternatives, limitations,
  offline evidence, and proposed ROS adaptation.
- `requirements.txt`: direct Python dependency.

## Input and output

Pass a NumPy array with shape `(21, 2)` in standard MediaPipe landmark order:

```python
from anatomical_rule_classifier import classify_anatomical_rules

result = classify_anatomical_rules(joints_2d)
print(result["state"])  # "OPEN" or "CLOSED"
```

Only 2D keypoints are used. The classifier does not run MediaPipe, read images,
calculate depth, or publish ROS messages.

## Verify the portable algorithm

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Before robot use, the receiving team must validate the rule on its own camera,
gloves, viewpoints, keypoint stream, and failure cases. The binary label is a
perception result; it should not directly command robot motion.

## ROS handoff

Read `open_close_rule.md` fully, especially section 11. Integrate the pure
classifier into the existing hand package or a downstream node, then build and
test it inside the destination ROS environment. Do not treat the example ROS
message and topic definitions as already verified.
