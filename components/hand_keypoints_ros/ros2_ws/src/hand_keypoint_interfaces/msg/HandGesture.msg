# One detected hand's classifier result derived from MediaPipe landmarks.
# hand_index is MediaPipe's frame-local result-array index, not a persistent
# track ID. If a HandKeypoints message with the same Header contains this
# index, both observations refer to the same MediaPipe hand. Gesture output is
# RGB-only, so it can exist without a depth-gated HandKeypoints counterpart.

int32 hand_index

bool has_handedness
string handedness_label
float32 handedness_score

# category_name is one of the supported gestures listed by the parent
# HandGestureArray, or the rejection category "None". It is valid only when
# has_classification is true. ``None`` is a valid, fail-closed classification.
bool has_classification
string category_name
float32 score
