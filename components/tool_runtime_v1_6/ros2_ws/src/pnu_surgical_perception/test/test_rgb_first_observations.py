"""Smoke tests for RGB-first 2D observations and optional depth pairing."""

from types import SimpleNamespace

import numpy as np

from pnu_surgical_perception.native_depth_sync import ApproximateRgbDepthPairer
from pnu_surgical_perception.pose_message_mapping import (
    to_observation_array_from_detections,
)

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


def _detections():
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:30, 20:60] = True
    item = SimpleNamespace(
        frame_local_instance_id=0,
        canonical_class_id=1,
        model_class_index=0,
        class_name='Scalpel',
        class_confidence=0.9,
        bbox_xyxy_px=(20.0, 10.0, 60.0, 30.0),
        mask=mask,
    )
    return SimpleNamespace(
        image_width=80,
        image_height=40,
        model_version='smoke',
        ontology_version='smoke',
        instances=[item],
    )


def _stamped(stamp_ns):
    message = CompressedImage()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return message


def _queue_like_tool_node(pairer, rgb, require_depth):
    """Mirror NativeDepthPoseNode._receive_rgb branch selection."""
    pair = pairer.add_rgb(rgb)
    if pair is not None:
        return 'pair', pair
    if require_depth:
        return 'drop', None
    return 'rgb_only', None


def test_rgb_only_observations_keep_origin_and_skip_depth():
    observations = to_observation_array_from_detections(
        detections=_detections(),
        header=Header(),
        sequence=1,
        view='cam4',
        aligned_depth_m=None,
    )
    item = observations.instances[0]
    assert item.observation_point_valid
    assert item.observation_point_selection_mode == 'longitudinal_axis_midpoint'
    assert not item.observation_point_depth_valid
    assert item.observation_point_depth_m == 0.0


def test_matching_depth_fills_observation_point_depth_m():
    detections = _detections()
    origin_u, origin_v = 40, 20
    depth = np.zeros((40, 80), dtype=np.float32)
    depth[origin_v, origin_u] = 0.61
    observations = to_observation_array_from_detections(
        detections=detections,
        header=Header(),
        sequence=2,
        view='cam4',
        aligned_depth_m=depth,
    )
    item = observations.instances[0]
    assert item.observation_point_depth_valid
    assert abs(item.observation_point_depth_m - 0.61) < 1e-5


def test_unmatched_rgb_runs_when_depth_not_required():
    pairer = ApproximateRgbDepthPairer(maximum_delta_ns=1_000_000)
    kind, pair = _queue_like_tool_node(pairer, _stamped(1_000_000_000), False)
    assert kind == 'rgb_only'
    assert pair is None


def test_unmatched_rgb_is_dropped_when_depth_required():
    pairer = ApproximateRgbDepthPairer(maximum_delta_ns=1_000_000)
    kind, pair = _queue_like_tool_node(pairer, _stamped(1_000_000_000), True)
    assert kind == 'drop'
    assert pair is None


def test_fresh_depth_pairs_before_rgb_only_fallback():
    pairer = ApproximateRgbDepthPairer(maximum_delta_ns=1_000_000)
    depth = _stamped(1_000_000_000)
    rgb = _stamped(1_000_000_050)
    assert pairer.add_depth(depth) is None
    kind, pair = _queue_like_tool_node(pairer, rgb, False)
    assert kind == 'pair'
    assert pair is not None
    assert pair.rgb is rgb
    assert pair.depth is depth
