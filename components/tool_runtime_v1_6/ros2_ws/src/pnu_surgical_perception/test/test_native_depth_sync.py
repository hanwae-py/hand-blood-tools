"""Tests for one-to-one RGB/native-depth timestamp pairing and lifecycle."""

from pnu_surgical_perception.native_depth_sync import (
    ApproximateRgbDepthPairer,
)

import pytest

import rclpy
from rclpy.context import Context
from sensor_msgs.msg import CompressedImage

from pnu_surgical_perception.native_depth_pose_node import NativeDepthPoseNode


def message_at(stamp_ns):
    """Build a minimal stamped compressed-image message."""
    message = CompressedImage()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    return message


def test_pairer_matches_reference_bag_scale_delta_once():
    """Match the approximately 63 microsecond delta measured in the bag."""
    pairer = ApproximateRgbDepthPairer(maximum_delta_ns=1_000_000)
    rgb = message_at(10_000_000_000)
    depth = message_at(10_000_062_988)

    assert pairer.add_rgb(rgb) is None
    pair = pairer.add_depth(depth)

    assert pair.rgb is rgb
    assert pair.depth is depth
    assert pair.delta_ns == 62_988
    assert pairer.queued_rgb == 0
    assert pairer.queued_depth == 0


def test_pairer_does_not_match_outside_tolerance():
    """Keep messages unmatched when their timestamps exceed tolerance."""
    pairer = ApproximateRgbDepthPairer(maximum_delta_ns=1_000_000)

    assert pairer.add_rgb(message_at(1_000_000_000)) is None
    assert pairer.add_depth(message_at(1_002_000_000)) is None
    assert pairer.queued_rgb == 1
    assert pairer.queued_depth == 1


def test_pairer_rejects_invalid_configuration():
    """Reject a negative matching tolerance."""
    with pytest.raises(ValueError, match='non-negative'):
        ApproximateRgbDepthPairer(maximum_delta_ns=-1)


def test_native_depth_node_lifecycle_keeps_rclpy_collections_intact(monkeypatch):
    """Construction/destruction guards against Node private-name collisions."""
    # The lifecycle regression does not need a model or calibrated hardware;
    # supply the minimum already-validated transport fields so constructor
    # reaches real publisher/subscription creation without inference startup.
    def lifecycle_parameters(node):
        node._rgb_topic = '/test/cam_3/color'
        node._color_info_topic = '/test/cam_3/color_info'
        node._depth_topic = '/test/cam_3/depth'
        node._depth_info_topic = '/test/cam_3/depth_info'
        node._extrinsics_topic = '/test/cam_3/extrinsics'
        node._processing_gate_topic = ''
        node._pose_topic = '/test/cam_3/poses'
        node._observation_topic = '/test/cam_3/observations'
        node._overlay_topic = '/test/cam_3/overlay'
        node._pose_overlay_topic = '/test/cam_3/pose_overlay'
        node._diagnostics_topic = '/test/cam_3/diagnostics'
        node._health_topic = '/test/cam_3/health'
        node._maximum_stamp_delta_ns = 1_000_000
        node._sync_queue_size = 1
        node._require_depth = True
        node._workspace_zone = 'test'
        node._publish_tool_tf = False

    monkeypatch.setattr(
        NativeDepthPoseNode, '_read_parameters', lifecycle_parameters
    )
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = NativeDepthPoseNode(context=context)
        assert node._context is context
        assert len(node._perception_subscriptions) >= 4
        assert isinstance(node._subscriptions, list)
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        rclpy.shutdown(context=context)
