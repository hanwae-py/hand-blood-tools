"""Tests for one-to-one RGB/native-depth timestamp pairing."""

from pnu_surgical_perception.native_depth_sync import (
    ApproximateRgbDepthPairer,
)

import pytest

from sensor_msgs.msg import CompressedImage


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
