"""Smoke tests for Blood optional centroid depth sampling."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CompressedImage

from surgical_task_coordinator.blood_detection_node import (
    BloodDetectionNode,
    mask_centroid,
    sample_centroid_depth_m,
    stamp_ns,
)


def test_mask_centroid_and_depth_sample():
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    centroid = mask_centroid(mask)
    assert centroid is not None
    np.testing.assert_allclose(centroid, [9.5, 9.5], atol=0.6)

    depth = np.zeros((20, 20), dtype=np.float32)
    u, v = int(round(centroid[0])), int(round(centroid[1]))
    depth[v, u] = 0.55
    assert sample_centroid_depth_m(depth, centroid) == pytest.approx(0.55)
    assert sample_centroid_depth_m(depth, None) is None
    assert sample_centroid_depth_m(np.zeros((20, 20), dtype=np.float32), centroid) is None


def test_stamp_window_and_shape_mismatch_skip_depth():
    rgb = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))
    )
    depth = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2_000_000))
    )
    assert abs(stamp_ns(rgb) - stamp_ns(depth)) > 1_000_000

    matched = np.zeros((10, 10), dtype=np.float32)
    mismatched = np.zeros((8, 10), dtype=np.float32)
    centroid = [5.0, 5.0]
    assert mismatched.shape != (10, 10)
    assert sample_centroid_depth_m(matched, centroid) is None


def _compressed_depth(height: int, width: int, millimetres: int, nsec: int = 0) -> CompressedImage:
    native = np.full((height, width), millimetres, dtype=np.uint16)
    success, encoded = cv2.imencode('.png', native)
    assert success
    message = CompressedImage()
    message.format = '16UC1; compressedDepth png'
    message.data = b'hdr' + encoded.tobytes()
    message.header.stamp.nanosec = nsec
    return message


def test_blood_node_samples_matching_depth_and_skips_otherwise():
    rclpy.init()
    node = BloodDetectionNode()
    try:
        assert bool(node.get_parameter('require_depth').value) is False
        rgb = CompressedImage()
        assert node._aligned_depth_m(rgb, 8, 12) is None

        node._on_depth(_compressed_depth(8, 12, 1500))
        depth_m = node._aligned_depth_m(rgb, 8, 12)
        assert depth_m is not None
        assert depth_m.shape == (8, 12)
        assert float(depth_m[3, 4]) == pytest.approx(1.5)

        assert node._aligned_depth_m(rgb, 7, 12) is None

        stale = CompressedImage()
        stale.header.stamp.nanosec = 2_000_000
        assert node._aligned_depth_m(stale, 8, 12) is None
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
