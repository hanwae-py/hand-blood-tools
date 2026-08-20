"""Smoke tests for Blood optional centroid depth sampling."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage

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


def _camera_info(
    width: int, height: int, frame: str, fx: float, fy: float, cx: float, cy: float
) -> CameraInfo:
    message = CameraInfo()
    message.width = width
    message.height = height
    message.header.frame_id = frame
    message.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    message.d = []
    return message


def test_blood_registers_native_depth_into_rgb_frame():
    rclpy.init()
    node = BloodDetectionNode()
    try:
        node.set_parameters(
            [
                Parameter(
                    "depth_to_color_rotation",
                    Parameter.Type.DOUBLE_ARRAY,
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                ),
                Parameter(
                    "depth_to_color_translation_m",
                    Parameter.Type.DOUBLE_ARRAY,
                    [0.0, 0.0, 0.0],
                ),
                Parameter("calibration_version", Parameter.Type.STRING, "test"),
            ]
        )
        node._on_color_info(_camera_info(12, 8, "color", 100.0, 100.0, 6.0, 4.0))
        node._on_depth_info(_camera_info(6, 4, "depth", 50.0, 50.0, 3.0, 2.0))
        node._on_depth(_compressed_depth(4, 6, 1500))
        aligned = node._aligned_depth_m(CompressedImage(), 8, 12)
        assert aligned is not None
        assert aligned.shape == (8, 12)
        finite = aligned[np.isfinite(aligned) & (aligned > 0)]
        assert finite.size > 0
        assert float(np.nanmedian(finite)) == pytest.approx(1.5, abs=0.05)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
