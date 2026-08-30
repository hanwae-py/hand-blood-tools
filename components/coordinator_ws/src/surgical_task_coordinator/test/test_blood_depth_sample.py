"""Smoke tests for Blood optional centroid depth sampling."""

import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage

from surgical_task_coordinator.blood_detection_node import (
    BloodDetectionNode,
    encode_coco_rle,
    image_quality_metrics,
    mask_centroid,
    sample_centroid_depth_m,
    stamp_ns,
)


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _drain_until(node, predicate, timeout_sec=3.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        node._drain_completed_frame()
        if predicate():
            return
        time.sleep(0.005)
    node._drain_completed_frame()
    assert predicate(), 'worker result was not available before timeout'


def _slow_coco_rle_reference(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    flat = binary.reshape(-1, order='F')
    counts = []
    previous = 0
    run_length = 0
    for pixel in flat:
        current = int(pixel != 0)
        if current == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = current
    counts.append(run_length)
    return {'size': [int(binary.shape[0]), int(binary.shape[1])], 'counts': counts}


def _jpeg(value_or_image) -> CompressedImage:
    image = (
        np.full((80, 120, 3), int(value_or_image), dtype=np.uint8)
        if np.isscalar(value_or_image) else np.asarray(value_or_image, dtype=np.uint8)
    )
    success, encoded = cv2.imencode('.jpg', image)
    assert success
    message = CompressedImage()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 34
    message.header.frame_id = 'flir_color_optical_frame'
    message.format = 'jpeg'
    message.data = encoded.tobytes()
    return message


def test_flir_image_quality_metrics_fail_dark_and_pass_gradient():
    dark = np.full((40, 60, 3), 3, dtype=np.uint8)
    gradient = np.tile(
        np.linspace(0, 220, 60, dtype=np.uint8), (40, 1))
    bright = np.repeat(gradient[:, :, None], 3, axis=2)
    dark_quality = image_quality_metrics(dark)
    bright_quality = image_quality_metrics(bright)
    assert dark_quality['gray_p99'] < 20.0
    assert dark_quality['gray_dynamic_range'] < 12.0
    assert bright_quality['gray_p99'] > 20.0
    assert bright_quality['gray_dynamic_range'] > 12.0
    with pytest.raises(ValueError):
        image_quality_metrics(np.zeros((10, 10), dtype=np.uint8))


@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (19, 23)])
def test_vectorized_coco_rle_matches_existing_contract(shape):
    rng = np.random.default_rng(sum(shape) + 11)
    masks = [
        np.zeros(shape, dtype=np.uint8),
        np.ones(shape, dtype=np.uint8),
        rng.integers(0, 4, size=shape, dtype=np.uint8),
    ]
    for mask in masks:
        encoded = encode_coco_rle(mask)
        assert encoded == _slow_coco_rle_reference(mask)
        assert all(type(count) is int for count in encoded['counts'])
        assert all(type(count) is int for count in json.loads(json.dumps(encoded))['counts'])


def test_vectorized_coco_rle_is_column_major_and_preserves_leading_one_run():
    mask = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    encoded = encode_coco_rle(mask)
    assert encoded == _slow_coco_rle_reference(mask)
    assert encoded['counts'] == [0, 2, 1, 2, 1]


def _empty_frame_output(image):
    height, width = image.shape[:2]
    return SimpleNamespace(
        mask=np.zeros((height, width), dtype=bool),
        centroids=[],
        action='wait_detection',
        ran_detector=True,
        detector_ms=1.0,
        tracker_ms=0.0,
        total_ms=1.0,
    )


def test_dark_flir_frame_is_unknown_without_model_or_mask_call():
    class Pipeline:
        calls = 0

        def step(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError('dark input must not reach BloodPipeline')

    rclpy.init()
    node = BloodDetectionNode()
    try:
        node.set_parameters([
            Parameter('reject_low_quality_input', Parameter.Type.BOOL, True),
            Parameter('camera', Parameter.Type.STRING, 'flir'),
        ])
        node._active = True
        node._pipeline = Pipeline()
        node._mask_pub = _Publisher()
        node._overlay_pub = _Publisher()
        node._semantics_pub = _Publisher()
        node._on_color(_jpeg(3))
        _drain_until(node, lambda: bool(node._semantics_pub.messages))
        assert node._pipeline.calls == 0
        assert node._mask_pub.messages == []
        assert len(node._overlay_pub.messages) == 1
        payload = json.loads(node._semantics_pub.messages[0].data)
        assert payload['observation_valid'] is False
        assert payload['mask_published'] is False
        assert payload['camera'] == 'flir'
        assert payload['header'] == {
            'stamp_sec': 12, 'stamp_nanosec': 34,
            'frame_id': 'flir_color_optical_frame'}
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_bright_flir_frame_runs_model_and_publishes_valid_empty_mask():
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    class Pipeline:
        calls = 0

        def step(self, image, **_kwargs):
            self.calls += 1
            return _empty_frame_output(image)

    gradient = np.tile(
        np.linspace(0, 220, 120, dtype=np.uint8), (80, 1))
    bright = np.repeat(gradient[:, :, None], 3, axis=2)
    rclpy.init()
    node = BloodDetectionNode()
    try:
        node.set_parameters([
            Parameter('reject_low_quality_input', Parameter.Type.BOOL, True),
            Parameter('camera', Parameter.Type.STRING, 'flir'),
        ])
        node._active = True
        node._pipeline = Pipeline()
        node._torch = Torch()
        node._mask_pub = _Publisher()
        node._overlay_pub = _Publisher()
        node._semantics_pub = _Publisher()
        node._on_color(_jpeg(bright))
        _drain_until(node, lambda: bool(node._semantics_pub.messages))
        assert node._pipeline.calls == 1
        assert len(node._mask_pub.messages) == 1
        payload = json.loads(node._semantics_pub.messages[0].data)
        assert payload['observation_valid'] is True
        assert payload['mask_published'] is True
        assert payload['depth_sampled'] is False
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_rgb_only_mode_never_touches_depth_path():
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    class Pipeline:
        def step(self, image, **_kwargs):
            return _empty_frame_output(image)

    rclpy.init()
    node = BloodDetectionNode()
    try:
        node._active = True
        node._pipeline = Pipeline()
        node._torch = Torch()
        node._mask_pub = _Publisher()
        node._overlay_pub = _Publisher()
        node._semantics_pub = _Publisher()
        node._aligned_depth_m = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('RGB-only mode must not call _aligned_depth_m')
        )
        node._on_depth(_compressed_depth(8, 12, 1500))
        node._on_color(_jpeg(180))
        _drain_until(node, lambda: bool(node._semantics_pub.messages))
        payload = json.loads(node._semantics_pub.messages[-1].data)
        assert payload['depth_sampled'] is False
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_latest_frame_worker_replaces_backlog_without_parallel_model_calls():
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    started = threading.Event()
    release = threading.Event()

    class Pipeline:
        calls = 0

        def step(self, image, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                started.set()
                assert release.wait(timeout=2.0)
            return _empty_frame_output(image)

    rclpy.init()
    node = BloodDetectionNode()
    try:
        node._active = True
        node._pipeline = Pipeline()
        node._torch = Torch()
        node._mask_pub = _Publisher()
        node._overlay_pub = _Publisher()
        node._semantics_pub = _Publisher()
        node._on_color(_jpeg(80))
        assert started.wait(timeout=2.0)
        for value in (100, 120, 140, 160):
            node._on_color(_jpeg(value))
        release.set()
        _drain_until(node, lambda: node._pipeline.calls >= 2)
        assert node._pipeline.calls == 2
        assert node._frames_dropped_latest >= 3
        assert node._worker_busy is False or node._pending_job is None
    finally:
        release.set()
        node.destroy_node()
        rclpy.try_shutdown()


def test_blood_health_requires_active_fresh_successful_observation():
    rclpy.init()
    node = BloodDetectionNode()
    try:
        node._health_pub = _Publisher()
        node._diagnostics_pub = _Publisher()
        node._active = True
        node._pipeline = object()
        node._image_quality_ready = True
        node._last_observation_valid = True
        node._last_input_at = time.monotonic()
        node._publish_status()
        assert json.loads(node._health_pub.messages[-1].data)['ready'] is True

        node._last_input_at = time.monotonic() - 5.0
        node._publish_status()
        stale = json.loads(node._health_pub.messages[-1].data)
        assert stale['ready'] is False
        assert stale['input_fresh'] is False
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


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
