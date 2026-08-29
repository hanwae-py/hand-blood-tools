import numpy as np
import pytest
import rclpy
from rclpy.context import Context

from pnu_surgical_perception.operator_quad_compositor import (
    OperatorQuadCompositor,
    letterbox,
    low_light_metrics,
    placeholder,
)


def test_operator_quad_defaults_to_cam1_auxiliary_hand_panel():
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = OperatorQuadCompositor(context=context)
        assert node.get_parameter(
            'cam1_hand_overlay_topic').value == (
                '/perception/cam_1/hand/overlay/compressed')
        assert 'cam1' in node._images
        assert 'cam2' not in node._images
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_letterbox_and_placeholder_have_exact_panel_shape():
    source = np.full((1496, 2048, 3), 7, dtype=np.uint8)
    assert letterbox(source, 960, 540).shape == (540, 960, 3)
    assert placeholder(960, 540, 'MISSING').shape == (540, 960, 3)


def test_low_light_metrics_separate_dark_and_usable_frames():
    dark = np.full((80, 120, 3), 3, dtype=np.uint8)
    gradient = np.tile(
        np.linspace(0, 220, 120, dtype=np.uint8), (80, 1))
    bright = np.repeat(gradient[:, :, None], 3, axis=2)
    dark_p99, dark_range = low_light_metrics(dark)
    bright_p99, bright_range = low_light_metrics(bright)
    assert dark_p99 < 20.0
    assert dark_range < 12.0
    assert bright_p99 > 20.0
    assert bright_range > 12.0


def test_letterbox_rejects_non_bgr_input():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10), dtype=np.uint8), 100, 100)
