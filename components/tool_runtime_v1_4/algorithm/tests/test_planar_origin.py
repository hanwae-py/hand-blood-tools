"""Smoke tests for RGB-first 2D origin, optional depth sample, and pose."""

from __future__ import annotations

import numpy as np
import pytest

from pnu_surgical_tool.depth_registration import (
    decode_compressed_depth_16uc1,
    metric_depth_in_rgb_frame,
    registrar_from_camera_fields,
)
from pnu_surgical_tool.planar_pose import (
    PlanarPoseEstimator,
    longitudinal_origin_uv,
    sample_depth_at_uv,
)
from pnu_surgical_tool.types import (
    CameraCalibration,
    DetectionBatch,
    DetectionInstance,
    SupportPlane,
)


def _rectangle_mask() -> np.ndarray:
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:30, 20:60] = True
    return mask


def test_longitudinal_origin_uv_is_axis_midpoint() -> None:
    origin = longitudinal_origin_uv(_rectangle_mask(), "Scalpel")
    assert origin is not None
    np.testing.assert_allclose(origin, [39.5, 19.5], atol=1.5)


def test_sample_depth_at_uv_skips_invalid() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 1] = 0.42
    assert sample_depth_at_uv(depth, np.array([1.0, 2.0])) == pytest.approx(0.42)
    assert sample_depth_at_uv(depth, np.array([0.0, 0.0])) is None
    assert sample_depth_at_uv(depth, np.array([-1.0, 0.0])) is None


def test_rgb_only_skips_depth_and_keeps_origin_uv() -> None:
    origin = longitudinal_origin_uv(_rectangle_mask(), "Scalpel")
    assert origin is not None
    assert sample_depth_at_uv(None, origin) is None  # type: ignore[arg-type]


def test_matching_depth_is_sampled_at_origin_uv() -> None:
    mask = _rectangle_mask()
    origin = longitudinal_origin_uv(mask, "Scalpel")
    assert origin is not None
    depth = np.zeros(mask.shape, dtype=np.float32)
    u, v = int(round(float(origin[0]))), int(round(float(origin[1])))
    depth[v, u] = 0.73
    assert sample_depth_at_uv(depth, origin) == pytest.approx(0.73)


def test_pose_with_depth_keeps_metric_p_obs() -> None:
    mask = _rectangle_mask()
    height, width = mask.shape
    instance = DetectionInstance(
        frame_local_instance_id=0,
        canonical_class_id=1,
        model_class_index=0,
        class_name="Scalpel",
        class_confidence=0.9,
        bbox_xyxy_px=(20.0, 10.0, 60.0, 30.0),
        mask=mask,
    )
    detections = DetectionBatch(
        image_width=width,
        image_height=height,
        model_version="smoke",
        ontology_version="smoke",
        instances=[instance],
    )
    depth = np.full((height, width), 0.8, dtype=np.float32)
    camera = CameraCalibration(
        width=width,
        height=height,
        k=np.array([[50.0, 0.0, 40.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]]),
        distortion=np.zeros(5),
        frame_name="cam4",
        calibration_version="smoke",
    )
    plane = SupportPlane(
        normal=np.array([0.0, 0.0, 1.0]),
        offset_m=-0.8,
        config_version="smoke-plane",
    )
    result = PlanarPoseEstimator().estimate(detections, depth, camera, plane)
    item = result.instances[0]
    assert item.position_valid
    assert item.observation_point_uv_px is not None
    assert item.observation_point_depth_m == pytest.approx(0.8)
    assert item.position_m is not None
    assert item.position_m[2] == pytest.approx(0.8, abs=0.05)


def test_decode_compressed_depth_png_payload() -> None:
    native = np.full((8, 12), 1500, dtype=np.uint16)
    native[0, 0] = 0
    success, encoded = __import__("cv2").imencode(".png", native)
    assert success
    payload = b"header12" + encoded.tobytes()
    decoded = decode_compressed_depth_16uc1(
        payload, "16UC1; compressedDepth png"
    )
    assert decoded.dtype == np.uint16
    assert decoded.shape == (8, 12)
    assert int(decoded[1, 1]) == 1500
    assert int(decoded[0, 0]) == 0


def test_metric_depth_same_shape_scales_without_registration() -> None:
    native = np.full((4, 5), 2000, dtype=np.uint16)
    native[0, 0] = 0
    depth = metric_depth_in_rgb_frame(native, 4, 5, 0.001)
    assert depth is not None
    assert depth.shape == (4, 5)
    assert float(depth[1, 1]) == pytest.approx(2.0)
    assert float(depth[0, 0]) == 0.0
    assert metric_depth_in_rgb_frame(native, 8, 10, 0.001) is None


def test_metric_depth_registers_native_into_rgb_frame() -> None:
    registrar = registrar_from_camera_fields(
        color_width=10,
        color_height=8,
        color_k=[[100.0, 0.0, 5.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]],
        color_d=[],
        color_frame="color",
        depth_width=5,
        depth_height=4,
        depth_k=[[50.0, 0.0, 2.5], [0.0, 50.0, 2.0], [0.0, 0.0, 1.0]],
        depth_d=[],
        depth_frame="depth",
        rotation=np.eye(3),
        translation_m=[0.0, 0.0, 0.0],
        calibration_version="test",
    )
    native = np.full((4, 5), 1500, dtype=np.uint16)
    aligned = metric_depth_in_rgb_frame(native, 8, 10, 0.001, registrar)
    assert aligned is not None
    assert aligned.shape == (8, 10)
    finite = aligned[np.isfinite(aligned)]
    assert finite.size > 0
    assert float(np.nanmedian(finite)) == pytest.approx(1.5, abs=0.05)


def test_package_import_does_not_load_detector() -> None:
    import sys

    assert "pnu_surgical_tool.rfdetr_inference" not in sys.modules
    assert "pnu_surgical_tool.api" not in sys.modules
