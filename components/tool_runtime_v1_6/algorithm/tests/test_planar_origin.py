"""Smoke tests for RGB-first 2D origin, optional depth sample, and pose."""

from __future__ import annotations

import numpy as np
import pytest

from pnu_surgical_tool.depth_registration import (
    decode_compressed_depth_16uc1,
    metric_depth_in_rgb_frame,
    registrar_from_camera_fields,
    rigid_transform_from_realsense_extrinsics,
)
from pnu_surgical_tool.planar_pose import (
    _pca_endpoints,
    _sign_policy,
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


# Non-identifying RF-DETR mask contours from the representative CAM4 frames.
# The polygons retain the endpoint morphology without storing patient imagery.
_BIPOLAR_CASE_FIXTURES = {
    "0704_6_60s_curved": {
        "contour": [
            [0, 2], [8, 11], [60, 22], [119, 18], [182, 29],
            [206, 29], [215, 25], [174, 12], [135, 9], [114, 5],
            [95, 5], [80, 10], [66, 11], [14, 0],
        ],
        "tip_uv": [0, 2],
        "handle_uv": [215, 25],
    },
    "0704_9_55s_straight": {
        "contour": [
            [216, 0], [192, 0], [175, 3], [129, 18], [104, 21],
            [58, 33], [25, 38], [4, 45], [0, 53], [11, 59],
            [74, 45], [140, 23], [159, 22], [189, 15], [207, 8],
        ],
        "tip_uv": [0, 53],
        "handle_uv": [216, 0],
    },
}


_ADSON_CASE_FIXTURES = {
    "0704_6_72s_placed": {
        "contour": [
            [0, 9], [2, 16], [10, 21], [63, 19], [115, 11],
            [117, 3], [16, 0], [2, 3],
        ],
        "tip_uv": [117, 3],
        "handle_uv": [0, 9],
    },
    "0704_9_70s_straight": {
        "contour": [
            [123, 5], [80, 0], [7, 6], [0, 13],
            [6, 18], [19, 19], [75, 18], [112, 12],
        ],
        "tip_uv": [123, 5],
        "handle_uv": [0, 13],
    },
}


def _rotate_uv_90_ccw(uv: np.ndarray, old_width: int) -> np.ndarray:
    return np.array((uv[1], old_width - 1 - uv[0]), dtype=np.float64)


@pytest.mark.parametrize("fixture_name", _BIPOLAR_CASE_FIXTURES)
@pytest.mark.parametrize("quarter_turns", range(4))
def test_bipolar_working_endpoint_points_to_tip_for_case_shapes(
    fixture_name: str,
    quarter_turns: int,
) -> None:
    fixture = _BIPOLAR_CASE_FIXTURES[fixture_name]
    contour = np.asarray(fixture["contour"], dtype=np.int32)
    height = int(contour[:, 1].max()) + 1
    width = int(contour[:, 0].max()) + 1
    mask = np.zeros((height, width), dtype=np.uint8)
    __import__("cv2").fillPoly(mask, [contour], 1)
    tip_uv = np.asarray(fixture["tip_uv"], dtype=np.float64)
    handle_uv = np.asarray(fixture["handle_uv"], dtype=np.float64)
    for _ in range(quarter_turns):
        old_width = mask.shape[1]
        mask = np.rot90(mask)
        tip_uv = _rotate_uv_90_ccw(tip_uv, old_width)
        handle_uv = _rotate_uv_90_ccw(handle_uv, old_width)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bipolar Forceps"),
    )

    working_uv = endpoints["working_uv"]
    estimated_handle_uv = endpoints["handle_uv"]
    assert np.linalg.norm(working_uv - tip_uv) < np.linalg.norm(
        estimated_handle_uv - tip_uv
    )
    assert np.linalg.norm(estimated_handle_uv - handle_uv) < np.linalg.norm(
        working_uv - handle_uv
    )
    assert endpoints["sign_confidence"] >= 0.2


@pytest.mark.parametrize("fixture_name", _ADSON_CASE_FIXTURES)
@pytest.mark.parametrize("quarter_turns", range(4))
def test_adson_uses_legacy_smaller_end_handle_policy(
    fixture_name: str,
    quarter_turns: int,
) -> None:
    fixture = _ADSON_CASE_FIXTURES[fixture_name]
    contour = np.asarray(fixture["contour"], dtype=np.int32)
    height = int(contour[:, 1].max()) + 1
    width = int(contour[:, 0].max()) + 1
    mask = np.zeros((height, width), dtype=np.uint8)
    __import__("cv2").fillPoly(mask, [contour], 1)
    for _ in range(quarter_turns):
        mask = np.rot90(mask)

    assert _sign_policy("Adson Forceps") == "smaller_end_is_handle"
    actual = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Adson Forceps"),
    )
    legacy = _pca_endpoints(
        mask.astype(bool),
        "smaller_end_is_handle",
    )
    np.testing.assert_allclose(actual["working_uv"], legacy["working_uv"])
    np.testing.assert_allclose(actual["handle_uv"], legacy["handle_uv"])
    np.testing.assert_allclose(actual["axis_uv"], legacy["axis_uv"])
    assert actual["sign_confidence"] == pytest.approx(
        legacy["sign_confidence"]
    )


@pytest.mark.parametrize(
    ("class_name", "canonical_class_id", "model_class_index", "fixture"),
    [
        (
            "Bipolar Forceps",
            5,
            4,
            _BIPOLAR_CASE_FIXTURES["0704_9_55s_straight"],
        ),
    ],
)
def test_pose_positive_y_points_to_working_tip(
    class_name: str,
    canonical_class_id: int,
    model_class_index: int,
    fixture: dict[str, list[list[int]] | list[int]],
) -> None:
    contour = np.asarray(fixture["contour"], dtype=np.int32)
    height = int(contour[:, 1].max()) + 1
    width = int(contour[:, 0].max()) + 1
    mask = np.zeros((height, width), dtype=np.uint8)
    __import__("cv2").fillPoly(mask, [contour], 1)
    instance = DetectionInstance(
        frame_local_instance_id=0,
        canonical_class_id=canonical_class_id,
        model_class_index=model_class_index,
        class_name=class_name,
        class_confidence=0.9,
        bbox_xyxy_px=(0.0, 0.0, float(width), float(height)),
        mask=mask.astype(bool),
    )
    detections = DetectionBatch(
        image_width=width,
        image_height=height,
        model_version=f"{class_name}-regression",
        ontology_version="test",
        instances=[instance],
    )
    depth = np.full((height, width), 0.8, dtype=np.float32)
    camera = CameraCalibration(
        width=width,
        height=height,
        k=np.array(
            [
                [200.0, 0.0, width / 2.0],
                [0.0, 200.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        distortion=np.zeros(5),
        frame_name="cam4",
        calibration_version="test",
    )
    plane = SupportPlane(
        normal=np.array([0.0, 0.0, 1.0]),
        offset_m=-0.8,
        config_version="test-plane",
    )

    item = PlanarPoseEstimator().estimate(
        detections,
        depth,
        camera,
        plane,
    ).instances[0]

    assert item.orientation_valid
    assert item.orientation_xyzw is not None
    x, y, z, w = item.orientation_xyzw
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    expected_y = np.append(
        np.asarray(fixture["tip_uv"], dtype=np.float64)
        - np.asarray(fixture["handle_uv"], dtype=np.float64),
        0.0,
    )
    expected_y /= np.linalg.norm(expected_y)
    assert float(rotation[:, 1] @ expected_y) > 0.99


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


def test_metric_depth_prefers_registrar_for_equal_resolution_grids() -> None:
    registrar = registrar_from_camera_fields(
        color_width=5,
        color_height=4,
        color_k=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
        color_d=[],
        color_frame="color",
        depth_width=5,
        depth_height=4,
        depth_k=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
        depth_d=[],
        depth_frame="depth",
        rotation=np.eye(3),
        translation_m=[0.01, 0.0, 0.0],
        calibration_version="same-size-test",
    )
    native = np.zeros((4, 5), dtype=np.uint16)
    native[1, 1] = 1000

    aligned = metric_depth_in_rgb_frame(native, 4, 5, 0.001, registrar)

    assert aligned is not None
    assert np.isnan(aligned[1, 1])
    assert float(aligned[1, 2]) == pytest.approx(1.0)


def test_realsense_flat_rotation_is_column_major() -> None:
    angle = np.deg2rad(17.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    raw_column_major = rotation.reshape(-1, order="F")

    transform = rigid_transform_from_realsense_extrinsics(
        raw_column_major,
        [0.0, 0.0, 0.0],
        source_frame="depth",
        target_frame="color",
        calibration_version="column-major-test",
    )

    np.testing.assert_allclose(transform.rotation, rotation, atol=1e-12)


def test_package_import_does_not_load_detector() -> None:
    import sys

    assert "pnu_surgical_tool.rfdetr_inference" not in sys.modules
    assert "pnu_surgical_tool.api" not in sys.modules
