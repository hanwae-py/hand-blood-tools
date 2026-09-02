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
    PlanarPoseConfig,
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


_ADSON_SINGLE_TIP_FIXTURE = {
    "contour": [
        [8, 24], [32, 15], [132, 11], [140, 14],
        [140, 34], [132, 37], [32, 33],
    ],
    "tip_uv": [8, 24],
    "handle_uv": [140, 24],
}


def _adson_two_prong_mask() -> np.ndarray:
    mask = np.zeros((52, 156), dtype=np.uint8)
    mask[13:39, 10:108] = 1
    mask[13:20, 100:146] = 1
    mask[32:39, 100:146] = 1
    return mask


def _adson_merged_two_tip_mask() -> np.ndarray:
    contour = np.array(
        [
            [10, 21], [104, 18], [145, 12],
            [145, 38], [104, 32], [10, 29],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((52, 156), dtype=np.uint8)
    __import__("cv2").fillPoly(mask, [contour], 1)
    return mask


_BOVIE_TAPER_FIXTURE = {
    "contour": [
        [0, 15], [24, 7], [100, 7], [112, 9],
        [112, 21], [100, 23], [24, 23],
    ],
    "tip_uv": [0, 15],
    "handle_uv": [112, 15],
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


@pytest.mark.parametrize("quarter_turns", range(4))
def test_bipolar_dark_endpoint_contributes_to_ensemble(quarter_turns: int) -> None:
    mask = np.zeros((52, 144), dtype=np.uint8)
    mask[10:42, 10:134] = 1
    image = np.full((*mask.shape, 3), 180, dtype=np.uint8)
    image[mask.astype(bool)] = (170, 170, 170)
    image[10:42, 10:46] = (20, 20, 20)
    dark_handle_uv = np.array((10.0, 25.5))
    working_tip_uv = np.array((133.0, 25.5))
    for _ in range(quarter_turns):
        old_width = mask.shape[1]
        mask = np.rot90(mask).copy()
        image = np.rot90(image).copy()
        dark_handle_uv = _rotate_uv_90_ccw(dark_handle_uv, old_width)
        working_tip_uv = _rotate_uv_90_ccw(working_tip_uv, old_width)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bipolar Forceps"),
        image_bgr=image,
    )

    evidence = endpoints["sign_evidence"]
    assert endpoints["sign_source"] == "bipolar_ensemble"
    assert evidence["colour_available"]
    assert evidence["colour"]["mode"] == "BLACK_HANDLE_ONLY"
    assert abs(evidence["colour_vote"]) == pytest.approx(0.65)
    assert np.linalg.norm(
        endpoints["handle_uv"] - dark_handle_uv
    ) < np.linalg.norm(endpoints["working_uv"] - dark_handle_uv)
    assert np.linalg.norm(
        endpoints["working_uv"] - working_tip_uv
    ) < np.linalg.norm(endpoints["handle_uv"] - working_tip_uv)
    assert endpoints["sign_confidence"] == pytest.approx(0.26)


@pytest.mark.parametrize("quarter_turns", range(4))
def test_bipolar_black_handle_and_blue_tip_agree(quarter_turns: int) -> None:
    mask = np.zeros((52, 144), dtype=np.uint8)
    mask[10:42, 10:134] = 1
    image = np.full((*mask.shape, 3), 170, dtype=np.uint8)
    image[mask.astype(bool)] = (170, 170, 170)
    image[10:42, 10:46] = (15, 15, 15)
    image[10:42, 98:134] = (220, 80, 35)
    black_handle_uv = np.array((10.0, 25.5))
    blue_tip_uv = np.array((133.0, 25.5))
    for _ in range(quarter_turns):
        old_width = mask.shape[1]
        mask = np.rot90(mask).copy()
        image = np.rot90(image).copy()
        black_handle_uv = _rotate_uv_90_ccw(black_handle_uv, old_width)
        blue_tip_uv = _rotate_uv_90_ccw(blue_tip_uv, old_width)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bipolar Forceps"),
        image_bgr=image,
    )

    colour = endpoints["sign_evidence"]["colour"]
    assert colour["mode"] == "BLACK_HANDLE_BLUE_TIP"
    assert colour["black_available"]
    assert colour["blue_available"]
    assert abs(colour["vote"]) == pytest.approx(1.0)
    assert np.linalg.norm(
        endpoints["handle_uv"] - black_handle_uv
    ) < np.linalg.norm(endpoints["working_uv"] - black_handle_uv)
    assert np.linalg.norm(
        endpoints["working_uv"] - blue_tip_uv
    ) < np.linalg.norm(endpoints["handle_uv"] - blue_tip_uv)


def test_bipolar_blue_endpoint_alone_votes_for_working_tip() -> None:
    mask = _rectangle_mask()
    image = np.full((*mask.shape, 3), 170, dtype=np.uint8)
    image[10:30, 45:60] = (220, 80, 35)

    endpoints = _pca_endpoints(
        mask,
        _sign_policy("Bipolar Forceps"),
        image_bgr=image,
    )

    colour = endpoints["sign_evidence"]["colour"]
    assert colour["mode"] == "BLUE_TIP_ONLY"
    assert not colour["black_available"]
    assert colour["blue_available"]
    assert endpoints["working_uv"][0] > endpoints["handle_uv"][0]


def test_bipolar_nonblack_endpoint_abstains_from_colour_vote() -> None:
    mask = _rectangle_mask()
    image = np.full((*mask.shape, 3), 170, dtype=np.uint8)
    image[10:30, 20:35] = (100, 100, 100)

    endpoints = _pca_endpoints(
        mask,
        _sign_policy("Bipolar Forceps"),
        image_bgr=image,
    )

    assert endpoints["sign_source"] == "bipolar_ensemble"
    assert not endpoints["sign_evidence"]["colour_available"]
    assert endpoints["sign_evidence"]["colour_vote"] == pytest.approx(0.0)
    assert endpoints["sign_confidence"] == pytest.approx(0.0)


def test_bipolar_uses_colour_but_ignores_external_wire() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:61, 60:121] = 1
    image = np.full((*mask.shape, 3), (170, 105, 35), dtype=np.uint8)
    image[mask.astype(bool)] = (150, 190, 215)
    cv2 = __import__("cv2")
    cv2.polylines(
        image,
        [np.array([[120, 50], [140, 53], [158, 66], [188, 63]], np.int32)],
        False,
        (235, 235, 235),
        4,
        cv2.LINE_AA,
    )
    image[40:61, 60:74] = (15, 15, 15)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bipolar Forceps"),
        image_bgr=image,
    )

    evidence = endpoints["sign_evidence"]
    assert endpoints["sign_source"] == "bipolar_ensemble"
    assert evidence["colour_available"]
    assert evidence["colour_vote"] < 0.0
    assert endpoints["working_uv"][0] > endpoints["handle_uv"][0]


def test_bipolar_ensemble_fuses_votes_by_weighted_sum() -> None:
    fixture = _BIPOLAR_CASE_FIXTURES["0704_9_55s_straight"]
    contour = np.asarray(fixture["contour"], dtype=np.int32)
    mask = np.zeros(
        (int(contour[:, 1].max()) + 1, int(contour[:, 0].max()) + 1),
        dtype=np.uint8,
    )
    __import__("cv2").fillPoly(mask, [contour], 1)
    image = np.full((*mask.shape, 3), 170, dtype=np.uint8)
    # Inject a dark cue at one endpoint so colour, taper and mass are all
    # represented in the final score rather than selected by an if/elif chain.
    image[:, : max(12, mask.shape[1] // 5)] = (15, 15, 15)
    image[~mask.astype(bool)] = 170

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bipolar Forceps"),
        image_bgr=image,
    )

    evidence = endpoints["sign_evidence"]
    assert evidence["colour_available"]
    expected = (
        0.45 * evidence["taper_vote"]
        + 0.40 * evidence["colour_vote"]
        + 0.15 * evidence["mass_vote"]
    )
    assert evidence["score"] == pytest.approx(expected)
    assert endpoints["sign_confidence"] == pytest.approx(abs(expected))


@pytest.mark.parametrize("quarter_turns", range(4))
def test_adson_single_tip_taper_points_to_working_tip(quarter_turns: int) -> None:
    fixture = _ADSON_SINGLE_TIP_FIXTURE
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

    assert _sign_policy("Adson Forceps") == "adson_layout_shape"
    endpoints = _pca_endpoints(
        mask.astype(bool),
        "adson_shape",
    )
    assert endpoints["sign_source"] == "adson_triangular_wide_tip"
    # The new placement policy intentionally uses the broad end of a global
    # triangular silhouette as the working side, even when a faint/narrow
    # terminal taper would have selected the opposite endpoint previously.
    assert np.linalg.norm(endpoints["working_uv"] - handle_uv) < np.linalg.norm(
        endpoints["handle_uv"] - handle_uv
    )
    assert np.linalg.norm(endpoints["handle_uv"] - tip_uv) < np.linalg.norm(
        endpoints["working_uv"] - tip_uv
    )
    assert endpoints["sign_confidence"] >= 0.2


@pytest.mark.parametrize("fixture_name", _ADSON_CASE_FIXTURES)
@pytest.mark.parametrize("quarter_turns", range(4))
def test_adson_recorded_contours_follow_layout_shape_direction(
    fixture_name: str,
    quarter_turns: int,
) -> None:
    fixture = _ADSON_CASE_FIXTURES[fixture_name]
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
        "adson_shape",
    )

    if endpoints["sign_source"] == "adson_triangular_wide_tip":
        layout = endpoints["sign_evidence"]["layout"]
        assert layout["accepted"]
        assert layout["layout"] == "TRIANGULAR_WIDE_TIP"
        assert endpoints["sign_confidence"] >= 0.2
    else:
        assert np.linalg.norm(endpoints["working_uv"] - tip_uv) < np.linalg.norm(
            endpoints["handle_uv"] - tip_uv
        )
        assert np.linalg.norm(endpoints["handle_uv"] - handle_uv) < np.linalg.norm(
            endpoints["working_uv"] - handle_uv
        )
    assert endpoints["sign_confidence"] >= 0.2


@pytest.mark.parametrize("quarter_turns", range(4))
def test_adson_two_prong_is_used_after_layout_rejection(quarter_turns: int) -> None:
    mask = _adson_two_prong_mask()
    tip_uv = np.array((145.0, 25.5))
    handle_uv = np.array((10.0, 25.5))
    for _ in range(quarter_turns):
        old_width = mask.shape[1]
        mask = np.rot90(mask).copy()
        tip_uv = _rotate_uv_90_ccw(tip_uv, old_width)
        handle_uv = _rotate_uv_90_ccw(handle_uv, old_width)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        "adson_shape",
    )

    assert endpoints["sign_source"] == "adson_two_prong_tip"
    assert np.linalg.norm(endpoints["working_uv"] - tip_uv) < np.linalg.norm(
        endpoints["handle_uv"] - tip_uv
    )
    assert np.linalg.norm(endpoints["handle_uv"] - handle_uv) < np.linalg.norm(
        endpoints["working_uv"] - handle_uv
    )
    assert endpoints["sign_confidence"] >= 0.35


@pytest.mark.parametrize("quarter_turns", range(4))
def test_adson_width_recovers_merged_face_on_tip(
    quarter_turns: int,
) -> None:
    mask = _adson_merged_two_tip_mask()
    tip_uv = np.array((145.0, 25.0))
    handle_uv = np.array((10.0, 25.0))
    for _ in range(quarter_turns):
        old_width = mask.shape[1]
        mask = np.rot90(mask).copy()
        tip_uv = _rotate_uv_90_ccw(tip_uv, old_width)
        handle_uv = _rotate_uv_90_ccw(handle_uv, old_width)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        "adson_face_on_shape",
    )

    assert endpoints["sign_source"] == "adson_triangular_wide_tip"
    assert np.linalg.norm(endpoints["working_uv"] - tip_uv) < np.linalg.norm(
        endpoints["handle_uv"] - tip_uv
    )
    assert np.linalg.norm(endpoints["handle_uv"] - handle_uv) < np.linalg.norm(
        endpoints["working_uv"] - handle_uv
    )
    assert endpoints["sign_confidence"] >= 0.3


def test_adson_layout_rule_is_always_enabled_for_class_based_pose() -> None:
    default = PlanarPoseEstimator()
    cam3 = PlanarPoseEstimator(
        PlanarPoseConfig(adson_face_on_width_enabled=True)
    )

    assert default._endpoint_sign_policy("Adson Forceps") == "adson_layout_shape"
    assert cam3._endpoint_sign_policy("Adson Forceps") == "adson_layout_shape"
    assert cam3._endpoint_sign_policy("Bovie") == "bovie_tip_taper"


def test_adson_ambiguous_rectangle_is_marked_as_fallback() -> None:
    endpoints = _pca_endpoints(
        _rectangle_mask(),
        "adson_shape",
    )

    assert endpoints["sign_source"] == "adson_shape_fallback"
    assert endpoints["sign_confidence"] == pytest.approx(0.0)


def test_camera_rule_changes_only_pca_axis_sign() -> None:
    mask = np.zeros((64, 96), dtype=np.uint8)
    __import__("cv2").line(mask, (12, 54), (84, 10), 9, 1)

    class_based = _pca_endpoints(mask.astype(bool), "larger_end_is_handle")
    cam3 = _pca_endpoints(mask.astype(bool), "positive_y_image_down")
    cam4 = _pca_endpoints(mask.astype(bool), "positive_y_image_right")

    cam3_positive_y = cam3["working_uv"] - cam3["handle_uv"]
    cam4_positive_y = cam4["working_uv"] - cam4["handle_uv"]
    assert cam3_positive_y[1] > 0.0
    assert cam4_positive_y[0] > 0.0
    np.testing.assert_allclose(
        abs(float(cam3["axis_uv"] @ class_based["axis_uv"])),
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        abs(float(cam4["axis_uv"] @ class_based["axis_uv"])),
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(cam3["axis_uv"], -cam4["axis_uv"], atol=1e-12)
    assert cam3["sign_confidence"] == pytest.approx(1.0)
    assert cam4["sign_confidence"] == pytest.approx(1.0)


def test_positive_y_image_direction_is_validated() -> None:
    with pytest.raises(ValueError, match="positive_y_image_direction"):
        PlanarPoseConfig(positive_y_image_direction="left")


@pytest.mark.parametrize("quarter_turns", range(4))
def test_bovie_tapered_endpoint_is_working_tip(quarter_turns: int) -> None:
    fixture = _BOVIE_TAPER_FIXTURE
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

    assert _sign_policy("Bovie") == "bovie_tip_taper"
    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bovie"),
    )
    assert np.linalg.norm(endpoints["working_uv"] - tip_uv) < np.linalg.norm(
        endpoints["handle_uv"] - tip_uv
    )
    assert np.linalg.norm(endpoints["handle_uv"] - handle_uv) < np.linalg.norm(
        endpoints["working_uv"] - handle_uv
    )
    assert endpoints["sign_confidence"] >= 0.2


@pytest.mark.parametrize("quarter_turns", range(4))
def test_bovie_external_wire_keeps_handle_priority(quarter_turns: int) -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:61, 60:121] = 1
    image = np.full((*mask.shape, 3), (170, 105, 35), dtype=np.uint8)
    image[mask.astype(bool)] = (150, 190, 215)
    cv2 = __import__("cv2")
    cv2.polylines(
        image,
        [np.array([[120, 50], [140, 53], [158, 66], [188, 63]], np.int32)],
        False,
        (235, 235, 235),
        4,
        cv2.LINE_AA,
    )
    # Bovie continues to use its external wire even though Bipolar ignores
    # both wire and colour cues.
    image[40:61, 60:74] = (15, 15, 15)
    handle_uv = np.array((120.0, 50.0))
    working_tip_uv = np.array((60.0, 50.0))
    for _ in range(quarter_turns):
        old_width = mask.shape[1]
        mask = np.rot90(mask).copy()
        image = np.rot90(image).copy()
        handle_uv = _rotate_uv_90_ccw(handle_uv, old_width)
        working_tip_uv = _rotate_uv_90_ccw(working_tip_uv, old_width)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bovie"),
        image_bgr=image,
    )

    assert endpoints["sign_source"] == "bovie_external_wire_handle"
    assert np.linalg.norm(
        endpoints["handle_uv"] - handle_uv
    ) < np.linalg.norm(endpoints["working_uv"] - handle_uv)
    assert np.linalg.norm(
        endpoints["working_uv"] - working_tip_uv
    ) < np.linalg.norm(endpoints["handle_uv"] - working_tip_uv)
    assert endpoints["sign_confidence"] >= 0.6


def test_thin_bovie_electrode_is_not_mistaken_for_wire() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:61, 60:121] = 1
    image = np.full((*mask.shape, 3), (170, 105, 35), dtype=np.uint8)
    image[mask.astype(bool)] = (150, 190, 215)
    __import__("cv2").line(image, (120, 50), (188, 50), (235, 235, 235), 1)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bovie"),
        image_bgr=image,
    )

    assert endpoints["sign_source"] == "bovie_tip_taper"


def test_broad_external_clutter_is_not_mistaken_for_wire() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[40:61, 60:121] = 1
    image = np.full((*mask.shape, 3), (170, 105, 35), dtype=np.uint8)
    image[mask.astype(bool)] = (150, 190, 215)
    __import__("cv2").rectangle(image, (119, 30), (185, 75), (235, 235, 235), -1)

    endpoints = _pca_endpoints(
        mask.astype(bool),
        _sign_policy("Bovie"),
        image_bgr=image,
    )

    assert endpoints["sign_source"] == "bovie_tip_taper"


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
