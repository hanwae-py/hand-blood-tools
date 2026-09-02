"""Mask/depth-based planar surgical-tool pose estimator.

The returned quaternion is a transport representation of a constrained pose:
translation comes from an observed depth-valid mask pixel, heading comes from
the mask's longitudinal axis, and the remaining orientation comes from the
supplied support-plane normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

from .types import (
    CameraCalibration,
    DetectionBatch,
    DetectionInstance,
    SupportPlane,
    ToolFrameResult,
    ToolInstanceResult,
)


POSE_MODE = "PLANAR_4DOF_WITH_NORMAL_PRIOR"
WIRE_CONTEXT_OUTWARD_LENGTH_FRACTION = 0.65
WIRE_CONTEXT_INWARD_ATTACHMENT_FRACTION = 0.25
WIRE_CONTEXT_HALF_WIDTH_FRACTION = 0.38
WIRE_CONTEXT_MINIMUM_EXTENT_FRACTION = 0.24
WIRE_CONTEXT_MINIMUM_RADIUS_PX = 0.80
WIRE_CONTEXT_MAXIMUM_RADIUS_PX = 3.50
WIRE_CONTEXT_MAXIMUM_RADIUS_FRACTION = 0.035
WIRE_CONTEXT_THIN_CANDIDATE_RADIUS_PX = 1.25
WIRE_CONTEXT_MINIMUM_CURVATURE_FRACTION = 0.025
WIRE_CONTEXT_MINIMUM_SCORE = 0.60
WIRE_CONTEXT_MINIMUM_SCORE_ADVANTAGE = 0.22
ADSON_PRONG_TERMINAL_FRACTION = 0.18


def _mask_coordinates_in_bbox(
    mask: np.ndarray,
    bbox_xyxy_px: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Scan the detector bbox instead of repeatedly scanning a full HD mask."""
    height, width = mask.shape
    if not all(math.isfinite(value) for value in bbox_xyxy_px):
        return np.nonzero(mask)
    x0 = max(0, int(math.floor(min(bbox_xyxy_px[0], bbox_xyxy_px[2]))) - 1)
    y0 = max(0, int(math.floor(min(bbox_xyxy_px[1], bbox_xyxy_px[3]))) - 1)
    x1 = min(width, int(math.ceil(max(bbox_xyxy_px[0], bbox_xyxy_px[2]))) + 1)
    y1 = min(height, int(math.ceil(max(bbox_xyxy_px[1], bbox_xyxy_px[3]))) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.nonzero(mask)
    crop_ys, crop_xs = np.nonzero(mask[y0:y1, x0:x1])
    if crop_xs.size == 0:
        return np.nonzero(mask)
    return crop_ys + y0, crop_xs + x0


ADSON_PRONG_MINIMUM_BALANCE = 0.35
ADSON_PRONG_MINIMUM_ADVANTAGE = 0.25
ADSON_LAYOUT_PROFILE_BINS = 9
ADSON_TRIANGLE_MINIMUM_WIDTH_RATIO = 1.40
ADSON_TRIANGLE_MINIMUM_LINEARITY_R2 = 0.65
ADSON_TRIANGLE_MINIMUM_MONOTONIC_FRACTION = 0.70
ADSON_TRIANGLE_MINIMUM_RELATIVE_WIDTH_CHANGE = 0.35
BIPOLAR_COLOUR_TERMINAL_FRACTION = 0.20
BIPOLAR_BLACK_LAB_L_THRESHOLD = 90.0
BIPOLAR_COLOUR_MINIMUM_DIRECTION_STRENGTH = 0.05
BIPOLAR_ENSEMBLE_TAPER_WEIGHT = 0.45
BIPOLAR_ENSEMBLE_COLOUR_WEIGHT = 0.40
BIPOLAR_ENSEMBLE_MASS_WEIGHT = 0.15


def _external_wire_handle_evidence(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    mean_uv: np.ndarray,
    direction_uv: np.ndarray,
    low: float,
    high: float,
    mask_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Find a clear thin continuation outside either endpoint.

    The detector mask remains the sole source for the pose axis and origin.
    RGB pixels outside that mask are used only to resolve its sign.  Evidence
    is intentionally conservative: a candidate must start at an endpoint,
    extend a meaningful distance, remain thin, and be distinct from the local
    background.  This rejects broad clutter and most drape folds while still
    permitting a curved insulated wire.
    """
    span = max(float(high - low), 1e-6)
    perpendicular = np.array((-direction_uv[1], direction_uv[0]))
    height, width = mask.shape
    if mask_coordinates is None:
        ys, xs = np.nonzero(mask)
    else:
        ys, xs = mask_coordinates
    transverse_center = float(
        np.median((np.column_stack((xs, ys)) - mean_uv) @ perpendicular)
    )
    outward_length = WIRE_CONTEXT_OUTWARD_LENGTH_FRACTION * span
    half_width = WIRE_CONTEXT_HALF_WIDTH_FRACTION * span

    def endpoint_region(
        end_projection: float,
        outward_sign: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        endpoint = (
            mean_uv
            + end_projection * direction_uv
            + transverse_center * perpendicular
        )
        outward = outward_sign * direction_uv
        radius = int(math.ceil(math.hypot(outward_length, half_width))) + 3
        x0 = max(0, int(math.floor(endpoint[0])) - radius)
        x1 = min(width, int(math.ceil(endpoint[0])) + radius + 1)
        y0 = max(0, int(math.floor(endpoint[1])) - radius)
        y1 = min(height, int(math.ceil(endpoint[1])) + radius + 1)
        return endpoint, outward, (x0, y0, x1, y1)

    regions = (
        endpoint_region(float(low), -1.0),
        endpoint_region(float(high), 1.0),
    )
    union_x0 = min(region[2][0] for region in regions)
    union_y0 = min(region[2][1] for region in regions)
    union_x1 = max(region[2][2] for region in regions)
    union_y1 = max(region[2][3] for region in regions)
    lab_union = cv2.cvtColor(
        image_bgr[union_y0:union_y1, union_x0:union_x1],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    exclusion_union = _dilated_mask_crop(
        mask,
        union_x0,
        union_y0,
        union_x1,
        union_y1,
    )

    def endpoint_score(
        endpoint: np.ndarray,
        outward: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> dict[str, float]:
        x0, y0, x1, y1 = bounds
        if x1 - x0 < 5 or y1 - y0 < 5:
            return {"score": 0.0, "extent": 0.0, "radius_p90": math.inf}

        crop_y, crop_x = np.mgrid[y0:y1, x0:x1]
        relative_x = crop_x.astype(np.float64) - endpoint[0]
        relative_y = crop_y.astype(np.float64) - endpoint[1]
        longitudinal = relative_x * outward[0] + relative_y * outward[1]
        transverse = relative_x * perpendicular[0] + relative_y * perpendicular[1]
        # A slightly widening corridor allows a real cable to curve, without
        # admitting the complete circular neighbourhood around the endpoint.
        allowed_half_width = 0.25 * span + 0.30 * np.maximum(longitudinal, 0.0)
        corridor = (
            (longitudinal >= -WIRE_CONTEXT_INWARD_ATTACHMENT_FRACTION * span)
            & (longitudinal <= outward_length)
            & (np.abs(transverse) <= np.minimum(allowed_half_width, half_width))
        )
        union_slice = (
            slice(y0 - union_y0, y1 - union_y0),
            slice(x0 - union_x0, x1 - union_x0),
        )
        excluded = exclusion_union[union_slice]
        background_region = corridor & ~excluded
        lab_crop = lab_union[union_slice]
        background_values = lab_crop[background_region]
        if len(background_values) < 40:
            return {"score": 0.0, "extent": 0.0, "radius_p90": math.inf}

        background = np.median(background_values, axis=0)
        colour_distance = np.linalg.norm(
            lab_crop - background.reshape(1, 1, 3),
            axis=2,
        )
        median_distance = float(np.median(colour_distance[background_region]))
        mad = float(
            np.median(
                np.abs(colour_distance[background_region] - median_distance)
            )
        )
        colour_threshold = max(22.0, median_distance + 4.0 * max(mad, 1.0))
        candidate = corridor & ~excluded & (colour_distance >= colour_threshold)
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        candidate[excluded] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )
        seed = (
            (
                longitudinal
                >= -WIRE_CONTEXT_INWARD_ATTACHMENT_FRACTION * span
            )
            & (longitudinal <= 0.09 * span)
            & (np.abs(transverse) <= 0.25 * span)
        )
        best = {"score": 0.0, "extent": 0.0, "radius_p90": math.inf}
        maximum_radius = max(
            WIRE_CONTEXT_MAXIMUM_RADIUS_PX,
            WIRE_CONTEXT_MAXIMUM_RADIUS_FRACTION * span,
        )
        minimum_extent = WIRE_CONTEXT_MINIMUM_EXTENT_FRACTION * span
        for label in range(1, count):
            component = labels == label
            if not np.any(component & seed):
                continue
            component_longitudinal = longitudinal[component]
            extent = float(np.max(component_longitudinal))
            if extent < minimum_extent:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            distance = cv2.distanceTransform(
                component.astype(np.uint8),
                cv2.DIST_L2,
                3,
            )
            # The cable may leave the side of a broad handle/connector.  Judge
            # thinness only after it has travelled beyond the tool endpoint;
            # otherwise the attachment body would make a clear wire look wide.
            outward_component = (
                component
                & (longitudinal >= 0.10 * span)
                & (longitudinal <= 0.42 * span)
            )
            radii = (
                distance[outward_component]
                if int(np.count_nonzero(outward_component)) >= 8
                else distance[component]
            )
            radius_p90 = float(np.quantile(radii, 0.90))
            if not WIRE_CONTEXT_MINIMUM_RADIUS_PX <= radius_p90 <= maximum_radius:
                continue
            occupied_bins = []
            bin_centres = []
            transverse_centres = []
            component_transverse = transverse[component]
            for start in np.linspace(0.0, extent, 9)[:-1]:
                stop = start + extent / 8.0
                in_bin = (
                    (component_longitudinal >= start)
                    & (component_longitudinal < stop)
                )
                occupied_bins.append(np.any(in_bin))
                if np.any(in_bin):
                    bin_centres.append(0.5 * (start + stop))
                    transverse_centres.append(
                        float(np.median(component_transverse[in_bin]))
                    )
            continuity = float(np.mean(occupied_bins))
            if continuity < 0.75:
                continue
            curvature_residual = 0.0
            if len(bin_centres) >= 4:
                line = np.polyfit(bin_centres, transverse_centres, 1)
                fitted = np.polyval(line, bin_centres)
                curvature_residual = float(
                    np.max(np.abs(np.asarray(transverse_centres) - fitted))
                )
            if (
                radius_p90 < WIRE_CONTEXT_THIN_CANDIDATE_RADIUS_PX
                and curvature_residual
                < WIRE_CONTEXT_MINIMUM_CURVATURE_FRACTION * span
            ):
                continue
            extent_score = min(extent / max(0.45 * span, 1.0), 1.0)
            thinness_score = min(maximum_radius / max(radius_p90, 1e-6), 1.0)
            score = extent_score * continuity * thinness_score
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "extent": extent,
                    "radius_p90": radius_p90,
                }
        return best

    low_result = endpoint_score(*regions[0])
    high_result = endpoint_score(*regions[1])
    low_score = float(low_result["score"])
    high_score = float(high_result["score"])
    best_score = max(low_score, high_score)
    accepted = bool(
        best_score >= WIRE_CONTEXT_MINIMUM_SCORE
        and abs(low_score - high_score) >= WIRE_CONTEXT_MINIMUM_SCORE_ADVANTAGE
    )
    return {
        "accepted": accepted,
        "handle_end": "low" if low_score >= high_score else "high",
        "low_score": low_score,
        "high_score": high_score,
        "low_extent": float(low_result["extent"]),
        "high_extent": float(high_result["extent"]),
        "confidence": best_score,
    }


def _dilated_mask_crop(
    mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    """Return an exact 3x3-dilated slice without dilating the full HD mask."""
    height, width = mask.shape
    expanded_x0 = max(0, int(x0) - 1)
    expanded_y0 = max(0, int(y0) - 1)
    expanded_x1 = min(width, int(x1) + 1)
    expanded_y1 = min(height, int(y1) + 1)
    expanded = np.ascontiguousarray(
        mask[expanded_y0:expanded_y1, expanded_x0:expanded_x1],
        dtype=np.uint8,
    )
    dilated = cv2.dilate(
        expanded,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    return dilated[
        int(y0) - expanded_y0:int(y1) - expanded_y0,
        int(x0) - expanded_x0:int(x1) - expanded_x0,
    ]


def _terminal_taper_scores(
    projection: np.ndarray,
    low: float,
    high: float,
) -> tuple[float, float]:
    """Measure contraction from each inner shoulder to its terminal end."""
    span = max(float(high - low), 1e-6)
    band = 0.10 * span
    low_terminal = int(np.sum((projection >= low) & (projection < low + band)))
    low_shoulder = int(
        np.sum((projection >= low + band) & (projection < low + 2.0 * band))
    )
    high_terminal = int(
        np.sum((projection <= high) & (projection > high - band))
    )
    high_shoulder = int(
        np.sum((projection <= high - band) & (projection > high - 2.0 * band))
    )
    low_taper = max(float(low_shoulder - low_terminal), 0.0) / max(
        float(low_shoulder), 1.0
    )
    high_taper = max(float(high_shoulder - high_terminal), 0.0) / max(
        float(high_shoulder), 1.0
    )
    return low_taper, high_taper


def _adson_terminal_prong_evidence(
    mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    projection: np.ndarray,
    low: float,
    high: float,
) -> dict[str, Any]:
    """Find a pair of substantial disconnected jaws in either terminal cap.

    Adson jaws can be resolved as two mask components in a face-on view. The
    components normally join farther toward the handle, so only the terminal
    cap is inspected. Small detached segmentation islands are excluded by an
    area ratio before the two largest components are compared.
    """
    span = max(float(high - low), 1e-6)

    def cap_score(selection: np.ndarray) -> tuple[float, int]:
        cap_ys = ys[selection]
        cap_xs = xs[selection]
        if len(cap_xs) == 0:
            return 0.0, 0
        y0, y1 = int(cap_ys.min()), int(cap_ys.max()) + 1
        x0, x1 = int(cap_xs.min()), int(cap_xs.max()) + 1
        cap = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cap[cap_ys - y0, cap_xs - x0] = 1
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            cap,
            connectivity=8,
        )
        areas = sorted(
            (int(area) for area in stats[1:, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        if len(areas) < 2:
            return 0.0, len(areas)
        minimum_area = max(3, int(math.ceil(sum(areas) * 0.10)))
        substantial = [area for area in areas if area >= minimum_area]
        if len(substantial) < 2:
            return 0.0, len(substantial)
        balance = float(substantial[1] / max(substantial[0], 1))
        return balance, len(substantial)

    cap_length = ADSON_PRONG_TERMINAL_FRACTION * span
    low_score, low_components = cap_score(projection <= low + cap_length)
    high_score, high_components = cap_score(projection >= high - cap_length)
    best_score = max(low_score, high_score)
    accepted = bool(
        best_score >= ADSON_PRONG_MINIMUM_BALANCE
        and abs(low_score - high_score) >= ADSON_PRONG_MINIMUM_ADVANTAGE
    )
    return {
        "accepted": accepted,
        "tip_end": "low" if low_score >= high_score else "high",
        "low_score": low_score,
        "high_score": high_score,
        "low_components": low_components,
        "high_components": high_components,
        "confidence": best_score,
    }


def _adson_layout_shape_evidence(
    mean_uv: np.ndarray,
    direction_uv: np.ndarray,
    low: float,
    high: float,
    mask_coordinates: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Classify a whole-mask triangular Adson placement before jaw cues.

    Face-on Adson masks often lose the faint gap between the two jaws.  The
    useful cue then is not a pair of terminal connected components, but a
    gradual, approximately triangular widening across the complete mask.  A
    short tapered electrode/side-on tip widens abruptly and then stays nearly
    parallel, so it should fail the global linearity test and continue to the
    prong/taper rules.
    """
    ys, xs = mask_coordinates
    uv = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    centered = uv - mean_uv
    projection = centered @ direction_uv
    transverse = centered @ np.array((-direction_uv[1], direction_uv[0]))
    span = max(float(high - low), 1e-6)
    edges = np.linspace(low, high, ADSON_LAYOUT_PROFILE_BINS + 1)
    centres: list[float] = []
    widths: list[float] = []
    for index, (start, stop) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        selection = (projection >= start) & (
            projection <= stop
            if index == ADSON_LAYOUT_PROFILE_BINS - 1
            else projection < stop
        )
        values = transverse[selection]
        if len(values) < 8:
            continue
        centres.append(float((0.5 * (start + stop) - low) / span))
        widths.append(
            float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        )

    if len(widths) < max(6, ADSON_LAYOUT_PROFILE_BINS - 2):
        return {
            "accepted": False,
            "layout": "INSUFFICIENT_WIDTH_PROFILE",
            "tip_end": "low",
            "confidence": 0.0,
            "widths": tuple(widths),
        }

    profile_x = np.asarray(centres, dtype=np.float64)
    profile_width = np.asarray(widths, dtype=np.float64)
    slope, intercept = np.polyfit(profile_x, profile_width, 1)
    fitted = slope * profile_x + intercept
    residual = float(np.sum((profile_width - fitted) ** 2))
    total = float(np.sum((profile_width - profile_width.mean()) ** 2))
    linearity_r2 = max(0.0, 1.0 - residual / max(total, 1e-9))

    low_width = float(np.mean(profile_width[:2]))
    high_width = float(np.mean(profile_width[-2:]))
    wider_end = "low" if low_width >= high_width else "high"
    wider_width = max(low_width, high_width)
    narrower_width = min(low_width, high_width)
    width_ratio = wider_width / max(narrower_width, 1e-6)
    oriented_differences = np.diff(profile_width)
    if wider_end == "low":
        oriented_differences *= -1.0
    tolerance = 0.05 * max(float(profile_width.mean()), 1.0)
    monotonic_fraction = float(np.mean(oriented_differences >= -tolerance))
    relative_width_change = abs(high_width - low_width) / max(
        float(profile_width.mean()), 1e-6
    )
    accepted = bool(
        narrower_width > 1.0
        and width_ratio >= ADSON_TRIANGLE_MINIMUM_WIDTH_RATIO
        and linearity_r2 >= ADSON_TRIANGLE_MINIMUM_LINEARITY_R2
        and monotonic_fraction >= ADSON_TRIANGLE_MINIMUM_MONOTONIC_FRACTION
        and relative_width_change
        >= ADSON_TRIANGLE_MINIMUM_RELATIVE_WIDTH_CHANGE
    )
    width_confidence = 1.0 - narrower_width / max(wider_width, 1e-6)
    confidence = (
        0.45 * width_confidence
        + 0.35 * linearity_r2
        + 0.20 * monotonic_fraction
    ) if accepted else 0.0
    return {
        "accepted": accepted,
        "layout": "TRIANGULAR_WIDE_TIP" if accepted else "SLENDER_OR_NONTRIANGULAR",
        "tip_end": wider_end,
        "low_width": low_width,
        "high_width": high_width,
        "width_ratio": float(width_ratio),
        "linearity_r2": linearity_r2,
        "monotonic_fraction": monotonic_fraction,
        "relative_width_change": relative_width_change,
        "widths": tuple(float(value) for value in profile_width),
        "confidence": float(confidence),
    }


def _bipolar_colour_handle_evidence(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    mean_uv: np.ndarray,
    direction_uv: np.ndarray,
    low: float,
    high: float,
    mask_coordinates: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Return a signed, abstaining colour vote for the Bipolar handle.

    A positive vote means the high PCA endpoint is the dark proximal handle;
    a negative vote means the low endpoint is the handle.  This function does
    not select the endpoint by itself: the caller combines this vote with
    taper and terminal-mass votes.
    """
    mask_ys, mask_xs = mask_coordinates
    x0, x1 = int(mask_xs.min()), int(mask_xs.max()) + 1
    y0, y1 = int(mask_ys.min()), int(mask_ys.max()) + 1
    mask_crop = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    eroded = cv2.erode(
        mask_crop,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    if int(np.count_nonzero(eroded)) < 50:
        eroded = mask_crop.astype(bool)
    ys, xs = np.nonzero(eroded)
    ys += y0
    xs += x0
    if len(xs) < 20:
        return {"available": False, "vote": 0.0, "confidence": 0.0}
    uv = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    projection = (uv - mean_uv) @ direction_uv
    span = max(float(high - low), 1e-6)
    cap = BIPOLAR_COLOUR_TERMINAL_FRACTION * span
    low_selection = projection <= low + cap
    high_selection = projection >= high - cap
    if (
        int(np.count_nonzero(low_selection)) < 10
        or int(np.count_nonzero(high_selection)) < 10
    ):
        return {"available": False, "vote": 0.0, "confidence": 0.0}

    lab_l_crop = cv2.cvtColor(
        image_bgr[y0:y1, x0:x1],
        cv2.COLOR_BGR2LAB,
    )[:, :, 0]
    low_values = lab_l_crop[
        ys[low_selection] - y0,
        xs[low_selection] - x0,
    ].astype(np.float64)
    high_values = lab_l_crop[
        ys[high_selection] - y0,
        xs[high_selection] - x0,
    ].astype(np.float64)
    low_median = float(np.median(low_values))
    high_median = float(np.median(high_values))
    low_black_fraction = float(
        np.mean(low_values <= BIPOLAR_BLACK_LAB_L_THRESHOLD)
    )
    high_black_fraction = float(
        np.mean(high_values <= BIPOLAR_BLACK_LAB_L_THRESHOLD)
    )

    # Positive signs consistently mean "high endpoint is the handle".
    median_vote = float(np.clip((low_median - high_median) / 30.0, -1.0, 1.0))
    black_fraction_vote = float(np.clip(
        (high_black_fraction - low_black_fraction) / 0.60,
        -1.0,
        1.0,
    ))
    darker_median = min(low_median, high_median)
    darkness_strength = float(np.clip(
        (BIPOLAR_BLACK_LAB_L_THRESHOLD - darker_median) / 25.0,
        0.0,
        1.0,
    ))
    black_strength = min(
        max(low_black_fraction, high_black_fraction) / 0.60,
        1.0,
    )
    colour_presence = max(darkness_strength, black_strength)
    vote = float(np.clip(
        (0.75 * median_vote + 0.25 * black_fraction_vote)
        * colour_presence,
        -1.0,
        1.0,
    ))
    return {
        "available": bool(
            abs(vote) >= BIPOLAR_COLOUR_MINIMUM_DIRECTION_STRENGTH
        ),
        "vote": vote,
        "confidence": abs(vote),
        "low_median_l": low_median,
        "high_median_l": high_median,
        "low_black_fraction": low_black_fraction,
        "high_black_fraction": high_black_fraction,
    }


def _bipolar_ensemble_evidence(
    mask: np.ndarray,
    projection: np.ndarray,
    low: float,
    high: float,
    low_mass: int,
    high_mass: int,
    mean_uv: np.ndarray,
    direction_uv: np.ndarray,
    image_bgr: np.ndarray | None,
    mask_coordinates: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Fuse taper, dark-colour and terminal-mass cues without precedence."""
    low_taper, high_taper = _terminal_taper_scores(projection, low, high)
    taper_separation = (high_taper - low_taper) / max(
        high_taper + low_taper,
        1e-6,
    )
    taper_strength = min(max(low_taper, high_taper) / 0.10, 1.0)
    taper_vote = float(np.clip(
        taper_separation * taper_strength,
        -1.0,
        1.0,
    ))
    mass_vote = float(np.clip(
        (high_mass - low_mass) / max(high_mass + low_mass, 1),
        -1.0,
        1.0,
    ))
    colour = (
        _bipolar_colour_handle_evidence(
            mask,
            image_bgr,
            mean_uv,
            direction_uv,
            low,
            high,
            mask_coordinates,
        )
        if image_bgr is not None
        else {"available": False, "vote": 0.0, "confidence": 0.0}
    )

    weighted_votes = [
        (BIPOLAR_ENSEMBLE_TAPER_WEIGHT, taper_vote),
        (BIPOLAR_ENSEMBLE_MASS_WEIGHT, mass_vote),
    ]
    if bool(colour["available"]):
        weighted_votes.append((
            BIPOLAR_ENSEMBLE_COLOUR_WEIGHT,
            float(colour["vote"]),
        ))
    total_weight = sum(weight for weight, _vote in weighted_votes)
    score = sum(weight * vote for weight, vote in weighted_votes) / total_weight
    return {
        "handle_end": "high" if score >= 0.0 else "low",
        "score": float(score),
        "confidence": abs(float(score)),
        "taper_vote": taper_vote,
        "mass_vote": mass_vote,
        "colour_vote": float(colour["vote"]),
        "colour_available": bool(colour["available"]),
        "colour": colour,
    }


def _pca_endpoints(
    mask: np.ndarray,
    sign_policy: str,
    image_bgr: np.ndarray | None = None,
    mask_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    if mask_coordinates is None:
        ys, xs = np.where(mask)
    else:
        ys, xs = mask_coordinates
    if len(xs) < 20:
        raise ValueError("MASK_TOO_SMALL")
    uv = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    mean = uv.mean(axis=0)
    centered = uv - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    direction /= np.linalg.norm(direction)
    projection = centered @ direction
    low, high = np.quantile(projection, [0.02, 0.98])
    span = max(float(high - low), 1e-6)
    low_mass = int(np.sum(projection <= low + 0.25 * span))
    high_mass = int(np.sum(projection >= high - 0.25 * span))
    sign_source = sign_policy

    if sign_policy in ("positive_y_image_down", "positive_y_image_right"):
        # The PCA eigenvector defines an undirected line.  Select only its
        # sign so that the projected tool-frame +Y direction points toward
        # the configured image direction; never rotate the PCA axis itself.
        desired_positive_y_uv = {
            "positive_y_image_down": np.array((0.0, 1.0)),
            "positive_y_image_right": np.array((1.0, 0.0)),
        }[sign_policy]
        positive_y_uv = -direction
        alignment = float(positive_y_uv @ desired_positive_y_uv)
        secondary = float(
            positive_y_uv[0]
            if sign_policy == "positive_y_image_down"
            else positive_y_uv[1]
        )
        if alignment < 0.0 or (
            abs(alignment) < 1e-6 and secondary < 0.0
        ):
            direction *= -1
            low, high = -high, -low
            low_mass, high_mass = high_mass, low_mass
        confidence = 1.0
    elif sign_policy == "cam4_positive_axis":
        if direction[0] < 0 or (abs(direction[0]) < 1e-6 and direction[1] < 0):
            direction *= -1
            low, high = -high, -low
            low_mass, high_mass = high_mass, low_mass
        confidence = 1.0
    elif sign_policy == "larger_end_is_handle":
        if low_mass > high_mass:
            direction *= -1
            low, high = -high, -low
            low_mass, high_mass = high_mass, low_mass
        confidence = abs(high_mass - low_mass) / max(high_mass + low_mass, 1)
    elif sign_policy == "smaller_end_is_handle":
        if low_mass < high_mass:
            direction *= -1
            low, high = -high, -low
            low_mass, high_mass = high_mass, low_mass
        confidence = abs(high_mass - low_mass) / max(high_mass + low_mass, 1)
    elif sign_policy in (
        "adson_shape",
        "adson_face_on_shape",
        "adson_layout_shape",
    ):
        layout = _adson_layout_shape_evidence(
            mean,
            direction,
            float(low),
            float(high),
            (ys, xs),
        )
        sign_evidence = {"layout": layout}
        if layout["accepted"]:
            if layout["tip_end"] == "high":
                direction *= -1
                low, high = -high, -low
                low_mass, high_mass = high_mass, low_mass
            confidence = float(layout["confidence"])
            sign_source = "adson_triangular_wide_tip"
        else:
            prongs = _adson_terminal_prong_evidence(
                mask,
                ys,
                xs,
                projection,
                float(low),
                float(high),
            )
            sign_evidence["prongs"] = prongs
            if prongs["accepted"]:
                if prongs["tip_end"] == "high":
                    direction *= -1
                    low, high = -high, -low
                    low_mass, high_mass = high_mass, low_mass
                confidence = float(prongs["confidence"])
                sign_source = "adson_two_prong_tip"
            else:
                low_taper, high_taper = _terminal_taper_scores(projection, low, high)
                sign_evidence["low_taper"] = low_taper
                sign_evidence["high_taper"] = high_taper
                taper_strength = max(low_taper, high_taper)
                if abs(low_taper - high_taper) < 1e-6:
                    # With no visible prong split, face-on evidence, or taper,
                    # prefer the smaller terminal mass as the tip but mark
                    # the sign as ambiguous.
                    if low_mass > high_mass:
                        direction *= -1
                        low, high = -high, -low
                        low_mass, high_mass = high_mass, low_mass
                    confidence = 0.0
                    sign_source = "adson_shape_fallback"
                else:
                    if high_taper > low_taper:
                        direction *= -1
                        low, high = -high, -low
                        low_mass, high_mass = high_mass, low_mass
                    taper_separation = abs(low_taper - high_taper) / max(
                        low_taper + high_taper,
                        1e-6,
                    )
                    confidence = taper_separation * min(taper_strength / 0.10, 1.0)
                    sign_source = "adson_tip_taper"
    elif sign_policy == "bipolar_ensemble":
        ensemble = _bipolar_ensemble_evidence(
            mask,
            projection,
            float(low),
            float(high),
            low_mass,
            high_mass,
            mean,
            direction,
            image_bgr,
            (ys, xs),
        )
        if ensemble["handle_end"] == "low":
            direction *= -1
            low, high = -high, -low
            low_mass, high_mass = high_mass, low_mass
        confidence = float(ensemble["confidence"])
        sign_source = "bipolar_ensemble"
        sign_evidence = ensemble
    elif sign_policy == "bovie_tip_taper":
        wire = (
            _external_wire_handle_evidence(
                mask,
                image_bgr,
                mean,
                direction,
                float(low),
                float(high),
                mask_coordinates=mask_coordinates,
            )
            if image_bgr is not None
            else None
        )
        if wire is not None and wire["accepted"]:
            if wire["handle_end"] == "low":
                direction *= -1
                low, high = -high, -low
                low_mass, high_mass = high_mass, low_mass
            confidence = float(wire["confidence"])
            sign_source = "bovie_external_wire_handle"
        else:
            low_taper, high_taper = _terminal_taper_scores(projection, low, high)
            taper_strength = max(low_taper, high_taper)
            if abs(low_taper - high_taper) < 1e-6:
                # Taper-based policies use the larger terminal mass as the
                # handle fallback. Confidence remains zero because taper did
                # not disambiguate it.
                if low_mass > high_mass:
                    direction *= -1
                    low, high = -high, -low
                    low_mass, high_mass = high_mass, low_mass
                confidence = 0.0
            else:
                should_flip = high_taper > low_taper
                if should_flip:
                    direction *= -1
                    low, high = -high, -low
                    low_mass, high_mass = high_mass, low_mass
                taper_separation = abs(low_taper - high_taper) / max(
                    low_taper + high_taper,
                    1e-6,
                )
                confidence = taper_separation * min(taper_strength / 0.10, 1.0)
    else:
        raise ValueError(f"Unknown sign policy: {sign_policy}")

    perpendicular = np.array((-direction[1], direction[0]))
    transverse_center = float(np.median(centered @ perpendicular))
    working_uv = mean + low * direction + transverse_center * perpendicular
    handle_uv = mean + high * direction + transverse_center * perpendicular
    return {
        "working_uv": working_uv,
        "handle_uv": handle_uv,
        "origin_uv": 0.5 * (working_uv + handle_uv),
        "axis_uv": direction,
        "axis_length_px": span,
        "axis_anisotropy": float(
            np.max(eigenvalues) / max(float(np.min(eigenvalues)), 1e-9)
        ),
        "sign_confidence": float(confidence),
        "sign_source": sign_source,
        "sign_evidence": locals().get("sign_evidence", {}),
    }


def longitudinal_origin_uv(
    mask: np.ndarray,
    class_name: str,
    bbox_xyxy_px: tuple[float, float, float, float] | None = None,
) -> np.ndarray | None:
    """Return the mask longitudinal-axis midpoint, or None if the mask is unusable."""
    try:
        coordinates = (
            _mask_coordinates_in_bbox(mask, bbox_xyxy_px)
            if bbox_xyxy_px is not None
            else None
        )
        return _pca_endpoints(
            mask,
            _sign_policy(class_name),
            mask_coordinates=coordinates,
        )["origin_uv"]
    except ValueError:
        return None


def sample_depth_at_uv(depth_m: np.ndarray, uv: np.ndarray) -> float | None:
    """Return metric depth at a pixel, or None when the sample is missing/invalid."""
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or uv is None:
        return None
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))
    if v < 0 or u < 0 or v >= depth.shape[0] or u >= depth.shape[1]:
        return None
    value = float(depth[v, u])
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _select_reference_pixel(
    mask: np.ndarray,
    desired_uv: np.ndarray,
    longitudinal_axis_uv: np.ndarray,
    axis_length_px: float,
    depth_m: np.ndarray,
    mask_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    if mask_coordinates is None:
        ys, xs = np.where(mask)
    else:
        ys, xs = mask_coordinates
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    distance_crop = cv2.distanceTransform(
        mask[y0:y1, x0:x1].astype(np.uint8), cv2.DIST_L2, 5
    )
    uv = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    relative = uv - desired_uv.reshape(1, 2)
    longitudinal_distance = np.abs(relative @ longitudinal_axis_uv)
    euclidean_distance = np.linalg.norm(relative, axis=1)
    depth_values = depth_m[ys, xs]
    depth_valid = np.isfinite(depth_values) & (depth_values > 0.0)
    central_half_width = max(4.0, 0.10 * axis_length_px)
    candidates = depth_valid & (longitudinal_distance <= central_half_width)
    selection_mode = "central_longitudinal_band_max_clearance"
    if not np.any(candidates):
        candidates = depth_valid
        selection_mode = "fallback_any_depth_valid_mask_pixel"
    if not np.any(candidates):
        raise ValueError("NO_VALID_DEPTH_IN_MASK")
    candidate_indices = np.where(candidates)[0]
    clearance = distance_crop[
        ys[candidate_indices] - y0,
        xs[candidate_indices] - x0,
    ].astype(np.float64)
    score = clearance / max(float(clearance.max()), 1e-6)
    score -= 0.35 * euclidean_distance[candidate_indices] / max(axis_length_px, 1.0)
    selected = int(candidate_indices[int(np.argmax(score))])
    u, v = int(xs[selected]), int(ys[selected])
    return {
        "uv": np.array((float(u), float(v)), dtype=np.float64),
        "depth_m": float(depth_m[v, u]),
        "selection_mode": selection_mode,
        "boundary_clearance_px": float(distance_crop[v - y0, u - x0]),
    }


def _pixel_rays(uv: np.ndarray, camera: CameraCalibration) -> np.ndarray:
    normalized = cv2.undistortPoints(
        uv.reshape(-1, 1, 2).astype(np.float64), camera.k, camera.distortion
    )
    xy = normalized.reshape(-1, 2)
    return np.column_stack((xy, np.ones(len(xy), dtype=np.float64)))


def _intersect_plane(rays: np.ndarray, plane: SupportPlane) -> np.ndarray:
    denominator = rays @ plane.normal
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("RAY_PARALLEL_TO_SUPPORT_PLANE")
    distances = -plane.offset_m / denominator
    if np.any(distances <= 0.0):
        raise ValueError("SUPPORT_PLANE_BEHIND_CAMERA")
    return rays * distances[:, None]


def _quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quaternion = np.array((qx, qy, qz, qw), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _sign_policy(class_name: str) -> str:
    if class_name == "Army-Navy Retractor":
        return "cam4_positive_axis"
    if class_name == "Adson Forceps":
        return "adson_layout_shape"
    if class_name == "Bipolar Forceps":
        return "bipolar_ensemble"
    if class_name == "Bovie":
        return "bovie_tip_taper"
    return "larger_end_is_handle"


@dataclass(frozen=True)
class PlanarPoseConfig:
    convention_version: str = "pnu.cam4.planar_tool_pose_convention.v2"
    minimum_mask_pixels: int = 20
    minimum_depth_ratio: float = 0.05
    minimum_axis_anisotropy: float = 2.0
    minimum_endpoint_sign_confidence: float = 0.20
    positive_y_image_direction: str = "class_based"
    # Retained for launch/API compatibility. Whole-mask Adson layout
    # classification is now always active for class-based endpoint signs.
    adson_face_on_width_enabled: bool = True

    def __post_init__(self) -> None:
        if self.positive_y_image_direction not in (
            "class_based",
            "down",
            "right",
        ):
            raise ValueError(
                "positive_y_image_direction must be class_based, down, or right"
            )


class PlanarPoseEstimator:
    def __init__(self, config: PlanarPoseConfig | None = None) -> None:
        self.config = config or PlanarPoseConfig()
        self.last_runtime_diagnostics: dict[str, Any] = {
            "instance_count": 0,
            "instance_latency_ms": [],
        }

    def _endpoint_sign_policy(self, class_name: str) -> str:
        if self.config.positive_y_image_direction == "class_based":
            return _sign_policy(class_name)
        return f"positive_y_image_{self.config.positive_y_image_direction}"

    def estimate(
        self,
        detections: DetectionBatch,
        aligned_depth_m: np.ndarray,
        camera: CameraCalibration,
        support_plane: SupportPlane,
        frame_key: str | int | None = None,
        image_bgr: np.ndarray | None = None,
    ) -> ToolFrameResult:
        depth = np.asarray(aligned_depth_m)
        expected = (detections.image_height, detections.image_width)
        if depth.shape != expected:
            raise ValueError(f"aligned_depth_m shape {depth.shape} != image shape {expected}")
        if depth.dtype not in (np.float32, np.float64):
            raise TypeError("aligned_depth_m must use float32 or float64 metres")
        if image_bgr is not None:
            image_bgr = np.asarray(image_bgr)
            if image_bgr.shape != (*expected, 3) or image_bgr.dtype != np.uint8:
                raise ValueError("image_bgr must be uint8 HxWx3 matching detections")
        if (camera.width, camera.height) != (
            detections.image_width,
            detections.image_height,
        ):
            raise ValueError("camera calibration resolution does not match detections")
        rows: list[ToolInstanceResult] = []
        instance_diagnostics: list[dict[str, Any]] = []
        for instance in detections.instances:
            instance_started = time.perf_counter()
            result = self._estimate_instance(
                instance,
                depth,
                camera,
                support_plane,
                image_bgr,
            )
            instance_latency_ms = (
                time.perf_counter() - instance_started
            ) * 1000.0
            rows.append(result)
            instance_diagnostics.append({
                "frame_local_instance_id": instance.frame_local_instance_id,
                "class_name": instance.class_name,
                "latency_ms": instance_latency_ms,
                "validity": result.validity,
            })
        self.last_runtime_diagnostics = {
            "instance_count": len(rows),
            "instance_latency_ms": instance_diagnostics,
            "maximum_instance_latency_ms": max(
                (
                    item["latency_ms"]
                    for item in instance_diagnostics
                ),
                default=0.0,
            ),
        }
        return ToolFrameResult(
            frame_key=frame_key,
            camera_frame_name=camera.frame_name,
            model_version=detections.model_version,
            ontology_version=detections.ontology_version,
            calibration_version=camera.calibration_version,
            pose_convention_version=self.config.convention_version,
            instances=rows,
        )

    def _invalid(
        self,
        instance: DetectionInstance,
        reason: str,
        depth_ratio: float = 0.0,
        pose_point_count: int = 0,
        anisotropy: float = 0.0,
        sign_confidence: float = 0.0,
    ) -> ToolInstanceResult:
        return ToolInstanceResult(
            frame_local_instance_id=instance.frame_local_instance_id,
            canonical_class_id=instance.canonical_class_id,
            model_class_index=instance.model_class_index,
            class_name=instance.class_name,
            class_confidence=instance.class_confidence,
            bbox_xyxy_px=instance.bbox_xyxy_px,
            mask=instance.mask,
            observation_point_uv_px=None,
            observation_point_selection_mode="",
            observation_point_boundary_clearance_px=0.0,
            position_m=None,
            orientation_xyzw=None,
            pose_mode=POSE_MODE,
            position_valid=False,
            orientation_valid=False,
            validity="INVALID",
            symmetry_type="C2" if instance.class_name == "Army-Navy Retractor" else "NONE",
            endpoint_sign_confidence=sign_confidence,
            valid_depth_ratio=depth_ratio,
            pose_point_count=pose_point_count,
            axis_anisotropy=anisotropy,
            status_flags=(reason,),
            invalid_reason=reason,
        )

    def _estimate_instance(
        self,
        instance: DetectionInstance,
        depth: np.ndarray,
        camera: CameraCalibration,
        plane: SupportPlane,
        image_bgr: np.ndarray | None,
    ) -> ToolInstanceResult:
        mask = instance.mask
        if mask.shape != depth.shape:
            return self._invalid(instance, "MASK_SHAPE_MISMATCH")
        ys, xs = _mask_coordinates_in_bbox(mask, instance.bbox_xyxy_px)
        mask_coordinates = (ys, xs)
        mask_pixels = int(xs.size)
        if mask_pixels < self.config.minimum_mask_pixels:
            return self._invalid(instance, "MASK_TOO_SMALL")
        mask_depth = depth[ys, xs]
        point_count = int(
            np.count_nonzero(np.isfinite(mask_depth) & (mask_depth > 0.0))
        )
        depth_ratio = float(point_count / mask_pixels)
        try:
            endpoint = _pca_endpoints(
                mask,
                self._endpoint_sign_policy(instance.class_name),
                image_bgr=image_bgr,
                mask_coordinates=mask_coordinates,
            )
            anisotropy = float(endpoint["axis_anisotropy"])
            sign_confidence = float(endpoint["sign_confidence"])
            reference = _select_reference_pixel(
                mask,
                endpoint["origin_uv"],
                endpoint["axis_uv"],
                endpoint["axis_length_px"],
                depth,
                mask_coordinates=mask_coordinates,
            )
            origin_ray = _pixel_rays(reference["uv"].reshape(1, 2), camera)[0]
            position = origin_ray * (reference["depth_m"] / origin_ray[2])
            endpoint_rays = _pixel_rays(
                np.stack((endpoint["working_uv"], endpoint["handle_uv"])), camera
            )
            working_3d, handle_3d = _intersect_plane(endpoint_rays, plane)
            y_axis = working_3d - handle_3d
            y_axis -= float(y_axis @ plane.normal) * plane.normal
            y_norm = float(np.linalg.norm(y_axis))
            if y_norm < 1e-8:
                raise ValueError("DEGENERATE_LONGITUDINAL_AXIS")
            y_axis /= y_norm
            z_axis = plane.normal.copy()
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            quaternion = _quaternion_xyzw(np.column_stack((x_axis, y_axis, z_axis)))
        except (ValueError, FloatingPointError) as exc:
            return self._invalid(
                instance,
                str(exc),
                depth_ratio,
                point_count,
                locals().get("anisotropy", 0.0),
                locals().get("sign_confidence", 0.0),
            )

        flags = ["POSITION_IS_MASK_INTERNAL_OBSERVED_SURFACE_POINT"]
        if self.config.positive_y_image_direction != "class_based":
            flags.append(
                "POSITIVE_Y_IMAGE_DIRECTION_"
                f"{self.config.positive_y_image_direction.upper()}"
            )
        if endpoint["sign_source"] == "bovie_external_wire_handle":
            flags.append("BOVIE_EXTERNAL_WIRE_HANDLE")
        elif endpoint["sign_source"] == "bipolar_ensemble":
            flags.append("BIPOLAR_COLOUR_SHAPE_ENSEMBLE")
            if endpoint["sign_evidence"].get("colour_available", False):
                flags.append("BIPOLAR_COLOUR_EVIDENCE_USED")
        elif endpoint["sign_source"] == "adson_triangular_wide_tip":
            flags.append("ADSON_TRIANGULAR_WIDE_TIP")
        elif endpoint["sign_source"] == "adson_two_prong_tip":
            flags.append("ADSON_TWO_PRONG_TIP")
        elif endpoint["sign_source"] == "adson_tip_taper":
            flags.append("ADSON_TIP_TAPER")
        elif endpoint["sign_source"] == "adson_shape_fallback":
            flags.append("ADSON_SHAPE_FALLBACK")
        position_valid = depth_ratio >= self.config.minimum_depth_ratio
        orientation_valid = anisotropy >= self.config.minimum_axis_anisotropy
        if instance.class_name != "Army-Navy Retractor":
            orientation_valid &= sign_confidence >= self.config.minimum_endpoint_sign_confidence
            if sign_confidence < self.config.minimum_endpoint_sign_confidence:
                flags.append("ENDPOINT_SIGN_LOW_CONFIDENCE")
        else:
            flags.append("C2_SYMMETRY_DETERMINISTIC_REPRESENTATIVE")
        if depth_ratio < self.config.minimum_depth_ratio:
            flags.append("REGISTERED_DEPTH_SUPPORT_LOW")
        if anisotropy < self.config.minimum_axis_anisotropy:
            flags.append("MASK_LONGITUDINAL_AXIS_AMBIGUOUS")
        if reference["selection_mode"].startswith("fallback"):
            flags.append("OBSERVATION_POINT_FALLBACK")

        validity = "VALID"
        if not position_valid:
            validity = "INVALID"
        elif not orientation_valid or "OBSERVATION_POINT_FALLBACK" in flags:
            validity = "DEGRADED"
        invalid_reason = "" if validity == "VALID" else ";".join(flags[1:])
        return ToolInstanceResult(
            frame_local_instance_id=instance.frame_local_instance_id,
            canonical_class_id=instance.canonical_class_id,
            model_class_index=instance.model_class_index,
            class_name=instance.class_name,
            class_confidence=instance.class_confidence,
            bbox_xyxy_px=instance.bbox_xyxy_px,
            mask=mask,
            observation_point_uv_px=tuple(float(value) for value in reference["uv"]),
            observation_point_selection_mode=str(reference["selection_mode"]),
            observation_point_boundary_clearance_px=float(reference["boundary_clearance_px"]),
            position_m=tuple(float(value) for value in position),
            orientation_xyzw=tuple(float(value) for value in quaternion),
            pose_mode=POSE_MODE,
            position_valid=position_valid,
            orientation_valid=orientation_valid,
            validity=validity,
            symmetry_type="C2" if instance.class_name == "Army-Navy Retractor" else "NONE",
            endpoint_sign_confidence=sign_confidence,
            valid_depth_ratio=depth_ratio,
            pose_point_count=point_count,
            axis_anisotropy=anisotropy,
            status_flags=tuple(flags),
            invalid_reason=invalid_reason,
            observation_point_depth_m=float(reference["depth_m"]),
        )
