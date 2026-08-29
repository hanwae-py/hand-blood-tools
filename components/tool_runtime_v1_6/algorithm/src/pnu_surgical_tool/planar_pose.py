"""Mask/depth-based planar surgical-tool pose estimator.

The returned quaternion is a transport representation of a constrained pose:
translation comes from an observed depth-valid mask pixel, heading comes from
the mask's longitudinal axis, and the remaining orientation comes from the
supplied support-plane normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


def _pca_endpoints(mask: np.ndarray, sign_policy: str) -> dict[str, Any]:
    ys, xs = np.where(mask)
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

    if sign_policy == "cam4_positive_axis":
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
    elif sign_policy in ("adson_tip_taper", "bipolar_connector_taper"):
        low_taper, high_taper = _terminal_taper_scores(projection, low, high)
        taper_strength = max(low_taper, high_taper)
        if abs(low_taper - high_taper) < 1e-6:
            # Both tools use the larger terminal mass as the handle fallback.
            # Confidence remains zero because taper did not disambiguate it.
            if low_mass > high_mass:
                direction *= -1
                low, high = -high, -low
                low_mass, high_mass = high_mass, low_mass
            confidence = 0.0
        else:
            stronger_taper_is_handle = sign_policy == "bipolar_connector_taper"
            should_flip = (
                low_taper > high_taper
                if stronger_taper_is_handle
                else high_taper > low_taper
            )
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
    }


def longitudinal_origin_uv(
    mask: np.ndarray, class_name: str
) -> np.ndarray | None:
    """Return the mask longitudinal-axis midpoint, or None if the mask is unusable."""
    try:
        return _pca_endpoints(mask, _sign_policy(class_name))["origin_uv"]
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
) -> dict[str, Any]:
    ys, xs = np.where(mask)
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
        return "smaller_end_is_handle"
    if class_name == "Bipolar Forceps":
        return "bipolar_connector_taper"
    return "larger_end_is_handle"


@dataclass(frozen=True)
class PlanarPoseConfig:
    convention_version: str = "pnu.cam4.planar_tool_pose_convention.v2"
    minimum_mask_pixels: int = 20
    minimum_depth_ratio: float = 0.05
    minimum_axis_anisotropy: float = 2.0
    minimum_endpoint_sign_confidence: float = 0.20


class PlanarPoseEstimator:
    def __init__(self, config: PlanarPoseConfig | None = None) -> None:
        self.config = config or PlanarPoseConfig()

    def estimate(
        self,
        detections: DetectionBatch,
        aligned_depth_m: np.ndarray,
        camera: CameraCalibration,
        support_plane: SupportPlane,
        frame_key: str | int | None = None,
    ) -> ToolFrameResult:
        depth = np.asarray(aligned_depth_m)
        expected = (detections.image_height, detections.image_width)
        if depth.shape != expected:
            raise ValueError(f"aligned_depth_m shape {depth.shape} != image shape {expected}")
        if depth.dtype not in (np.float32, np.float64):
            raise TypeError("aligned_depth_m must use float32 or float64 metres")
        if (camera.width, camera.height) != (
            detections.image_width,
            detections.image_height,
        ):
            raise ValueError("camera calibration resolution does not match detections")
        rows = [
            self._estimate_instance(instance, depth, camera, support_plane)
            for instance in detections.instances
        ]
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
    ) -> ToolInstanceResult:
        mask = instance.mask
        if mask.shape != depth.shape:
            return self._invalid(instance, "MASK_SHAPE_MISMATCH")
        mask_pixels = int(mask.sum())
        if mask_pixels < self.config.minimum_mask_pixels:
            return self._invalid(instance, "MASK_TOO_SMALL")
        valid_depth = mask & np.isfinite(depth) & (depth > 0.0)
        point_count = int(valid_depth.sum())
        depth_ratio = float(point_count / mask_pixels)
        try:
            endpoint = _pca_endpoints(mask, _sign_policy(instance.class_name))
            anisotropy = float(endpoint["axis_anisotropy"])
            sign_confidence = float(endpoint["sign_confidence"])
            reference = _select_reference_pixel(
                mask,
                endpoint["origin_uv"],
                endpoint["axis_uv"],
                endpoint["axis_length_px"],
                depth,
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
