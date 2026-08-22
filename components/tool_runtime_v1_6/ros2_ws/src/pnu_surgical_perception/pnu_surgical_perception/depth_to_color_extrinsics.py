"""Validation for the latched RealSense depth-to-color transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthToColorExtrinsics:
    """A validated transform from a native depth frame into its color frame."""

    rotation: np.ndarray
    translation_m: np.ndarray
    baseline_m: float


def _finite_vector(name: str, values: Any, length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain {length} finite values')
    return vector


def validate_depth_to_color_extrinsics(
    rotation_column_major: Any,
    translation_m: Any,
    *,
    minimum_baseline_m: float,
    maximum_baseline_m: float,
    expected_translation_direction: Any,
    minimum_direction_cosine: float,
    orthonormal_tolerance: float,
    determinant_tolerance: float,
) -> DepthToColorExtrinsics:
    """Validate an ``Extrinsics`` message and convert its column-major R.

    Librealsense publishes the rotation vector in column-major order.  The
    returned matrix is therefore directly suitable for
    ``P_color = R @ P_depth + t``.
    """
    if not (
        np.isfinite(minimum_baseline_m)
        and np.isfinite(maximum_baseline_m)
        and 0.0 < minimum_baseline_m <= maximum_baseline_m
    ):
        raise ValueError('invalid depth-to-color baseline bounds')
    if not (
        np.isfinite(minimum_direction_cosine)
        and -1.0 <= minimum_direction_cosine <= 1.0
    ):
        raise ValueError('minimum_direction_cosine must be in [-1, 1]')
    if not (
        np.isfinite(orthonormal_tolerance)
        and np.isfinite(determinant_tolerance)
        and orthonormal_tolerance > 0.0
        and determinant_tolerance > 0.0
    ):
        raise ValueError('invalid extrinsics rotation tolerances')

    rotation_vector = _finite_vector(
        'depth_to_color rotation', rotation_column_major, 9
    )
    # `realsense2_camera_msgs/Extrinsics.rotation` is explicitly column-major.
    rotation = rotation_vector.reshape((3, 3), order='F')
    translation = _finite_vector(
        'depth-to-color translation', translation_m, 3
    )
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=orthonormal_tolerance,
    ):
        raise ValueError('depth-to-color rotation is not orthonormal')
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > determinant_tolerance:
        raise ValueError(
            f'depth-to-color rotation determinant {determinant:.8f} is invalid'
        )

    baseline_m = float(np.linalg.norm(translation))
    if not minimum_baseline_m <= baseline_m <= maximum_baseline_m:
        raise ValueError(
            f'depth-to-color baseline {baseline_m:.6f} m is outside '
            f'[{minimum_baseline_m:.6f}, {maximum_baseline_m:.6f}] m'
        )
    expected_direction = _finite_vector(
        'expected depth-to-color translation direction',
        expected_translation_direction,
        3,
    )
    expected_norm = float(np.linalg.norm(expected_direction))
    if expected_norm <= 0.0:
        raise ValueError('expected depth-to-color direction must be non-zero')
    cosine = float(
        np.dot(translation / baseline_m, expected_direction / expected_norm)
    )
    if cosine < minimum_direction_cosine:
        raise ValueError(
            f'depth-to-color translation direction cosine {cosine:.6f} '
            f'is below {minimum_direction_cosine:.6f}'
        )

    rotation = rotation.copy()
    translation = translation.copy()
    rotation.setflags(write=False)
    translation.setflags(write=False)
    return DepthToColorExtrinsics(
        rotation=rotation,
        translation_m=translation,
        baseline_m=baseline_m,
    )
