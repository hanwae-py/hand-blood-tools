"""Regression tests for RealSense depth-to-color transform validation."""

from pnu_surgical_perception.depth_to_color_extrinsics import (
    validate_depth_to_color_extrinsics,
)
from pnu_surgical_perception.native_depth_pose_node import (
    depth_to_color_extrinsics_qos,
)

import numpy as np
import pytest
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy


def _validated(rotation, translation=(-0.059, 0.0002, -0.0003)):
    """Validate a plausible D455 transform encoded as a ROS message."""
    return validate_depth_to_color_extrinsics(
        rotation.reshape(-1, order='F'),
        translation,
        minimum_baseline_m=0.02,
        maximum_baseline_m=0.12,
        expected_translation_direction=(-1.0, 0.0, 0.0),
        minimum_direction_cosine=0.95,
        orthonormal_tolerance=1e-6,
        determinant_tolerance=1e-6,
    )


def test_column_major_message_is_converted_to_color_from_depth_matrix():
    """A non-symmetric R must not be silently treated as row-major."""
    theta = np.deg2rad(17.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    extrinsics = _validated(rotation)

    assert np.allclose(extrinsics.rotation, rotation)
    depth_point = np.array((0.10, 0.20, 0.30))
    assert np.allclose(
        extrinsics.rotation @ depth_point + extrinsics.translation_m,
        rotation @ depth_point + np.array((-0.059, 0.0002, -0.0003)),
    )


def test_reflection_is_rejected_even_when_orthonormal():
    """A matrix with determinant -1 must fail rigid-transform validation."""
    with pytest.raises(ValueError, match='determinant'):
        _validated(np.diag((-1.0, 1.0, 1.0)))


def test_impossible_baseline_or_direction_is_rejected():
    """Reject transforms outside the D455 baseline and direction envelope."""
    with pytest.raises(ValueError, match='baseline'):
        _validated(np.eye(3), (-0.25, 0.0, 0.0))
    with pytest.raises(ValueError, match='direction'):
        _validated(np.eye(3), (0.059, 0.0, 0.0))


def test_extrinsics_subscription_uses_latched_reliable_qos():
    """Late subscribers must receive one retained reliable calibration."""
    qos = depth_to_color_extrinsics_qos()
    assert qos.depth == 1
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
