"""Tests for typed metric tool-pose and 2-D evidence messages."""

from types import SimpleNamespace

import numpy as np

from pnu_surgical_perception.pose_message_mapping import (
    to_pose_and_observation_arrays,
)

from std_msgs.msg import Header

from surgical_perception_msgs.msg import ToolPose


def test_mapping_preserves_metric_pose_and_marks_prior_dof():
    """Map metric position while exposing only the actually observed DoF."""
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 2:7] = True
    item = SimpleNamespace(
        frame_local_instance_id=0,
        canonical_class_id=1,
        model_class_index=0,
        class_name='Scalpel',
        class_confidence=0.9,
        bbox_xyxy_px=(2.0, 2.0, 7.0, 5.0),
        mask=mask,
        observation_point_uv_px=(4.0, 3.0),
        observation_point_selection_mode=(
            'central_longitudinal_band_max_clearance'
        ),
        observation_point_boundary_clearance_px=2.0,
        position_m=(0.1, -0.2, 0.8),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        pose_mode='PLANAR_4DOF_WITH_NORMAL_PRIOR',
        position_valid=True,
        orientation_valid=True,
        validity='VALID',
        symmetry_type='NONE',
        endpoint_sign_confidence=0.8,
        valid_depth_ratio=0.75,
        pose_point_count=12,
        axis_anisotropy=4.0,
        status_flags=('POSITION_IS_MASK_INTERNAL_OBSERVED_SURFACE_POINT',),
        invalid_reason='',
    )
    result = SimpleNamespace(
        model_version='model-v1',
        ontology_version='ontology-v1',
        calibration_version='calibration-v1',
        pose_convention_version='pose-v2',
        instances=[item],
    )
    plane = SimpleNamespace(inlier_ratio=0.7, residual_p95_m=0.004)
    header = Header()
    header.frame_id = 'cam_4_color_optical_frame'

    poses, observations = to_pose_and_observation_arrays(
        result,
        header,
        sequence=3,
        view='cam4',
        support_plane=plane,
    )

    pose = poses.tools[0]
    assert poses.observation_id == 'cam4:3'
    assert poses.header.frame_id == 'cam_4_color_optical_frame'
    position = [
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z,
    ]
    assert position == [
        0.1,
        -0.2,
        0.8,
    ]
    assert pose.pose_mode == (
        ToolPose.POSE_MODE_PLANAR_4DOF_WITH_NORMAL_PRIOR
    )
    assert list(pose.dof_observed) == [True, True, True, False, False, True]
    assert pose.validity == ToolPose.VALIDITY_VALID
    assert observations.instances[0].observation_point_depth_valid
    assert observations.instances[0].mask_counts


def test_mapping_degrades_unverified_metric_configuration():
    """Prevent an unverified scale from being published as fully valid."""
    mask = np.ones((2, 2), dtype=bool)
    item = SimpleNamespace(
        frame_local_instance_id=0,
        canonical_class_id=1,
        model_class_index=0,
        class_name='Scalpel',
        class_confidence=1.0,
        bbox_xyxy_px=(0.0, 0.0, 2.0, 2.0),
        mask=mask,
        observation_point_uv_px=(0.0, 0.0),
        observation_point_selection_mode='test',
        observation_point_boundary_clearance_px=1.0,
        position_m=(0.0, 0.0, 1.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        pose_mode='PLANAR_4DOF_WITH_NORMAL_PRIOR',
        position_valid=True,
        orientation_valid=True,
        validity='VALID',
        symmetry_type='NONE',
        endpoint_sign_confidence=1.0,
        valid_depth_ratio=1.0,
        pose_point_count=4,
        axis_anisotropy=3.0,
        status_flags=(),
        invalid_reason='',
    )
    result = SimpleNamespace(
        model_version='model',
        ontology_version='ontology',
        calibration_version='calibration',
        pose_convention_version='pose',
        instances=[item],
    )
    plane = SimpleNamespace(inlier_ratio=None, residual_p95_m=None)

    poses, _ = to_pose_and_observation_arrays(
        result,
        Header(),
        sequence=1,
        view='cam4',
        support_plane=plane,
        additional_status_flags=('DEPTH_SCALE_UNVERIFIED',),
        degrade_for_additional_flags=True,
    )

    assert poses.tools[0].validity == ToolPose.VALIDITY_DEGRADED
    assert 'DEPTH_SCALE_UNVERIFIED' in poses.tools[0].status_flags
