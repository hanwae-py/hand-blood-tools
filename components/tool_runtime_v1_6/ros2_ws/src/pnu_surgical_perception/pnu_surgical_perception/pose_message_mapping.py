"""Map ROS-independent surgical-tool pose results to typed ROS messages."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from pnu_surgical_perception.perception_contract import (
    mask_to_compressed_coco_rle_with_geometry,
)

from surgical_perception_msgs.msg import (
    ToolObservation2D,
    ToolObservation2DArray,
    ToolPose,
    ToolPoseArray,
)


POSE_SCHEMA = 'pnu.surgical_tool_pose_array.v1.3'
OBSERVATION_SCHEMA = 'pnu.surgical_tool_observation_2d_array.v1.3'
POINT_DEFINITION = 'mask_internal_depth_valid_observed_surface_point_v1'
AXIS_DEFINITION = (
    '+Y handle/proximal to working tip; +Z support plane to free space; '
    '+X=+Yx+Z'
)


def _mask_bbox(mask: np.ndarray) -> list[float]:
    """Return an exclusive-max bounding box around a binary mask."""
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]


def _point_flags(item: Any) -> tuple[bool, bool, bool]:
    """Return observation-point present, inside-mask, and depth-valid flags."""
    if item.observation_point_uv_px is None:
        return False, False, False
    u, v = (int(round(value)) for value in item.observation_point_uv_px)
    inside_image = 0 <= v < item.mask.shape[0] and 0 <= u < item.mask.shape[1]
    inside_mask = bool(inside_image and item.mask[v, u])
    return True, inside_mask, bool(item.position_valid and inside_mask)


def _pose_mode(item: Any) -> int:
    """Translate the algorithm pose-mode string to the ROS enum."""
    return {
        'PLANAR_4DOF_WITH_NORMAL_PRIOR': (
            ToolPose.POSE_MODE_PLANAR_4DOF_WITH_NORMAL_PRIOR
        ),
        'POSITION_3D_ONLY': ToolPose.POSE_MODE_POSITION_3D_ONLY,
        'FULL_6D': ToolPose.POSE_MODE_FULL_6D,
        'AMBIGUOUS': ToolPose.POSE_MODE_AMBIGUOUS,
    }.get(item.pose_mode, ToolPose.POSE_MODE_INVALID)


def _validity(item: Any) -> int:
    """Translate the algorithm validity string to the ROS enum."""
    return {
        'VALID': ToolPose.VALIDITY_VALID,
        'DEGRADED': ToolPose.VALIDITY_DEGRADED,
        'STALE': ToolPose.VALIDITY_STALE,
    }.get(item.validity, ToolPose.VALIDITY_INVALID)


def _to_pose(
    item: Any,
    support_plane: Any,
    additional_status_flags: tuple[str, ...],
    degrade_for_additional_flags: bool,
) -> ToolPose:
    """Build one typed pose message."""
    message = ToolPose()
    message.frame_local_instance_id = int(item.frame_local_instance_id)
    message.canonical_class_id = int(item.canonical_class_id)
    message.model_class_index = int(item.model_class_index)
    message.class_name = str(item.class_name)
    message.class_confidence = float(item.class_confidence)
    if item.position_m is not None:
        (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ) = item.position_m
    if item.orientation_xyzw is not None:
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = item.orientation_xyzw
    message.pose_mode = _pose_mode(item)
    message.position_valid = bool(item.position_valid)
    message.orientation_valid = bool(item.orientation_valid)
    message.dof_observed = [
        bool(item.position_valid),
        bool(item.position_valid),
        bool(item.position_valid),
        False,
        False,
        bool(item.orientation_valid),
    ]

    point_valid, inside_mask, depth_valid = _point_flags(item)
    message.observation_point_definition = POINT_DEFINITION
    if point_valid:
        message.observation_point_uv_px = list(item.observation_point_uv_px)
    message.observation_point_inside_mask = inside_mask
    message.observation_point_depth_valid = depth_valid
    message.observation_point_selection_mode = str(
        item.observation_point_selection_mode
    )
    message.observation_point_boundary_clearance_px = float(
        item.observation_point_boundary_clearance_px
    )
    message.axis_definition = AXIS_DEFINITION
    message.symmetry_type = str(item.symmetry_type)
    message.endpoint_sign_confidence = float(item.endpoint_sign_confidence)
    message.valid_depth_ratio = float(item.valid_depth_ratio)
    message.pose_point_count = int(item.pose_point_count)
    message.axis_anisotropy = float(item.axis_anisotropy)
    message.support_plane_inlier_ratio = float(
        support_plane.inlier_ratio or 0.0
    )
    message.support_plane_residual_p95_m = float(
        support_plane.residual_p95_m or 0.0
    )
    message.pose_confidence = 0.0
    message.pose_confidence_calibrated = False
    message.validity = _validity(item)
    if (
        degrade_for_additional_flags
        and additional_status_flags
        and message.validity == ToolPose.VALIDITY_VALID
    ):
        message.validity = ToolPose.VALIDITY_DEGRADED
    message.status_flags = list(item.status_flags) + list(
        additional_status_flags
    )
    message.invalid_reason = str(item.invalid_reason)
    return message


def _to_observation(item: Any) -> ToolObservation2D:
    """Build one typed 2-D evidence message."""
    segmentation, geometry = mask_to_compressed_coco_rle_with_geometry(
        item.mask,
        tuple(float(value) for value in item.bbox_xyxy_px),
    )
    message = ToolObservation2D()
    message.frame_local_instance_id = int(item.frame_local_instance_id)
    message.canonical_class_id = int(item.canonical_class_id)
    message.model_class_index = int(item.model_class_index)
    message.class_name = str(item.class_name)
    message.class_confidence = float(item.class_confidence)
    message.segmentation_confidence = float(item.class_confidence)
    message.bbox_xyxy_px = list(item.bbox_xyxy_px)
    message.mask_bbox_xyxy_px = (
        [float(value) for value in geometry['bbox_xyxy_px']]
        if geometry is not None
        else [0.0, 0.0, 0.0, 0.0]
    )
    message.mask_area_px = (
        int(geometry['area_px']) if geometry is not None else 0
    )
    point_valid, inside_mask, depth_valid = _point_flags(item)
    if point_valid:
        message.observation_point_uv_px = list(item.observation_point_uv_px)
    message.observation_point_valid = point_valid
    message.observation_point_inside_mask = inside_mask
    message.observation_point_depth_valid = depth_valid
    depth_m = getattr(item, 'observation_point_depth_m', None)
    message.observation_point_depth_m = (
        float(depth_m) if depth_valid and depth_m is not None else 0.0
    )
    message.observation_point_selection_mode = str(
        item.observation_point_selection_mode
    )
    message.observation_point_boundary_clearance_px = float(
        item.observation_point_boundary_clearance_px
    )
    message.mask_encoding = (
        ToolObservation2D.MASK_ENCODING_COCO_RLE_COMPRESSED
    )
    message.mask_height = int(item.mask.shape[0])
    message.mask_width = int(item.mask.shape[1])
    message.mask_counts = segmentation['counts']
    return message


def to_pose_and_observation_arrays(
    result: Any,
    header: Any,
    sequence: int,
    view: str,
    support_plane: Any,
    additional_status_flags: tuple[str, ...] = (),
    degrade_for_additional_flags: bool = False,
) -> tuple[ToolPoseArray, ToolObservation2DArray]:
    """Convert one algorithm frame result to the two authoritative arrays."""
    observation_id = f'{view}:{sequence}'
    pose_array = to_pose_array_from_result(
        result=result,
        header=header,
        sequence=sequence,
        view=view,
        support_plane=support_plane,
        additional_status_flags=additional_status_flags,
        degrade_for_additional_flags=degrade_for_additional_flags,
    )

    observation_array = ToolObservation2DArray()
    observation_array.header = header
    observation_array.sequence = int(sequence)
    observation_array.schema_version = OBSERVATION_SCHEMA
    observation_array.observation_id = observation_id
    observation_array.view = view
    if result.instances:
        height, width = result.instances[0].mask.shape
        observation_array.image_width = int(width)
        observation_array.image_height = int(height)
    observation_array.model_version = str(result.model_version)
    observation_array.ontology_version = str(result.ontology_version)
    observation_array.instances = [
        _to_observation(item) for item in result.instances
    ]
    return pose_array, observation_array


def to_pose_array_from_result(
    result: Any,
    header: Any,
    sequence: int,
    view: str,
    support_plane: Any,
    additional_status_flags: tuple[str, ...] = (),
    degrade_for_additional_flags: bool = False,
) -> ToolPoseArray:
    """Build only the pose array when 2-D observations already exist."""
    observation_id = f'{view}:{sequence}'
    pose_array = ToolPoseArray()
    pose_array.header = header
    pose_array.sequence = int(sequence)
    pose_array.schema_version = POSE_SCHEMA
    pose_array.observation_id = observation_id
    pose_array.source_view = view
    pose_array.model_version = str(result.model_version)
    pose_array.ontology_version = str(result.ontology_version)
    pose_array.calibration_version = str(result.calibration_version)
    pose_array.pose_convention_version = str(result.pose_convention_version)
    pose_array.tools = [
        _to_pose(
            item,
            support_plane,
            additional_status_flags,
            degrade_for_additional_flags,
        )
        for item in result.instances
    ]
    return pose_array


def to_observation_array_from_detections(
    detections: Any,
    header: Any,
    sequence: int,
    view: str,
    aligned_depth_m: np.ndarray | None = None,
) -> ToolObservation2DArray:
    """Publish 2-D mask evidence from RGB detections.

    Depth is sampled at the existing longitudinal-axis midpoint when an
    aligned depth map is provided. Missing depth skips those fields.
    """
    from pnu_surgical_tool.planar_pose import (
        longitudinal_origin_uv,
        sample_depth_at_uv,
    )

    observation_id = f'{view}:{sequence}'
    observation_array = ToolObservation2DArray()
    observation_array.header = header
    observation_array.sequence = int(sequence)
    observation_array.schema_version = OBSERVATION_SCHEMA
    observation_array.observation_id = observation_id
    observation_array.view = view
    observation_array.image_width = int(detections.image_width)
    observation_array.image_height = int(detections.image_height)
    observation_array.model_version = str(detections.model_version)
    observation_array.ontology_version = str(detections.ontology_version)

    instances = []
    for item in detections.instances:
        origin = longitudinal_origin_uv(
            item.mask,
            item.class_name,
            item.bbox_xyxy_px,
        )
        depth_m = None
        if origin is not None and aligned_depth_m is not None:
            depth_m = sample_depth_at_uv(aligned_depth_m, origin)
        inside_mask = False
        if origin is not None:
            u, v = (int(round(value)) for value in origin)
            height, width = item.mask.shape
            if 0 <= v < height and 0 <= u < width:
                inside_mask = bool(item.mask[v, u])
        mapped = SimpleNamespace(
            frame_local_instance_id=item.frame_local_instance_id,
            canonical_class_id=item.canonical_class_id,
            model_class_index=item.model_class_index,
            class_name=item.class_name,
            class_confidence=item.class_confidence,
            bbox_xyxy_px=item.bbox_xyxy_px,
            mask=item.mask,
            observation_point_uv_px=(
                tuple(float(value) for value in origin)
                if origin is not None
                else None
            ),
            observation_point_selection_mode=(
                'longitudinal_axis_midpoint' if origin is not None else ''
            ),
            observation_point_boundary_clearance_px=0.0,
            position_valid=depth_m is not None,
            observation_point_depth_m=depth_m,
        )
        message = _to_observation(mapped)
        message.observation_point_inside_mask = inside_mask
        instances.append(message)
    observation_array.instances = instances
    return observation_array
