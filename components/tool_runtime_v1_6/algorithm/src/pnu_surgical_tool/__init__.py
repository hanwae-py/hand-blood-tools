"""Standalone surgical-tool recognition and constrained-pose package."""

from .api import SurgicalToolAlgorithm
from .depth_registration import (
    decode_compressed_depth_16uc1,
    DepthRegistrationResult,
    DepthToColorRegistrar,
    rigid_transform_from_realsense_extrinsics,
    RigidTransform,
    validate_rgb_depth_timestamps,
)
from .planar_pose import (
    PlanarPoseEstimator,
    longitudinal_origin_uv,
    sample_depth_at_uv,
)
from .rfdetr_inference import (
    class_agnostic_nms_indices,
    DetectorConfig,
    SurgicalToolDetector,
)
from .visualization import draw_pose_axes_bgr
from .types import (
    CameraCalibration,
    DetectionBatch,
    DetectionInstance,
    SupportPlane,
    ToolFrameResult,
    ToolInstanceResult,
)

__all__ = [
    "CameraCalibration",
    "class_agnostic_nms_indices",
    "draw_pose_axes_bgr",
    "DetectionBatch",
    "DetectionInstance",
    "DetectorConfig",
    "decode_compressed_depth_16uc1",
    "DepthRegistrationResult",
    "DepthToColorRegistrar",
    "PlanarPoseEstimator",
    "longitudinal_origin_uv",
    "sample_depth_at_uv",
    "rigid_transform_from_realsense_extrinsics",
    "RigidTransform",
    "SupportPlane",
    "SurgicalToolAlgorithm",
    "SurgicalToolDetector",
    "ToolFrameResult",
    "ToolInstanceResult",
    "validate_rgb_depth_timestamps",
]

__version__ = "1.6.0rc1-compatible"
