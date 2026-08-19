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
from .planar_pose import PlanarPoseEstimator
from .rfdetr_inference import DetectorConfig, SurgicalToolDetector
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
    "DetectionBatch",
    "DetectionInstance",
    "DetectorConfig",
    "decode_compressed_depth_16uc1",
    "DepthRegistrationResult",
    "DepthToColorRegistrar",
    "PlanarPoseEstimator",
    "rigid_transform_from_realsense_extrinsics",
    "RigidTransform",
    "SupportPlane",
    "SurgicalToolAlgorithm",
    "SurgicalToolDetector",
    "ToolFrameResult",
    "ToolInstanceResult",
    "validate_rgb_depth_timestamps",
]

__version__ = "1.0.0rc1"
