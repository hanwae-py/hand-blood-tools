"""Standalone surgical-tool recognition and constrained-pose package."""

from typing import Any

from .depth_registration import (
    decode_compressed_depth_16uc1,
    DepthRegistrationResult,
    DepthToColorRegistrar,
    finite_vector_or_none,
    metric_depth_in_rgb_frame,
    registrar_from_camera_fields,
    registrar_from_camera_messages,
    rigid_transform_from_realsense_extrinsics,
    RigidTransform,
    validate_rgb_depth_timestamps,
)
from .detection_postprocessing import (
    DetectionPostprocessor,
    DetectionPostprocessorConfig,
    SmallComponentCleanupConfig,
    TemporalClassConfig,
    WorkspaceRoiConfig,
)
from .planar_pose import (
    PlanarPoseConfig,
    PlanarPoseEstimator,
    longitudinal_origin_uv,
    sample_depth_at_uv,
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
    "DetectionPostprocessor",
    "DetectionPostprocessorConfig",
    "SmallComponentCleanupConfig",
    "DetectorConfig",
    "decode_compressed_depth_16uc1",
    "DepthRegistrationResult",
    "DepthToColorRegistrar",
    "finite_vector_or_none",
    "PlanarPoseEstimator",
    "PlanarPoseConfig",
    "longitudinal_origin_uv",
    "metric_depth_in_rgb_frame",
    "registrar_from_camera_fields",
    "registrar_from_camera_messages",
    "sample_depth_at_uv",
    "rigid_transform_from_realsense_extrinsics",
    "RigidTransform",
    "SupportPlane",
    "SurgicalToolAlgorithm",
    "SurgicalToolDetector",
    "ToolFrameResult",
    "ToolInstanceResult",
    "TemporalClassConfig",
    "validate_rgb_depth_timestamps",
    "WorkspaceRoiConfig",
]

__version__ = "1.6.0rc1-compatible"

_LAZY_ATTRS = {
    "SurgicalToolAlgorithm": (".api", "SurgicalToolAlgorithm"),
    "DetectorConfig": (".rfdetr_inference", "DetectorConfig"),
    "SurgicalToolDetector": (".rfdetr_inference", "SurgicalToolDetector"),
    "class_agnostic_nms_indices": (".rfdetr_inference", "class_agnostic_nms_indices"),
}


def __getattr__(name: str) -> Any:
    """Load detector/API symbols only when needed so Hand can import registration."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
