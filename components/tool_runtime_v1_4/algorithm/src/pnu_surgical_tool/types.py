"""Public, ROS-independent data types for the algorithm boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    k: np.ndarray
    distortion: np.ndarray
    frame_name: str
    calibration_version: str

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=np.float64)
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise ValueError("camera.k must be a finite 3x3 matrix")
        if not np.all(np.isfinite(distortion)):
            raise ValueError("camera.distortion must be finite")
        if not self.frame_name or not self.calibration_version:
            raise ValueError("frame_name and calibration_version are required")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "distortion", distortion)


@dataclass(frozen=True)
class SupportPlane:
    normal: np.ndarray
    offset_m: float
    config_version: str
    inlier_ratio: float | None = None
    residual_p95_m: float | None = None

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64).reshape(-1)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("support-plane normal must be a finite 3-vector")
        length = float(np.linalg.norm(normal))
        if length < 1e-9 or not np.isfinite(self.offset_m):
            raise ValueError("support plane is degenerate")
        if not self.config_version:
            raise ValueError("support-plane config_version is required")
        object.__setattr__(self, "normal", normal / length)
        object.__setattr__(self, "offset_m", float(self.offset_m) / length)


@dataclass
class DetectionInstance:
    frame_local_instance_id: int
    canonical_class_id: int
    model_class_index: int
    class_name: str
    class_confidence: float
    bbox_xyxy_px: tuple[float, float, float, float]
    mask: np.ndarray

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError("instance mask must be HxW")


@dataclass
class DetectionBatch:
    image_width: int
    image_height: int
    model_version: str
    ontology_version: str
    instances: list[DetectionInstance]
    inference_latency_ms: float | None = None


@dataclass
class ToolInstanceResult:
    frame_local_instance_id: int
    canonical_class_id: int
    model_class_index: int
    class_name: str
    class_confidence: float
    bbox_xyxy_px: tuple[float, float, float, float]
    mask: np.ndarray
    observation_point_uv_px: tuple[float, float] | None
    observation_point_selection_mode: str
    observation_point_boundary_clearance_px: float
    position_m: tuple[float, float, float] | None
    orientation_xyzw: tuple[float, float, float, float] | None
    pose_mode: str
    position_valid: bool
    orientation_valid: bool
    validity: str
    symmetry_type: str
    endpoint_sign_confidence: float
    valid_depth_ratio: float
    pose_point_count: int
    axis_anisotropy: float
    status_flags: tuple[str, ...]
    invalid_reason: str


@dataclass
class ToolFrameResult:
    frame_key: str | int | None
    camera_frame_name: str
    model_version: str
    ontology_version: str
    calibration_version: str
    pose_convention_version: str
    instances: list[ToolInstanceResult]


def _json_safe(value: Any, include_masks: bool) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == bool and not include_masks:
            return None
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        scalar = value.item()
        return scalar if not isinstance(scalar, float) or np.isfinite(scalar) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item, include_masks) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, include_masks) for item in value]
    return value


def result_to_dict(result: ToolFrameResult, include_masks: bool = False) -> dict[str, Any]:
    """Convert a result to JSON-safe data. Masks are omitted unless requested."""
    return _json_safe(asdict(result), include_masks=include_masks)
