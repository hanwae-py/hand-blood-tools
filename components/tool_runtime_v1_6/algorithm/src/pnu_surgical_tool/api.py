"""Convenience facade combining detection and constrained pose."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .depth_registration import DepthRegistrationResult, DepthToColorRegistrar
from .planar_pose import PlanarPoseEstimator
from .rfdetr_inference import SurgicalToolDetector
from .types import CameraCalibration, DetectionBatch, SupportPlane, ToolFrameResult


class SurgicalToolAlgorithm:
    def __init__(
        self,
        detector: SurgicalToolDetector,
        pose_estimator: PlanarPoseEstimator | None = None,
    ) -> None:
        self.detector = detector
        self.pose_estimator = pose_estimator or PlanarPoseEstimator()

    def detect(
        self,
        image: np.ndarray,
        color_order: Literal["RGB", "BGR"],
        confidence_threshold: float | None = None,
    ) -> DetectionBatch:
        return self.detector.predict(image, color_order, confidence_threshold)

    def detect_and_estimate(
        self,
        image: np.ndarray,
        aligned_depth_m: np.ndarray,
        camera: CameraCalibration,
        support_plane: SupportPlane,
        color_order: Literal["RGB", "BGR"],
        frame_key: str | int | None = None,
        confidence_threshold: float | None = None,
    ) -> ToolFrameResult:
        detections = self.detect(image, color_order, confidence_threshold)
        return self.pose_estimator.estimate(
            detections,
            aligned_depth_m,
            camera,
            support_plane,
            frame_key=frame_key,
        )

    def detect_and_estimate_from_native_depth(
        self,
        image: np.ndarray,
        native_depth: np.ndarray,
        depth_registrar: DepthToColorRegistrar,
        depth_scale_m_per_unit: float,
        support_plane: SupportPlane,
        color_order: Literal["RGB", "BGR"],
        frame_key: str | int | None = None,
        confidence_threshold: float | None = None,
        minimum_depth_m: float = 0.05,
        maximum_depth_m: float = 10.0,
    ) -> tuple[ToolFrameResult, DepthRegistrationResult]:
        """Register native depth to RGB pixels, then run detection and pose."""

        if image.shape[:2] != (
            depth_registrar.color_camera.height,
            depth_registrar.color_camera.width,
        ):
            raise ValueError("image shape does not match registrar color camera")
        registration = depth_registrar.register(
            native_depth,
            depth_scale_m_per_unit,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
        )
        result = self.detect_and_estimate(
            image=image,
            aligned_depth_m=registration.aligned_depth_m,
            camera=depth_registrar.color_camera,
            support_plane=support_plane,
            color_order=color_order,
            frame_key=frame_key,
            confidence_threshold=confidence_threshold,
        )
        return result, registration
