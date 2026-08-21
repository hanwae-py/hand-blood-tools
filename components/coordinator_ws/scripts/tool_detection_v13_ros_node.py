#!/usr/bin/env python3
"""ROS 2 lifecycle adapter for Surgical Tool Component v1.3.0-rc1.

The delivered component intentionally owns no ROS subscriptions.  This host
adapter performs exact RGB/depth stamp matching, converts CameraInfo, calls the
component API, and publishes the canonical typed v1.3 outputs.  If aligned
depth, CameraInfo, or a validated support-plane configuration is unavailable,
detection and masks are still published while every pose is explicitly marked
invalid; the adapter never invents a robot-ready pose.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from surgical_perception_msgs.msg import ToolObservation2DArray, ToolPoseArray


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_compressed_depth(message: CompressedImage) -> np.ndarray:
    """Decode the provider's 16UC1 compressedDepth PNG in-process."""
    format_text = str(message.format or "")
    if "compressedDepth" not in format_text or "16UC1" not in format_text:
        raise ValueError(
            "expected 16UC1 compressedDepth; received "
            f"{format_text!r}"
        )
    payload = bytes(message.data)
    png_offset = payload.find(_PNG_SIGNATURE)
    if png_offset < 0:
        raise ValueError("compressedDepth payload has no PNG signature")
    decoded = cv2.imdecode(
        np.frombuffer(payload[png_offset:], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if decoded is None:
        raise ValueError("OpenCV failed to decode compressedDepth PNG")
    if decoded.ndim != 2 or decoded.dtype != np.uint16:
        raise ValueError(
            "compressedDepth must decode to a 2-D uint16 image; "
            f"got shape={decoded.shape}, dtype={decoded.dtype}"
        )
    return np.ascontiguousarray(decoded)


def _stamp_key(message) -> tuple[int, int]:
    return int(message.header.stamp.sec), int(message.header.stamp.nanosec)


class ToolDetectionV13Node(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("tool_detection_node")

        self.declare_parameter(
            "bundle_root",
            str(Path.home() / "models" / "tool_detection_component_v1_3_rc1"),
        )
        self.declare_parameter(
            "image_topic", "/synced/cam_4/color/image_raw/compressed"
        )
        self.declare_parameter("image_transport", "compressed")
        self.declare_parameter(
            "depth_topic",
            "/synced/cam_4/depth/image_rect_raw/compressedDepth",
        )
        self.declare_parameter("depth_transport", "compressed_depth")
        self.declare_parameter("depth_alignment_validated", False)
        self.declare_parameter(
            "camera_info_topic", "/synced/cam_4/color/camera_info"
        )
        self.declare_parameter(
            "pose_topic", "/perception/cam_4/tool/poses"
        )
        self.declare_parameter(
            "observation_topic", "/perception/cam_4/tool/observations"
        )
        self.declare_parameter(
            "semantics_topic", "/perception/cam_4/tool/semantics"
        )
        self.declare_parameter(
            "overlay_topic", "/perception/cam_4/tool/overlay/compressed"
        )
        self.declare_parameter("health_topic", "/perception/cam_4/tool/health")
        self.declare_parameter(
            "diagnostics_topic", "/perception/cam_4/tool/diagnostics"
        )
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("optimize", True)
        self.declare_parameter("jit_compile", True)
        self.declare_parameter("fp16", True)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("overlay_jpeg_quality", 80)
        self.declare_parameter("rgb_wait_timeout_sec", 0.08)
        self.declare_parameter("input_stale_timeout_sec", 2.0)
        self.declare_parameter("camera_info_stale_timeout_sec", 2.0)
        self.declare_parameter("depth_scale_16u", 0.001)
        self.declare_parameter("calibration_version", "")
        self.declare_parameter("support_plane_normal", [0.0, 0.0, 0.0])
        self.declare_parameter("support_plane_offset_m", 0.0)
        self.declare_parameter("support_plane_config_version", "")
        self.declare_parameter("autostart", False)

        self.bridge = CvBridge()
        self.detector = None
        self.pose_estimator = None
        self.support_plane = None
        self._subscriptions = []
        self._active = False
        self._latest_info = None
        self._latest_info_at = None
        self._pending_rgb = {}
        self._pending_depth = {}
        self._last_depth_at = None
        self._last_depth_frame_id = ""
        self._last_source_stamp_sec = 0
        self._last_source_stamp_nanosec = 0
        self._last_source_frame_id = ""
        self._last_observation_id = ""
        self._last_sequence = 0
        self._last_decode_ms = 0.0
        self._last_depth_to_xyz_ms = 0.0
        self._last_pose_ms = 0.0
        self._last_render_encode_ms = 0.0
        self._last_source_to_output_ms = 0.0
        self._last_queue_age_ms = 0.0
        self._dropped_frames = 0
        self._endpoint_sign_low_count = 0
        self._last_model_version = ""
        self._last_error_code = "MODEL_NOT_CONFIGURED"
        self._last_error_message = "tool detector is not configured"
        self._sequence = 0
        self._frames = 0
        self._instances_last = 0
        self._valid_poses_last = 0
        self._last_inference_ms = 0.0
        self._last_total_ms = 0.0
        self._last_frame_at = None
        self._errors = 0
        self._sample_at = time.monotonic()
        self._sample_frames = 0
        self._processed_hz = 0.0
        self._imports = {}
        self.image_transport = str(
            self.get_parameter("image_transport").value
        ).strip().lower()
        self.input_stale_timeout_sec = float(
            self.get_parameter("input_stale_timeout_sec").value
        )
        self.camera_info_stale_timeout_sec = float(
            self.get_parameter("camera_info_stale_timeout_sec").value
        )

        get = self.get_parameter
        observations_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        poses_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        semantics_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        diagnostics_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        overlay_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        health_output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_poses = self.create_lifecycle_publisher(
            ToolPoseArray, get("pose_topic").value, poses_output_qos
        )
        self.pub_observations = self.create_lifecycle_publisher(
            ToolObservation2DArray,
            get("observation_topic").value,
            observations_output_qos,
        )
        self.pub_semantics = self.create_lifecycle_publisher(
            String, get("semantics_topic").value, semantics_output_qos
        )
        self.pub_overlay = self.create_lifecycle_publisher(
            CompressedImage, get("overlay_topic").value, overlay_output_qos
        )
        self.pub_health = self.create_publisher(
            String, get("health_topic").value, health_output_qos
        )
        self.pub_diagnostics = self.create_publisher(
            String, get("diagnostics_topic").value, diagnostics_output_qos
        )
        self.create_timer(0.02, self._flush_unmatched_rgb)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "Tool Component v1.3 lifecycle adapter created (unconfigured)"
        )

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        root = Path(self.get_parameter("bundle_root").value).expanduser().resolve()
        if not root.is_dir():
            self.get_logger().error(f"bundle_root does not exist: {root}")
            return TransitionCallbackReturn.FAILURE

        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from integration_example import tool_result_to_ros as ros_mapper
            from pycocotools import mask as mask_utils
            from pnu_surgical_tool import (
                CameraCalibration,
                DetectorConfig,
                PlanarPoseEstimator,
                SupportPlane,
                SurgicalToolDetector,
                ToolFrameResult,
                ToolInstanceResult,
            )

            def fast_coco_compressed_rle_counts(mask: np.ndarray) -> str:
                """Equivalent COCO RLE using pycocotools' compiled encoder."""
                encoded = mask_utils.encode(
                    np.asfortranarray(np.asarray(mask, dtype=np.uint8))
                )
                counts = encoded["counts"]
                return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)

            # The delivered reference mapper uses a correct but pixel-by-pixel
            # Python encoder. At 1280x720 with many instances that dominated the
            # complete frame time. Preserve its wire format while using the
            # standard compiled COCO encoder already required by this project.
            ros_mapper.coco_compressed_rle_counts = fast_coco_compressed_rle_counts

            self._imports = {
                "CameraCalibration": CameraCalibration,
                "RosFrameMetadata": ros_mapper.RosFrameMetadata,
                "ToolFrameResult": ToolFrameResult,
                "ToolInstanceResult": ToolInstanceResult,
                "to_overlay_message": ros_mapper.to_overlay_message,
                "to_ros_arrays": ros_mapper.to_ros_arrays,
            }
            config = DetectorConfig(
                checkpoint_path=root / "algorithm/model/cam4_rfdetr_seg_small_v1.pth",
                ontology_path=root / "algorithm/model/ontology.json",
                confidence_threshold=float(self.get_parameter("threshold").value),
                optimize=bool(self.get_parameter("optimize").value),
                jit_compile=bool(self.get_parameter("jit_compile").value),
                fp16=bool(self.get_parameter("fp16").value),
            )
            self.detector = SurgicalToolDetector(config)
            self.get_logger().info("loading Surgical Tool Component v1.3 RF-DETR")
            self.detector.load()
            self._last_error_code = ""
            self._last_error_message = ""
            self.pose_estimator = PlanarPoseEstimator()

            normal = np.asarray(
                self.get_parameter("support_plane_normal").value, dtype=np.float64
            )
            offset = float(self.get_parameter("support_plane_offset_m").value)
            plane_version = str(
                self.get_parameter("support_plane_config_version").value
            )
            if normal.shape == (3,) and np.linalg.norm(normal) > 1e-9 and plane_version:
                self.support_plane = SupportPlane(
                    normal=normal,
                    offset_m=offset,
                    config_version=plane_version,
                )
                self.get_logger().info(
                    f"support-plane pose enabled: {plane_version}"
                )
            else:
                self.support_plane = None
                self.get_logger().warning(
                    "support plane is not configured: typed detection/masks will be "
                    "published, but Tool poses will be explicitly INVALID"
                )

            self.image_transport = str(
                self.get_parameter("image_transport").value
            ).strip().lower()
            if self.image_transport not in ("compressed", "raw"):
                raise ValueError(
                    "image_transport must be 'compressed' or 'raw'; received "
                    f"{self.image_transport!r}"
                )
            self.input_stale_timeout_sec = float(
                self.get_parameter("input_stale_timeout_sec").value
            )
            self.camera_info_stale_timeout_sec = float(
                self.get_parameter("camera_info_stale_timeout_sec").value
            )
            self.depth_transport = str(
                self.get_parameter("depth_transport").value
            ).strip().lower()
            if self.depth_transport not in ("compressed_depth", "raw"):
                raise ValueError(
                    "depth_transport must be 'compressed_depth' or 'raw'; "
                    f"received {self.depth_transport!r}"
                )
            self.depth_alignment_validated = bool(
                self.get_parameter("depth_alignment_validated").value
            )
            if self.input_stale_timeout_sec <= 0.0:
                raise ValueError("input_stale_timeout_sec must be > 0")
            if self.camera_info_stale_timeout_sec <= 0.0:
                raise ValueError("camera_info_stale_timeout_sec must be > 0")

            rgb_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=20,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            info_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=20,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            depth_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=20,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            image_type = CompressedImage if self.image_transport == "compressed" else Image
            self._subscriptions = [
                self.create_subscription(
                    image_type,
                    self.get_parameter("image_topic").value,
                    self._on_rgb,
                    rgb_qos,
                ),
                self.create_subscription(
                    (
                        CompressedImage
                        if self.depth_transport == "compressed_depth"
                        else Image
                    ),
                    self.get_parameter("depth_topic").value,
                    self._on_depth,
                    depth_qos,
                ),
                self.create_subscription(
                    CameraInfo,
                    self.get_parameter("camera_info_topic").value,
                    self._on_camera_info,
                    info_qos,
                ),
            ]
        except Exception:
            self.get_logger().error("configuration failed:\n" + traceback.format_exc())
            self._release_model()
            return TransitionCallbackReturn.FAILURE

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.get_logger().info(f"v1.3 configured on {device}; waiting for activation")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result
        self._active = True
        self._frames = 0
        self._sequence = 0
        self._last_depth_at = None
        self._pending_rgb.clear()
        self._pending_depth.clear()
        self._sample_at = time.monotonic()
        self._sample_frames = 0
        self.get_logger().info("ACTIVE: processing v1.3 RGB/depth frames")
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._pending_rgb.clear()
        self._pending_depth.clear()
        self.get_logger().info(f"INACTIVE after {self._frames} processed frames")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        for subscription in self._subscriptions:
            self.destroy_subscription(subscription)
        self._subscriptions = []
        self._latest_info = None
        self._latest_info_at = None
        self._pending_rgb.clear()
        self._pending_depth.clear()
        self._release_model()
        self.get_logger().info("cleaned up: v1.3 RF-DETR model released")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._release_model()
        return TransitionCallbackReturn.SUCCESS

    def _release_model(self) -> None:
        if self.detector is not None:
            self.detector._model = None
        self.detector = None
        self.pose_estimator = None
        self.support_plane = None
        self._imports = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _on_camera_info(self, message: CameraInfo) -> None:
        self._latest_info = message
        self._latest_info_at = time.monotonic()

    def _on_rgb(self, message: Image) -> None:
        if not self._active or self.detector is None:
            return
        key = _stamp_key(message)
        self._pending_rgb[key] = (message, time.monotonic())
        depth = self._pending_depth.pop(key, None)
        if depth is not None:
            rgb, _ = self._pending_rgb.pop(key)
            self._last_queue_age_ms = 0.0
            self._process(rgb, depth)
        self._trim_buffers()

    def _on_depth(self, message: Image | CompressedImage) -> None:
        if not self._active or self.detector is None:
            return
        self._last_depth_at = time.monotonic()
        self._last_depth_frame_id = str(message.header.frame_id)
        key = _stamp_key(message)
        rgb_record = self._pending_rgb.pop(key, None)
        if rgb_record is not None:
            self._last_queue_age_ms = max(
                0.0, (time.monotonic() - rgb_record[1]) * 1000.0
            )
            self._process(rgb_record[0], message)
        else:
            self._pending_depth[key] = message
        self._trim_buffers()

    def _flush_unmatched_rgb(self) -> None:
        if not self._active:
            return
        cutoff = time.monotonic() - float(
            self.get_parameter("rgb_wait_timeout_sec").value
        )
        expired = [key for key, (_, arrived) in self._pending_rgb.items() if arrived < cutoff]
        for key in expired:
            message, arrived = self._pending_rgb.pop(key)
            self._last_queue_age_ms = max(
                0.0, (time.monotonic() - arrived) * 1000.0
            )
            self._process(message, None)

    def _trim_buffers(self) -> None:
        while len(self._pending_rgb) > 5:
            self._pending_rgb.pop(next(iter(self._pending_rgb)))
            self._dropped_frames += 1
        while len(self._pending_depth) > 5:
            self._pending_depth.pop(next(iter(self._pending_depth)))
            self._dropped_frames += 1

    def _depth_metres(
        self, message: Image | CompressedImage | None
    ) -> np.ndarray | None:
        if message is None:
            return None
        if isinstance(message, CompressedImage):
            array = decode_compressed_depth(message)
            encoding = "16UC1"
        else:
            array = np.asarray(
                self.bridge.imgmsg_to_cv2(
                    message, desired_encoding="passthrough"
                )
            )
            encoding = message.encoding
        if encoding == "32FC1":
            depth = array.astype(np.float32, copy=True)
        elif encoding in ("16UC1", "mono16"):
            depth = array.astype(np.float32) * float(
                self.get_parameter("depth_scale_16u").value
            )
        else:
            raise ValueError(f"unsupported depth encoding: {encoding}")
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        return depth

    def _camera(self, frame_id: str):
        if self._latest_info is None:
            return None
        if self._latest_info_at is None or (
            time.monotonic() - self._latest_info_at
            > self.camera_info_stale_timeout_sec
        ):
            return None
        message = self._latest_info
        return self._imports["CameraCalibration"](
            width=int(message.width),
            height=int(message.height),
            k=np.asarray(message.k, dtype=np.float64).reshape(3, 3),
            distortion=np.asarray(message.d, dtype=np.float64),
            frame_name=frame_id,
            calibration_version=str(self.get_parameter("calibration_version").value),
        )

    def _invalid_result(self, detections, frame_id: str, reason: str):
        ToolInstanceResult = self._imports["ToolInstanceResult"]
        items = [
            ToolInstanceResult(
                frame_local_instance_id=item.frame_local_instance_id,
                canonical_class_id=item.canonical_class_id,
                model_class_index=item.model_class_index,
                class_name=item.class_name,
                class_confidence=item.class_confidence,
                bbox_xyxy_px=item.bbox_xyxy_px,
                mask=item.mask,
                observation_point_uv_px=None,
                observation_point_selection_mode="",
                observation_point_boundary_clearance_px=0.0,
                position_m=None,
                orientation_xyzw=None,
                pose_mode="INVALID",
                position_valid=False,
                orientation_valid=False,
                validity="INVALID",
                symmetry_type="",
                endpoint_sign_confidence=0.0,
                valid_depth_ratio=0.0,
                pose_point_count=0,
                axis_anisotropy=0.0,
                status_flags=(reason,),
                invalid_reason=reason,
            )
            for item in detections.instances
        ]
        return self._imports["ToolFrameResult"](
            frame_key=f"cam_4:{self._sequence}",
            camera_frame_name=frame_id,
            model_version=detections.model_version,
            ontology_version=detections.ontology_version,
            calibration_version=str(self.get_parameter("calibration_version").value),
            pose_convention_version="cam4_planar_pose_convention_v2",
            instances=items,
        )

    def _process(
        self,
        rgb_message: Image | CompressedImage,
        depth_message: Image | CompressedImage | None,
    ) -> None:
        try:
            started = time.perf_counter()
            self._last_source_stamp_sec = int(rgb_message.header.stamp.sec)
            self._last_source_stamp_nanosec = int(rgb_message.header.stamp.nanosec)
            self._last_source_frame_id = str(rgb_message.header.frame_id)
            self._last_sequence = self._sequence
            self._last_observation_id = f"cam_4:{self._sequence}"
            decode_started = time.perf_counter()
            if isinstance(rgb_message, CompressedImage):
                image_bgr = self.bridge.compressed_imgmsg_to_cv2(
                    rgb_message, desired_encoding="bgr8"
                )
            else:
                image_bgr = self.bridge.imgmsg_to_cv2(
                    rgb_message, desired_encoding="bgr8"
                )
            self._last_decode_ms = (time.perf_counter() - decode_started) * 1000.0
            detections = self.detector.predict(
                np.asarray(image_bgr, dtype=np.uint8),
                color_order="BGR",
                confidence_threshold=float(self.get_parameter("threshold").value),
            )
            self._last_inference_ms = float(detections.inference_latency_ms or 0.0)
            self._last_model_version = str(detections.model_version)
            frame_id = str(rgb_message.header.frame_id or "cam_4_color_optical_frame")
            depth_started = time.perf_counter()
            depth = (
                self._depth_metres(depth_message)
                if self.depth_alignment_validated
                else None
            )
            self._last_depth_to_xyz_ms = (
                time.perf_counter() - depth_started
            ) * 1000.0
            camera = self._camera(frame_id)

            pose_started = time.perf_counter()
            if not self.depth_alignment_validated:
                result = self._invalid_result(
                    detections, frame_id, "DEPTH_ALIGNMENT_NOT_VALIDATED"
                )
            elif self.support_plane is None:
                result = self._invalid_result(
                    detections, frame_id, "SUPPORT_PLANE_NOT_CONFIGURED"
                )
            elif depth is None:
                result = self._invalid_result(detections, frame_id, "ALIGNED_DEPTH_MISSING")
            elif camera is None:
                result = self._invalid_result(detections, frame_id, "CAMERA_INFO_MISSING")
            elif depth.shape != image_bgr.shape[:2]:
                result = self._invalid_result(detections, frame_id, "DEPTH_SHAPE_MISMATCH")
            else:
                result = self.pose_estimator.estimate(
                    detections,
                    depth,
                    camera,
                    self.support_plane,
                    frame_key=f"cam_4:{self._sequence}",
                )
            self._last_pose_ms = (time.perf_counter() - pose_started) * 1000.0

            metadata = self._imports["RosFrameMetadata"](
                stamp=rgb_message.header.stamp,
                frame_id=frame_id,
                sequence=self._sequence,
                observation_id=f"cam_4:{self._sequence}",
                view="cam_4",
            )
            pose_array, observation_array = self._imports["to_ros_arrays"](
                result, metadata, self.support_plane
            )
            observation_array.image_width = int(detections.image_width)
            observation_array.image_height = int(detections.image_height)
            self.pub_poses.publish(pose_array)
            self.pub_observations.publish(observation_array)

            compatibility = {
                "schema": "pnu.surgical.tool.compatibility.v1.3",
                "header": {
                    "stamp_sec": int(rgb_message.header.stamp.sec),
                    "stamp_nanosec": int(rgb_message.header.stamp.nanosec),
                    "frame_id": frame_id,
                },
                "sequence": self._sequence,
                "observation_id": f"cam_4:{self._sequence}",
                "inference_latency_ms": self._last_inference_ms,
                "instances": [
                    {
                        "frame_local_instance_id": item.frame_local_instance_id,
                        "canonical_class_id": item.canonical_class_id,
                        "model_class_index": item.model_class_index,
                        "class_name": item.class_name,
                        "confidence": item.class_confidence,
                        "bbox_xyxy_px": list(item.bbox_xyxy_px),
                        "pose_valid": bool(item.position_valid and item.orientation_valid),
                    }
                    for item in result.instances
                ],
            }
            self.pub_semantics.publish(
                String(data=json.dumps(compatibility, separators=(",", ":")))
            )

            render_started = time.perf_counter()
            if bool(self.get_parameter("publish_overlay").value):
                overlay = self._imports["to_overlay_message"](
                    image_bgr,
                    result,
                    metadata,
                    jpeg_quality=int(self.get_parameter("overlay_jpeg_quality").value),
                )
                self.pub_overlay.publish(overlay)
            self._last_render_encode_ms = (
                time.perf_counter() - render_started
            ) * 1000.0

            self._frames += 1
            self._sequence += 1
            self._instances_last = len(result.instances)
            self._valid_poses_last = sum(
                bool(item.position_valid and item.orientation_valid)
                for item in result.instances
            )
            self._endpoint_sign_low_count = sum(
                float(item.endpoint_sign_confidence) < 0.5
                for item in result.instances
            )
            self._last_total_ms = (time.perf_counter() - started) * 1000.0
            self._last_frame_at = time.monotonic()
            output_ns = int(self.get_clock().now().nanoseconds)
            source_ns = (
                self._last_source_stamp_sec * 1_000_000_000
                + self._last_source_stamp_nanosec
            )
            self._last_source_to_output_ms = max(
                0.0, (output_ns - source_ns) / 1_000_000.0
            )
            self._last_error_code = ""
            self._last_error_message = ""
            self.get_logger().info(
                f"published v1.3 tools: {len(result.instances)} instances; "
                f"valid_poses={self._valid_poses_last}",
                throttle_duration_sec=1.0,
            )
        except Exception:
            self._errors += 1
            self._dropped_frames += 1
            self._last_error_code = "PROCESSING_ERROR"
            self._last_error_message = traceback.format_exc().splitlines()[-1]
            self.get_logger().error("v1.3 processing failed:\n" + traceback.format_exc())

    def _state_name(self) -> str:
        try:
            return self._state_machine.current_state[1]
        except Exception:
            return "unknown"

    def _publish_status(self) -> None:
        if not rclpy.ok():
            return
        now = time.monotonic()
        elapsed = now - self._sample_at
        if elapsed > 0:
            self._processed_hz = (self._frames - self._sample_frames) / elapsed
            self._sample_at = now
            self._sample_frames = self._frames
        stale = None if self._last_frame_at is None else now - self._last_frame_at
        input_fresh = stale is not None and stale < self.input_stale_timeout_sec
        camera_info_age = (
            None
            if self._latest_info_at is None
            else now - self._latest_info_at
        )
        camera_info_fresh = (
            camera_info_age is not None
            and camera_info_age < self.camera_info_stale_timeout_sec
        )
        model_ready = self.detector is not None
        rgb_ready = input_fresh
        depth_topic = str(self.get_parameter("depth_topic").value)
        # count_publishers(), not "is the name in the graph": the Taskplanner's
        # cv_contract_monitor already subscribes to the aligned-depth name on
        # the integration LAN, so the topic exists with zero publishers until
        # the provider ships it.
        depth_publisher_seen = self.count_publishers(depth_topic) > 0
        depth_stream_ready = (
            depth_publisher_seen
            and self._last_depth_at is not None
            and now - self._last_depth_at < self.input_stale_timeout_sec
        )
        depth_alignment_validated = bool(
            self.get_parameter("depth_alignment_validated").value
        )
        depth_ready = bool(
            depth_stream_ready and depth_alignment_validated)
        calibration_version = str(
            self.get_parameter("calibration_version").value
        ).strip()
        cam4_calibration_ready = bool(
            camera_info_fresh and calibration_version
        )
        cam4_pose_ready = bool(
            self._active
            and rgb_ready
            and camera_info_fresh
            and depth_ready
            and cam4_calibration_ready
            and model_ready
            and self.support_plane is not None
        )
        error_code = self._last_error_code
        error_message = self._last_error_message
        if not error_code:
            if not model_ready:
                error_code = "MODEL_NOT_CONFIGURED"
                error_message = "tool detector model is not configured"
            elif not rgb_ready:
                error_code = "CAM4_RGB_STALE_OR_MISSING"
                error_message = "fresh CAM4 RGB has not been processed"
            elif not camera_info_fresh:
                error_code = "CAM4_CAMERA_INFO_STALE_OR_MISSING"
                error_message = "fresh CAM4 CameraInfo is unavailable"
            elif not depth_ready:
                if not depth_publisher_seen:
                    error_code = "CAM4_ALIGNED_DEPTH_NOT_PROVIDED"
                    error_message = f"no publisher on {depth_topic}"
                elif depth_stream_ready and not depth_alignment_validated:
                    error_code = "CAM4_DEPTH_ALIGNMENT_NOT_VALIDATED"
                    error_message = (
                        "compressed depth is received but RGB/depth alignment "
                        "and calibration are not approved"
                    )
                else:
                    error_code = "CAM4_DEPTH_STALE_OR_MISSING"
                    error_message = "fresh CAM4 aligned depth is unavailable"
            elif not cam4_calibration_ready:
                error_code = "CAM4_CALIBRATION_NOT_CONFIGURED"
                error_message = "validated CAM4 calibration version is unavailable"
            elif self.support_plane is None:
                error_code = "CAM4_SUPPORT_PLANE_NOT_CONFIGURED"
                error_message = "validated CAM4 support plane is unavailable"
        stamp_ns = int(self.get_clock().now().nanoseconds)
        health = {
            "schema": "pnu.rfdetr_health.v2",
            "stamp_sec": stamp_ns // 1_000_000_000,
            "stamp_nanosec": stamp_ns % 1_000_000_000,
            "node": self.get_name(),
            "state": self._state_name(),
            "cam4_rgb_ready": rgb_ready,
            "cam4_camera_info_ready": camera_info_fresh,
            "cam4_depth_ready": depth_ready,
            "cam4_depth_stream_ready": depth_stream_ready,
            "cam4_depth_alignment_validated": depth_alignment_validated,
            "cam4_calibration_ready": cam4_calibration_ready,
            "cam4_pose_ready": cam4_pose_ready,
            "tray_rgb_ready": False,
            "tray_camera_info_ready": False,
            "tray_depth_ready": False,
            "tray_calibration_ready": False,
            "tray_model_ready": False,
            "tray_pose_ready": False,
            "model_ready": model_ready,
            "last_error_code": error_code,
            "last_error_message": error_message,
        }
        diagnostics = {
            "schema": "pnu.rfdetr_diagnostics.v2",
            "view": "cam_4",
            "source_stamp_sec": self._last_source_stamp_sec,
            "source_stamp_nanosec": self._last_source_stamp_nanosec,
            "frame_id": self._last_source_frame_id,
            "observation_id": self._last_observation_id,
            "sequence": self._last_sequence,
            "decode_latency_ms": round(self._last_decode_ms, 3),
            "depth_to_xyz_latency_ms": round(self._last_depth_to_xyz_ms, 3),
            "inference_latency_ms": round(self._last_inference_ms, 3),
            "pose_latency_ms": round(self._last_pose_ms, 3),
            "render_encode_latency_ms": round(self._last_render_encode_ms, 3),
            "source_to_output_latency_ms": round(
                self._last_source_to_output_ms, 3
            ),
            "queue_age_ms": round(self._last_queue_age_ms, 3),
            "dropped_frames": self._dropped_frames,
            "instance_count": self._instances_last,
            "valid_pose_count": self._valid_poses_last,
            "endpoint_sign_low_count": self._endpoint_sign_low_count,
            "model_version": self._last_model_version,
            "calibration_version": calibration_version,
            "depth_topic": depth_topic,
            "depth_transport": self.get_parameter("depth_transport").value,
            "depth_frame_id": self._last_depth_frame_id,
            "depth_alignment_validated": depth_alignment_validated,
            "error_code": error_code,
            "error_message": error_message,
        }
        try:
            self.pub_health.publish(String(data=json.dumps(health)))
            self.pub_diagnostics.publish(String(data=json.dumps(diagnostics)))
        except Exception:
            if rclpy.ok():
                raise


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ToolDetectionV13Node()
        if node.get_parameter("autostart").value:
            node.trigger_configure()
            node.trigger_activate()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
