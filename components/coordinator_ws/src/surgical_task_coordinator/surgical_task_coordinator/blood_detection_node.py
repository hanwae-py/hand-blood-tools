#!/usr/bin/env python3
"""ROS 2 lifecycle adapter for the supplied RF-DETR Seg-Small Blood model.

RGB recognition publishes 2D masks always. When a fresh compressedDepth frame
matches the RGB stamp, each mask centroid is sampled for metric depth in the
RGB frame: same HxW is used directly, otherwise native depth is registered
with color/depth CameraInfo and the CAM4 depth-to-color extrinsics. Missing
or unmappable depth skips those fields. Set require_depth to skip frames that
have no usable depth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String


def reliable_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=np.uint8)
    flat = binary.reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in flat:
        current = int(pixel != 0)
        if current == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = current
    counts.append(run_length)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def mask_centroid(mask: np.ndarray) -> list[float] | None:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def sample_centroid_depth_m(
    depth_m: np.ndarray, centroid_xy: list[float] | None
) -> float | None:
    if centroid_xy is None or depth_m.ndim != 2:
        return None
    u = int(round(centroid_xy[0]))
    v = int(round(centroid_xy[1]))
    if v < 0 or u < 0 or v >= depth_m.shape[0] or u >= depth_m.shape[1]:
        return None
    value = float(depth_m[v, u])
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


class BloodDetectionNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("blood_detection_node")
        self.declare_parameter("color_topic", "/synced/cam_4/color/image_raw/compressed")
        self.declare_parameter(
            "depth_topic", "/synced/cam_4/depth/image_rect_raw/compressedDepth"
        )
        self.declare_parameter(
            "color_camera_info_topic", "/synced/cam_4/color/camera_info"
        )
        self.declare_parameter(
            "depth_camera_info_topic", "/synced/cam_4/depth/camera_info"
        )
        self.declare_parameter(
            "checkpoint", str(Path.home() / "models" / "blood_detection.pth")
        )
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("optimize", True)
        self.declare_parameter("require_depth", False)
        self.declare_parameter("maximum_stamp_delta_ns", 1_000_000)
        self.declare_parameter("depth_scale_m_per_unit", 0.001)
        self.declare_parameter("depth_to_color_rotation", [float("nan")] * 9)
        self.declare_parameter("depth_to_color_translation_m", [float("nan")] * 3)
        self.declare_parameter("calibration_version", "")
        self.declare_parameter("mask_topic", "/surgery/perception/cam4/blood_mask")
        self.declare_parameter("overlay_topic", "/surgery/images/cam4/blood_overlay/compressed")
        self.declare_parameter("semantics_topic", "/surgery/perception/cam4/blood/semantics/json")
        self.declare_parameter("health_topic", "/surgery/perception/blood/health")
        self.declare_parameter("diagnostics_topic", "/surgery/perception/blood/diagnostics/json")

        self._active = False
        self._model = None
        self._torch = None
        self._frames_processed = 0
        self._errors = 0
        self._last_process_ms: float | None = None
        self._last_instances = 0
        self._last_error = ""
        self._latest_depth: CompressedImage | None = None
        self._color_info: CameraInfo | None = None
        self._depth_info: CameraInfo | None = None
        self._registrar = None
        self._registrar_key: tuple[object, ...] | None = None
        self._mask_pub = None
        self._overlay_pub = None
        self._semantics_pub = None
        self._health_pub = None
        self._diagnostics_pub = None
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("color_topic").value),
            self._on_color,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("color_camera_info_topic").value),
            self._on_color_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("depth_camera_info_topic").value),
            self._on_depth_info,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info("blood_detection_node created (unconfigured)")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        try:
            threshold = float(self.get_parameter("confidence_threshold").value)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence_threshold must be in [0, 1]")
            checkpoint = str(self.get_parameter("checkpoint").value)
            import torch
            from rfdetr import RFDETRSegSmall

            self.get_logger().info("configuring: loading RF-DETR Seg-Small Blood model")
            model = RFDETRSegSmall.from_checkpoint(checkpoint)
            if bool(self.get_parameter("optimize").value) and torch.cuda.is_available():
                model.optimize_for_inference(
                    compile=True, batch_size=1, dtype=torch.float16, inplace=False
                )
            self._model = model
            self._torch = torch
            self._mask_pub = self.create_lifecycle_publisher(
                Image, str(self.get_parameter("mask_topic").value), reliable_qos(5)
            )
            self._overlay_pub = self.create_lifecycle_publisher(
                CompressedImage, str(self.get_parameter("overlay_topic").value), reliable_qos(5)
            )
            self._semantics_pub = self.create_lifecycle_publisher(
                String, str(self.get_parameter("semantics_topic").value), reliable_qos(5)
            )
            self._health_pub = self.create_lifecycle_publisher(
                String, str(self.get_parameter("health_topic").value), reliable_qos(1)
            )
            self._diagnostics_pub = self.create_lifecycle_publisher(
                String, str(self.get_parameter("diagnostics_topic").value), reliable_qos(1)
            )
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            self.get_logger().info(f"configured on {device}; waiting for activation")
            return TransitionCallbackReturn.SUCCESS
        except Exception as exc:
            self._last_error = str(exc)
            self.get_logger().error(f"Blood configuration failed: {exc}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        result = super().on_activate(state)
        if result == TransitionCallbackReturn.SUCCESS:
            self._active = True
            self.get_logger().info(
                "ACTIVE: processing RGB frames for Blood masks "
                f"(require_depth={bool(self.get_parameter('require_depth').value)})"
            )
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self.get_logger().info(f"INACTIVE after {self._frames_processed} processed frames")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self.get_logger().info("cleaned up: Blood model released")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._model = None
        return TransitionCallbackReturn.SUCCESS

    def _on_depth(self, message: CompressedImage) -> None:
        self._latest_depth = message

    def _on_color_info(self, message: CameraInfo) -> None:
        self._color_info = message
        self._registrar = None
        self._registrar_key = None

    def _on_depth_info(self, message: CameraInfo) -> None:
        self._depth_info = message
        self._registrar = None
        self._registrar_key = None

    def _depth_to_color_registrar(
        self, rgb_height: int, rgb_width: int, native_shape: tuple[int, ...]
    ):
        if len(native_shape) != 2:
            return None
        try:
            from pnu_surgical_tool.depth_registration import (
                finite_vector_or_none,
                registrar_from_camera_messages,
            )
        except ImportError as exc:
            self.get_logger().warn(
                f"Blood depth-to-color registration unavailable: {exc}",
                throttle_duration_sec=10.0,
            )
            return None
        color_info = self._color_info
        depth_info = self._depth_info
        rotation = finite_vector_or_none(
            self.get_parameter("depth_to_color_rotation").value, 9
        )
        translation = finite_vector_or_none(
            self.get_parameter("depth_to_color_translation_m").value, 3
        )
        version = str(self.get_parameter("calibration_version").value).strip()
        if (
            color_info is None
            or depth_info is None
            or rotation is None
            or translation is None
            or not version
        ):
            return None
        if int(color_info.width) != rgb_width or int(color_info.height) != rgb_height:
            return None
        if (int(depth_info.height), int(depth_info.width)) != native_shape:
            return None
        key = (
            int(color_info.width),
            int(color_info.height),
            tuple(np.asarray(color_info.k, dtype=np.float64).ravel()),
            tuple(np.asarray(color_info.d, dtype=np.float64).ravel()),
            str(color_info.header.frame_id),
            int(depth_info.width),
            int(depth_info.height),
            tuple(np.asarray(depth_info.k, dtype=np.float64).ravel()),
            tuple(np.asarray(depth_info.d, dtype=np.float64).ravel()),
            str(depth_info.header.frame_id),
            tuple(rotation.ravel()),
            tuple(translation.ravel()),
            version,
        )
        if self._registrar is not None and self._registrar_key == key:
            return self._registrar
        try:
            registrar = registrar_from_camera_messages(
                color_info, depth_info, rotation, translation, version
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(
                f"Blood depth-to-color registrar skipped: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        self._registrar = registrar
        self._registrar_key = key
        return registrar

    def _aligned_depth_m(
        self, rgb: CompressedImage, height: int, width: int
    ) -> np.ndarray | None:
        depth_msg = self._latest_depth
        if depth_msg is None:
            return None
        maximum_delta_ns = int(self.get_parameter("maximum_stamp_delta_ns").value)
        if abs(stamp_ns(rgb) - stamp_ns(depth_msg)) > maximum_delta_ns:
            return None
        try:
            from pnu_surgical_tool.depth_registration import (
                decode_compressed_depth_16uc1,
                metric_depth_in_rgb_frame,
            )

            native = decode_compressed_depth_16uc1(depth_msg.data, depth_msg.format)
        except Exception as exc:
            self.get_logger().warn(
                f"Blood depth decode skipped: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        registrar = None
        if native.shape != (height, width):
            registrar = self._depth_to_color_registrar(height, width, native.shape)
        aligned = metric_depth_in_rgb_frame(
            native,
            height,
            width,
            float(self.get_parameter("depth_scale_m_per_unit").value),
            registrar,
        )
        if aligned is None and native.shape != (height, width):
            self.get_logger().warn(
                "Blood native depth HxW differs from RGB and could not be "
                "registered; centroid depth skipped",
                throttle_duration_sec=5.0,
            )
        return aligned

    def _on_color(self, message: CompressedImage) -> None:
        if not self._active or self._model is None:
            return
        try:
            data = np.frombuffer(message.data, dtype=np.uint8)
            image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError("failed to decode compressed RGB image")
            height, width = image_bgr.shape[:2]
            depth_m = self._aligned_depth_m(message, height, width)
            if bool(self.get_parameter("require_depth").value) and depth_m is None:
                return
            torch = self._torch
            assert torch is not None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            detections = self._model.predict(
                image_bgr,
                threshold=float(self.get_parameter("confidence_threshold").value),
                include_source_image=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._last_process_ms = (time.perf_counter() - started) * 1000.0
            self._publish_result(message, image_bgr, detections, depth_m)
        except Exception as exc:
            self._errors += 1
            self._last_error = str(exc)
            self.get_logger().error(f"Blood processing failed: {exc}", throttle_duration_sec=2.0)

    def _publish_result(
        self,
        source: CompressedImage,
        image_bgr: np.ndarray,
        detections,
        depth_m: np.ndarray | None,
    ) -> None:
        height, width = image_bgr.shape[:2]
        raw_masks = getattr(detections, "mask", None)
        if raw_masks is None:
            raise RuntimeError("Blood checkpoint returned no segmentation masks")
        union_mask = np.zeros((height, width), dtype=bool)
        overlay = image_bgr.copy()
        instances: list[dict[str, object]] = []
        for item_id, (box, class_id, confidence) in enumerate(
            zip(detections.xyxy, detections.class_id, detections.confidence, strict=True)
        ):
            if int(class_id) != 0:
                raise RuntimeError(f"unexpected Blood class index: {class_id}")
            mask = np.asarray(raw_masks[item_id], dtype=bool)
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            union_mask |= mask
            colored = overlay.copy()
            colored[mask] = (230, 80, 30)  # blue, BGR
            overlay = cv2.addWeighted(overlay, 0.70, colored, 0.30, 0.0)
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (230, 80, 30), 2)
            cv2.putText(overlay, f"blood {float(confidence):.2f}", (x0, max(18, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 80, 30), 2, cv2.LINE_AA)
            centroid = mask_centroid(mask)
            instance: dict[str, object] = {
                "instance_id": item_id,
                "class_id": 1,
                "class_name": "blood",
                "confidence": float(confidence),
                "bbox_xyxy_px": [float(value) for value in box],
                "centroid_xy_px": centroid,
                "mask_rle": encode_coco_rle(mask),
            }
            centroid_depth_m = (
                sample_centroid_depth_m(depth_m, centroid) if depth_m is not None else None
            )
            if centroid_depth_m is not None:
                instance["centroid_depth_m"] = centroid_depth_m
            instances.append(instance)

        mask_msg = Image()
        mask_msg.header = source.header
        mask_msg.height = height
        mask_msg.width = width
        mask_msg.encoding = "mono8"
        mask_msg.is_bigendian = False
        mask_msg.step = width
        mask_msg.data = (union_mask * 255).astype(np.uint8).tobytes()
        self._mask_pub.publish(mask_msg)

        success, encoded = cv2.imencode(".jpg", overlay)
        if not success:
            raise RuntimeError("failed to JPEG encode Blood overlay")
        overlay_msg = CompressedImage()
        overlay_msg.header = source.header
        overlay_msg.format = "jpeg"
        overlay_msg.data = encoded.tobytes()
        self._overlay_pub.publish(overlay_msg)

        union_centroid = mask_centroid(union_mask)
        union_centroid_depth_m = (
            sample_centroid_depth_m(depth_m, union_centroid) if depth_m is not None else None
        )
        payload: dict[str, object] = {
            "schema": "pnu.surgical_blood_observations.v1",
            "header": {
                "stamp_sec": source.header.stamp.sec,
                "stamp_nanosec": source.header.stamp.nanosec,
                "frame_id": source.header.frame_id,
            },
            "image": {"width": width, "height": height},
            "model": "RF-DETR Seg-Small",
            "classes": ["blood"],
            "confidence_threshold": float(self.get_parameter("confidence_threshold").value),
            "inference_latency_ms": self._last_process_ms,
            "depth_sampled": depth_m is not None,
            "instances": instances,
            "combined_blood_mask_rle": encode_coco_rle(union_mask),
            "combined_blood_centroid_xy_px": union_centroid,
        }
        if union_centroid_depth_m is not None:
            payload["combined_blood_centroid_depth_m"] = union_centroid_depth_m
        self._semantics_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self._frames_processed += 1
        self._last_instances = len(instances)
        self.get_logger().info(
            f"published Blood masks: {len(instances)} instances",
            throttle_duration_sec=1.0,
        )

    def _publish_status(self) -> None:
        if self._health_pub is None or self._diagnostics_pub is None:
            return
        state = "active" if self._active else "inactive"
        self._health_pub.publish(String(data=json.dumps({
            "node": self.get_name(), "ready": self._model is not None,
            "lifecycle_state": state, "explicit_classes": ["blood"],
            "background_is_implicit": True,
            "require_depth": bool(self.get_parameter("require_depth").value),
            "last_error": self._last_error,
        })))
        self._diagnostics_pub.publish(String(data=json.dumps({
            "node": self.get_name(), "lifecycle_state": state,
            "frames_processed": self._frames_processed,
            "blood_instances_last_frame": self._last_instances,
            "last_process_ms": self._last_process_ms, "errors": self._errors,
        })))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BloodDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
