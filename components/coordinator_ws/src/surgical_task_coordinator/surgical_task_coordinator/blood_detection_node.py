#!/usr/bin/env python3
"""ROS 2 lifecycle adapter for the supplied RF-DETR Seg-Small Blood model.

It receives compressed RGB images only.  While ACTIVE it publishes a binary
Blood mask, blue JPEG overlay, and JSON observations.  It intentionally does
not publish a robot suction pose: that requires a separately validated RGB-D
target-selection contract.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
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


class BloodDetectionNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("blood_detection_node")
        self.declare_parameter("color_topic", "/synced/cam_4/color/image_raw/compressed")
        self.declare_parameter("checkpoint", "/home/hanwae/blood/blood_detection.pth")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("optimize", True)
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
        self._mask_pub = None
        self._overlay_pub = None
        self._semantics_pub = None
        self._health_pub = None
        self._diagnostics_pub = None
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("color_topic").value),
            self._on_color,
            reliable_qos(),
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
            self.get_logger().info("ACTIVE: processing RGB frames for Blood masks")
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

    def _on_color(self, message: CompressedImage) -> None:
        if not self._active or self._model is None:
            return
        try:
            data = np.frombuffer(message.data, dtype=np.uint8)
            image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError("failed to decode compressed RGB image")
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
            self._publish_result(message, image_bgr, detections)
        except Exception as exc:
            self._errors += 1
            self._last_error = str(exc)
            self.get_logger().error(f"Blood processing failed: {exc}", throttle_duration_sec=2.0)

    def _publish_result(self, source: CompressedImage, image_bgr: np.ndarray, detections) -> None:
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
            instances.append({
                "instance_id": item_id,
                "class_id": 1,
                "class_name": "blood",
                "confidence": float(confidence),
                "bbox_xyxy_px": [float(value) for value in box],
                "centroid_xy_px": mask_centroid(mask),
                "mask_rle": encode_coco_rle(mask),
            })

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

        payload = {
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
            "instances": instances,
            "combined_blood_mask_rle": encode_coco_rle(union_mask),
        }
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
            "background_is_implicit": True, "last_error": self._last_error,
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
