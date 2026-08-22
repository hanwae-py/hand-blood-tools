"""Publish transparent Debug-only perception layers over a shared camera frame.

The algorithm workers emit complete JPEGs because each is independently useful
outside Debug. This adapter retains the original compressed colour frame and
turns each worker's visual differences into transparent PNG layers. The Debug
UI can compose them on one pixel-aligned camera frame without changing
inference or lifecycle state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage


LAYER_NAMES = ("tool", "pose", "hand", "blood")


def synced_stream_qos() -> QoSProfile:
    """Match VIPLab's reliable /synced image contract."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def stamp_ns(message: CompressedImage) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def decode(message: CompressedImage) -> np.ndarray | None:
    """Decode a compressed frame without raising from a transient bad packet."""
    return cv2.imdecode(
        np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )


@dataclass(frozen=True)
class DecodedFrame:
    message: CompressedImage
    image: np.ndarray
    stamp_ns: int
    received_monotonic: float


class DebugOverlayCompositor(Node):
    """Generate one base JPEG and alpha PNG layers for one colour camera."""

    def __init__(self) -> None:
        super().__init__("debug_overlay_compositor")
        self._declare_parameters()
        self._read_parameters()

        self._base_publisher = self.create_publisher(
            CompressedImage, self._base_topic, qos_profile_sensor_data
        )
        self._layer_publishers = {
            name: self.create_publisher(
                CompressedImage, topic, qos_profile_sensor_data
            )
            for name, topic in self._layer_topics.items()
        }
        self._raw_frames: deque[DecodedFrame] = deque(maxlen=240)
        self._layers: dict[str, deque[DecodedFrame]] = {
            name: deque(maxlen=120) for name in LAYER_NAMES
        }
        self._last_publish_signature: tuple[int, ...] | None = None
        self._last_rejected_layer_log = {name: 0.0 for name in LAYER_NAMES}

        self.create_subscription(
            CompressedImage,
            self._color_topic,
            self._on_color,
            synced_stream_qos(),
        )
        for name, topic in self._input_topics.items():
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, layer=name: self._on_overlay(layer, message),
                qos_profile_sensor_data,
            )
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_latest)
        self.get_logger().info(
            f"Debug compositor ready for {self._camera}: {self._color_topic}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("camera", "cam_4")
        self.declare_parameter("color_topic", "")
        self.declare_parameter("tool_overlay_topic", "")
        self.declare_parameter("pose_overlay_topic", "")
        self.declare_parameter("hand_overlay_topic", "")
        self.declare_parameter("blood_overlay_topic", "")
        self.declare_parameter("base_topic", "")
        self.declare_parameter("tool_layer_topic", "")
        self.declare_parameter("pose_layer_topic", "")
        self.declare_parameter("hand_layer_topic", "")
        self.declare_parameter("blood_layer_topic", "")
        self.declare_parameter("difference_threshold", 36)
        # Every worker preserves the input colour-frame timestamp.  Never
        # borrow a neighbouring video frame for a transparent layer: a moving
        # hand/arm would otherwise be emitted as false foreground "ghosting".
        self.declare_parameter("max_source_delta_ms", 20)
        # Worker overlays include their own status banners.  The Debug UI
        # already reports those states, so exclude the banner strip from the
        # transparent layer rather than covering the video with a black band.
        self.declare_parameter("header_mask_rows", 40)
        # A Tool/Pose/Hand visual should only contain sparse annotations.  A
        # large changed region means its opaque source did not share the same
        # camera pixels as the raw frame, so fail closed for that layer.
        self.declare_parameter("max_annotation_coverage", 0.08)
        self.declare_parameter("publish_rate_hz", 8.0)

    def _read_parameters(self) -> None:
        def value(name: str) -> Any:
            return self.get_parameter(name).value

        self._camera = str(value("camera"))
        output_prefix = f"/perception/{self._camera}/debug"
        self._color_topic = str(value("color_topic")) or (
            f"/synced/{self._camera}/color/image_raw/compressed"
        )
        self._input_topics = {
            "tool": str(value("tool_overlay_topic")) or (
                f"/perception/{self._camera}/tool/overlay/compressed"
            ),
            "pose": str(value("pose_overlay_topic")) or (
                f"/perception/{self._camera}/tool/pose_overlay/compressed"
            ),
            "hand": str(value("hand_overlay_topic")) or (
                f"/perception/{self._camera}/hand/overlay/compressed"
            ),
            "blood": str(value("blood_overlay_topic")) or (
                f"/perception/{self._camera}/blood/overlay/compressed"
            ),
        }
        self._base_topic = str(value("base_topic")) or (
            f"{output_prefix}/base/compressed"
        )
        self._layer_topics = {
            name: str(value(f"{name}_layer_topic")) or (
                f"{output_prefix}/{name}/compressed"
            )
            for name in LAYER_NAMES
        }
        self._difference_threshold = max(
            1, min(255, int(value("difference_threshold")))
        )
        self._max_source_delta_ns = max(
            0, int(float(value("max_source_delta_ms")) * 1_000_000)
        )
        self._header_mask_rows = max(0, int(value("header_mask_rows")))
        self._max_annotation_coverage = max(
            0.001, min(1.0, float(value("max_annotation_coverage")))
        )
        self._publish_rate_hz = max(
            1.0, min(30.0, float(value("publish_rate_hz")))
        )

    def _on_color(self, message: CompressedImage) -> None:
        image = decode(message)
        if image is None:
            return
        self._raw_frames.append(DecodedFrame(
            message=message,
            image=image,
            stamp_ns=stamp_ns(message),
            received_monotonic=time.monotonic(),
        ))

    def _on_overlay(self, name: str, message: CompressedImage) -> None:
        image = decode(message)
        if image is None:
            return
        self._layers[name].append(DecodedFrame(
            message=message,
            image=image,
            stamp_ns=stamp_ns(message),
            received_monotonic=time.monotonic(),
        ))

    def _publish_latest(self) -> None:
        raw = self._target_raw()
        if raw is None:
            return
        overlays = {
            name: self._nearest_frame(self._layers[name], raw.stamp_ns)
            for name in LAYER_NAMES
        }
        signature = (raw.stamp_ns, *(item.stamp_ns if item else -1 for item in overlays.values()))
        if signature == self._last_publish_signature:
            return
        self._last_publish_signature = signature
        self._base_publisher.publish(raw.message)
        for name, publisher in self._layer_publishers.items():
            overlay = overlays[name]
            overlay_base = self._raw_for_stamp(overlay.stamp_ns) if overlay else None
            transparent = self._transparent_layer(
                name,
                overlay_base.image if overlay_base else raw.image,
                overlay.image if overlay and overlay_base else None,
            )
            success, encoded = cv2.imencode(".png", transparent)
            if not success:
                self.get_logger().warning(
                    f"Could not encode {name} transparent Debug layer"
                )
                continue
            output = CompressedImage()
            output.header = raw.message.header
            output.format = "bgra8; png compressed bgra8"
            output.data = encoded.tobytes()
            publisher.publish(output)

    def _target_raw(self) -> DecodedFrame | None:
        """Anchor one composite update to the newest available Tool frame."""
        tool_frames = self._layers["tool"]
        if tool_frames:
            matched = self._raw_for_stamp(tool_frames[-1].stamp_ns)
            if matched is not None:
                return matched
        return self._raw_frames[-1] if self._raw_frames else None

    def _raw_for_stamp(self, stamp: int) -> DecodedFrame | None:
        return self._nearest_frame(self._raw_frames, stamp)

    def _nearest_frame(
        self, frames: deque[DecodedFrame], stamp: int
    ) -> DecodedFrame | None:
        if not frames:
            return None
        candidate = min(frames, key=lambda item: abs(item.stamp_ns - stamp))
        if abs(candidate.stamp_ns - stamp) > self._max_source_delta_ns:
            return None
        return candidate

    def _transparent_layer(
        self, name: str, base: np.ndarray, overlay: np.ndarray | None
    ) -> np.ndarray:
        height, width = base.shape[:2]
        transparent = np.zeros((height, width, 4), dtype=np.uint8)
        if overlay is None or overlay.shape[:2] != (height, width):
            return transparent

        difference = cv2.absdiff(overlay, base).max(axis=2)
        strong = difference >= self._difference_threshold
        # Retain anti-aliased edges only where they touch a definite annotation;
        # normal JPEG background noise therefore stays transparent.
        nearby = difference >= max(1, self._difference_threshold // 2)
        support = cv2.dilate(
            strong.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        ).astype(bool)
        alpha = np.where(strong | (nearby & support), 255, 0).astype(np.uint8)
        alpha[: min(height, self._header_mask_rows), :] = 0

        # Blood can validly cover a large surgical region.  Tool, pose and hand
        # drawings are sparse; reject a broad raw-frame mismatch instead of
        # rendering a second, stale video image as an overlay.
        coverage = float(np.count_nonzero(alpha)) / float(alpha.size)
        if name != "blood" and coverage > self._max_annotation_coverage:
            now = time.monotonic()
            if now - self._last_rejected_layer_log[name] >= 1.0:
                self.get_logger().warning(
                    f"Dropped {name} Debug layer with implausible foreground "
                    f"coverage {coverage:.1%}; source frames were not pixel-aligned"
                )
                self._last_rejected_layer_log[name] = now
            return transparent
        # A transparent pixel must carry zero RGB as well as zero alpha.  Apart
        # from making the layer semantically clean, this keeps the PNG compact
        # enough for rosbridge's non-fragmented binary transport.
        foreground = alpha.astype(bool)
        transparent[:, :, :3] = np.where(
            foreground[:, :, None], overlay, 0
        )
        transparent[:, :, 3] = alpha
        return transparent


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DebugOverlayCompositor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
