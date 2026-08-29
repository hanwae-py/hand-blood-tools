"""Operator 2x2 monitor: existing CAM3/CAM4, CAM1 Hand, and FLIR Blood."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


def image_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def status_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def decode_jpeg(message: CompressedImage) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
    return image if image is not None and image.ndim == 3 else None


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image must be BGR HxWx3')
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))),
         max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def low_light_metrics(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p01, p99 = np.percentile(gray, [1.0, 99.0])
    return float(p99), float(p99 - p01)


def _outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA)


def placeholder(width: int, height: int, label: str) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    _outlined_text(panel, label, (28, 64), 0.9, (45, 190, 255), 2)
    return panel


@dataclass
class LatestImage:
    message: CompressedImage | None = None
    image: np.ndarray | None = None
    received_monotonic: float | None = None
    source_stamp_ns: int | None = None
    received: int = 0
    dropped: int = 0


class OperatorQuadCompositor(Node):
    def __init__(self, *, context: Context | None = None) -> None:
        super().__init__('operator_quad_compositor', context=context)
        self.declare_parameter(
            'top_overlay_topic', '/perception/debug/final_overlay/compressed')
        self.declare_parameter(
            'cam1_hand_overlay_topic', '/perception/cam_1/hand/overlay/compressed')
        self.declare_parameter(
            'flir_blood_overlay_topic', '/perception/flir/blood/overlay/compressed')
        self.declare_parameter(
            'flir_raw_topic', '/perception/ingress/flir/color/image_raw/compressed')
        self.declare_parameter(
            'fusion_status_topic', '/perception/hand/fused/status')
        self.declare_parameter(
            'output_topic', '/perception/debug/operator_quad/compressed')
        self.declare_parameter(
            'status_topic', '/perception/debug/operator_quad/status')
        self.declare_parameter('panel_width', 960)
        self.declare_parameter('panel_height', 540)
        self.declare_parameter('output_rate_hz', 10.0)
        self.declare_parameter('jpeg_quality', 92)
        self.declare_parameter('top_max_age_sec', 1.0)
        self.declare_parameter('cam1_max_age_sec', 0.5)
        self.declare_parameter('flir_blood_max_age_sec', 2.0)
        self.declare_parameter('flir_raw_max_age_sec', 1.0)
        self.declare_parameter('minimum_flir_gray_p99', 20.0)
        self.declare_parameter('minimum_flir_dynamic_range', 12.0)

        self._panel_width = max(320, int(self.get_parameter('panel_width').value))
        self._panel_height = max(180, int(self.get_parameter('panel_height').value))
        self._jpeg_quality = max(30, min(100, int(self.get_parameter('jpeg_quality').value)))
        self._max_ages = {
            'top': float(self.get_parameter('top_max_age_sec').value),
            'cam1': float(self.get_parameter('cam1_max_age_sec').value),
            'flir_blood': float(self.get_parameter('flir_blood_max_age_sec').value),
            'flir_raw': float(self.get_parameter('flir_raw_max_age_sec').value),
        }
        self._minimum_flir_gray_p99 = float(
            self.get_parameter('minimum_flir_gray_p99').value)
        self._minimum_flir_dynamic_range = float(
            self.get_parameter('minimum_flir_dynamic_range').value)
        self._images = {name: LatestImage() for name in self._max_ages}
        self._fusion_status: dict[str, Any] = {}
        self._fusion_status_received: float | None = None
        self._last_signature: tuple[Any, ...] | None = None
        self._last_output_at: float | None = None
        self._output_hz = 0.0
        self._last_output_bytes = 0
        self._last_header: Any | None = None

        topic = lambda name: str(self.get_parameter(name).value)
        self._quad_subscriptions = [
            self.create_subscription(
                CompressedImage, topic('top_overlay_topic'),
                lambda message: self._on_image('top', message), image_qos()),
            self.create_subscription(
                CompressedImage, topic('cam1_hand_overlay_topic'),
                lambda message: self._on_image('cam1', message), image_qos()),
            self.create_subscription(
                CompressedImage, topic('flir_blood_overlay_topic'),
                lambda message: self._on_image('flir_blood', message), image_qos()),
            self.create_subscription(
                CompressedImage, topic('flir_raw_topic'),
                lambda message: self._on_image('flir_raw', message), image_qos()),
            self.create_subscription(
                String, topic('fusion_status_topic'),
                self._on_fusion_status, status_qos()),
        ]
        self._output_pub = self.create_publisher(
            CompressedImage, topic('output_topic'), image_qos())
        self._status_pub = self.create_publisher(
            String, topic('status_topic'), status_qos())
        rate = max(1.0, min(20.0, float(self.get_parameter('output_rate_hz').value)))
        self.create_timer(1.0 / rate, self._publish_quad)
        self.create_timer(0.5, self._publish_status)

    def _on_image(self, name: str, message: CompressedImage) -> None:
        state = self._images[name]
        state.received += 1
        image = decode_jpeg(message)
        if image is None:
            state.dropped += 1
            return
        state.message = message
        state.image = image
        state.received_monotonic = time.monotonic()
        state.source_stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _on_fusion_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get('schema') != 'pnu.perception.multiview_hand_fusion.v1':
                return
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self._fusion_status = payload
        self._fusion_status_received = time.monotonic()

    def _is_live(self, name: str, now: float) -> bool:
        received = self._images[name].received_monotonic
        return received is not None and now - received <= self._max_ages[name]

    def _render_bottom(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        if self._is_live('cam1', now) and self._images['cam1'].image is not None:
            cam1 = letterbox(
                self._images['cam1'].image, self._panel_width, self._panel_height)
        else:
            cam1 = placeholder(
                self._panel_width, self._panel_height, 'CAM_1 HAND OVERLAY STALE')
        cv2.rectangle(cam1, (0, 0), (self._panel_width, 44), (12, 24, 36), -1)
        _outlined_text(cam1, 'CAM_1  HAND', (14, 31), 0.68, (225, 240, 250), 2)
        if (
            self._fusion_status_received is not None
            and now - self._fusion_status_received <= 1.5
        ):
            selected = self._fusion_status.get('selected_camera')
            quality = self._fusion_status.get('selected_quality')
            suffix = '' if quality is None else f' q={float(quality):.2f}'
            text = f'FUSED SELECT: {str(selected).upper() if selected else "NONE"}{suffix}'
            color = (60, 235, 80) if selected == 'cam_1' else (80, 210, 255)
            gesture_rows = self._fusion_status.get('selected_gestures') or []
            facing_rows = self._fusion_status.get('selected_facings') or []
            gesture_text = ', '.join(
                f'H{row.get("hand_index", -1)} {row.get("category", "None")}'
                for row in gesture_rows[:2]) or 'None'
            facing_text = ', '.join(
                f'H{row.get("hand_index", -1)} {row.get("category", "UNKNOWN")}'
                for row in facing_rows[:2]) or 'NO VALID FACING'
            if not bool(self._fusion_status.get('gesture_facing_joinable', False)):
                facing_text = 'UNAVAILABLE (NO SAME-VIEW DEPTH MATCH)'
            cv2.rectangle(
                cam1, (0, self._panel_height - 92),
                (self._panel_width, self._panel_height), (12, 24, 36), -1)
            _outlined_text(cam1, text, (16, self._panel_height - 64), 0.58, color, 2)
            _outlined_text(
                cam1, f'GESTURE: {gesture_text}',
                (16, self._panel_height - 38), 0.52, (90, 235, 110), 1)
            _outlined_text(
                cam1, f'FACING: {facing_text}',
                (16, self._panel_height - 14), 0.48,
                (90, 235, 110) if facing_rows else (80, 210, 255), 1)

        flir_source = None
        flir_state = 'BLOOD OVERLAY MISSING'
        if self._is_live('flir_blood', now) and self._images['flir_blood'].image is not None:
            flir_source = self._images['flir_blood'].image
            flir_state = 'BLOOD OVERLAY LIVE'
        elif self._is_live('flir_raw', now) and self._images['flir_raw'].image is not None:
            flir_source = self._images['flir_raw'].image
            flir_state = 'BLOOD OVERLAY STALE'
        if flir_source is None:
            flir = placeholder(
                self._panel_width, self._panel_height, 'FLIR BLOOD VIEW STALE')
        else:
            flir = letterbox(flir_source, self._panel_width, self._panel_height)
        cv2.rectangle(flir, (0, 0), (self._panel_width, 44), (12, 24, 36), -1)
        _outlined_text(flir, f'FLIR  {flir_state}', (14, 31), 0.62, (225, 240, 250), 2)
        if flir_source is not None:
            quality_source = (
                self._images['flir_raw'].image
                if self._is_live('flir_raw', now)
                and self._images['flir_raw'].image is not None
                else flir_source
            )
            p99, dynamic_range = low_light_metrics(quality_source)
            if (
                p99 < self._minimum_flir_gray_p99
                or dynamic_range < self._minimum_flir_dynamic_range
            ):
                _outlined_text(
                    flir,
                    f'INPUT DARK - BLOOD UNKNOWN  p99={p99:.0f}',
                    (16, self._panel_height - 20),
                    0.66,
                    (45, 190, 255),
                    2,
                )
        return cam1, flir

    def _publish_quad(self) -> None:
        now = time.monotonic()
        if not self._is_live('top', now) or self._images['top'].image is None:
            return
        top = cv2.resize(
            self._images['top'].image,
            (self._panel_width * 2, self._panel_height),
            interpolation=cv2.INTER_AREA,
        )
        cam1, flir = self._render_bottom(now)
        signature = tuple(
            self._images[name].source_stamp_ns if self._is_live(name, now) else None
            for name in ('top', 'cam1', 'flir_blood', 'flir_raw')
        ) + (
            self._fusion_status.get('selected_camera'),
            self._fusion_status.get('selected_source_stamp_ns'),
        )
        if signature == self._last_signature:
            return
        quad = np.vstack((top, np.hstack((cam1, flir))))
        success, encoded = cv2.imencode(
            '.jpg', quad, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not success:
            return
        output = CompressedImage()
        output.header = self._images['top'].message.header
        output.format = 'jpeg'
        output.data = encoded.tobytes()
        self._output_pub.publish(output)
        if self._last_output_at is not None and now > self._last_output_at:
            self._output_hz = 1.0 / (now - self._last_output_at)
        self._last_output_at = now
        self._last_output_bytes = len(output.data)
        self._last_header = output.header
        self._last_signature = signature

    def _publish_status(self) -> None:
        now = time.monotonic()
        payload = {
            'schema': 'pnu.perception.operator_quad.v2',
            'output': {
                'width': self._panel_width * 2,
                'height': self._panel_height * 2,
                'hz': round(self._output_hz, 3),
                'bytes': self._last_output_bytes,
            },
            'panels': {},
            'fusion': self._fusion_status,
            'robot_authority': False,
        }
        for name in ('top', 'cam1', 'flir_blood', 'flir_raw'):
            state = self._images[name]
            age = (
                None if state.received_monotonic is None
                else max(0.0, now - state.received_monotonic)
            )
            payload['panels'][name] = {
                'state': 'live' if self._is_live(name, now) else (
                    'missing' if state.message is None else 'stale'),
                'age_sec': None if age is None else round(age, 3),
                'source_stamp_ns': state.source_stamp_ns,
                'received': state.received,
                'dropped': state.dropped,
            }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(',', ':'))))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OperatorQuadCompositor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
