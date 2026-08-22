"""Render one current, structured Debug overlay for CAM3 and CAM4.

Unlike the legacy difference-image compositor, this node never compares two
opaque JPEG video frames.  It draws Tool observations/poses, Hand keypoints,
and Blood masks from their typed result messages on the single latest ingress
base frame.  Every layer is independently freshness-gated, so a slow worker
cannot freeze or ghost the video from another worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
from typing import Any

import cv2
import numpy as np

import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from hand_keypoint_interfaces.msg import HandKeypoints
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from surgical_perception_msgs.msg import ToolObservation2DArray, ToolPoseArray

from pnu_surgical_perception.final_overlay_contract import (
    CAMERA_STATUS_KEYS,
    LAYER_NAMES,
    STATUS_SCHEMA,
    Freshness,
    freshness_state,
    layer_is_drawable,
    stamp_dict,
    stamp_ns,
)


CAMERAS = ('cam_3', 'cam_4')
TOOL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    'Scalpel': (0, 145, 255),
    'Allis Forceps': (255, 130, 30),
    'Mosquito': (70, 190, 80),
    'Adson Forceps': (205, 70, 195),
    'Bipolar Forceps': (0, 230, 230),
    'Bovie': (100, 60, 255),
    'Army-Navy Retractor': (185, 85, 155),
    'Thyroid Retractor': (210, 180, 30),
}
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def image_reader_qos() -> QoSProfile:
    """Latest-frame image reader QoS for the local ingress fan-out."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def result_qos() -> QoSProfile:
    """Reliable typed-result QoS with no accumulating Debug backlog."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def camera_info_qos() -> QoSProfile:
    """Reliable CameraInfo contract; images use ``image_reader_qos`` instead."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def status_qos() -> QoSProfile:
    """Small retained status document for a Debug UI that joins late."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def decode_jpeg(message: CompressedImage) -> np.ndarray | None:
    """Decode one base frame without letting a malformed packet kill Debug."""
    return cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_binary_mask(message: Image) -> np.ndarray | None:
    """Decode the current Blood ``mono8`` result without changing its stamp."""
    encoding = str(message.encoding).lower()
    if encoding not in {'mono8', '8uc1'}:
        return None
    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or step < width:
        return None
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size < height * step:
        return None
    return data[:height * step].reshape(height, step)[:, :width] > 0


def quaternion_matrix_xyzw(quaternion: Any) -> np.ndarray | None:
    """Return a rotation matrix for a finite, normalized ROS quaternion."""
    values = np.asarray(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=np.float64
    )
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    x, y, z, w = values / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass
class LatestBase:
    message: CompressedImage | None = None
    image: np.ndarray | None = None
    source_stamp_ns: int | None = None
    freshness: Freshness = field(default_factory=lambda: Freshness(None))
    received: int = 0
    dropped: int = 0


@dataclass
class LatestLayer:
    message: Any | None = None
    payload: Any | None = None
    source_stamp_ns: int | None = None
    freshness: Freshness = field(default_factory=lambda: Freshness(None))
    count: int = 0
    dropped: int = 0
    last_drop_signature: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class LayerDecision:
    state: str
    drawable: bool
    age_sec: float | None


class FinalOverlayCompositor(Node):
    """Publish one 2-up JPEG and its compact, source-stamp-preserving status."""

    def __init__(self, *, context: Context | None = None) -> None:
        super().__init__('final_overlay_compositor', context=context)
        self._declare_parameters()
        self._read_parameters()
        self._base = {camera: LatestBase() for camera in CAMERAS}
        self._layers = {
            camera: {name: LatestLayer() for name in LAYER_NAMES}
            for camera in CAMERAS
        }
        self._camera_info: dict[str, CameraInfo | None] = {camera: None for camera in CAMERAS}
        self._last_signature: tuple[Any, ...] | None = None
        self._last_output_at: float | None = None
        self._last_output_source_header: Any | None = None
        self._output_hz = 0.0
        self._last_output_bytes = 0
        self._last_output_width = 0
        self._last_output_height = 0

        self._image_publisher = self.create_publisher(
            CompressedImage, self._output_topic, image_reader_qos())
        self._status_publisher = self.create_publisher(
            String, self._status_topic, status_qos())
        # ``Node`` owns ``_subscriptions`` internally.  Keep application
        # references under a non-rclpy name so Node.destroy_node() can tear
        # down waitables safely.
        self._perception_subscriptions: list[Any] = []
        for camera in CAMERAS:
            prefix = f'/perception/ingress/{camera}'
            self._perception_subscriptions.append(self.create_subscription(
                CompressedImage, self._base_topics[camera],
                lambda message, cam=camera: self._on_base(cam, message), image_reader_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                CameraInfo, self._camera_info_topics[camera],
                lambda message, cam=camera: self._on_camera_info(cam, message), camera_info_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                ToolObservation2DArray, self._tool_topics[camera],
                lambda message, cam=camera: self._on_result(cam, 'tool', message), result_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                ToolPoseArray, self._pose_topics[camera],
                lambda message, cam=camera: self._on_result(cam, 'pose', message), result_qos()))
            if camera == 'cam_4' and self._enable_hand:
                self._perception_subscriptions.append(self.create_subscription(
                    HandKeypoints, self._hand_topic,
                    lambda message: self._on_result('cam_4', 'hand', message), result_qos()))
            if camera == 'cam_4' and self._enable_blood:
                self._perception_subscriptions.append(self.create_subscription(
                    Image, self._blood_mask_topic, self._on_blood_mask, result_qos()))

        self.create_timer(1.0 / self._output_rate_hz, self._publish_if_current)
        self.create_timer(1.0 / self._status_rate_hz, self._publish_status)
        self.get_logger().info(
            f'final overlay reads only local ingress: cam3={self._base_topics["cam_3"]}, '
            f'cam4={self._base_topics["cam_4"]}; output={self._output_topic}; '
            f'layer_age<={self._max_layer_age_sec:.3f}s; '
            f'source_delta<={self._max_source_delta_ns / 1_000_000:.1f}ms')

    def _declare_parameters(self) -> None:
        for camera in CAMERAS:
            prefix = f'/perception/ingress/{camera}'
            output = f'/perception/{camera}/tool'
            self.declare_parameter(
                f'{camera}_color_topic', f'{prefix}/color/image_raw/compressed')
            self.declare_parameter(
                f'{camera}_camera_info_topic', f'{prefix}/color/camera_info')
            self.declare_parameter(
                f'{camera}_tool_topic', f'{output}/observations')
            self.declare_parameter(
                f'{camera}_pose_topic', f'{output}/poses')
        self.declare_parameter('hand_topic', '/perception/cam_4/hand/keypoints')
        self.declare_parameter('blood_mask_topic', '/perception/cam_4/blood/mask')
        self.declare_parameter('enable_hand', True)
        self.declare_parameter('enable_blood', True)
        self.declare_parameter('output_topic', '/perception/debug/final_overlay/compressed')
        self.declare_parameter('status_topic', '/perception/debug/final_overlay/status')
        self.declare_parameter('output_rate_hz', 10.0)
        self.declare_parameter('status_rate_hz', 2.0)
        self.declare_parameter('max_base_age_sec', 1.0)
        self.declare_parameter('max_layer_age_sec', 1.5)
        # The GPU workers publish a source-stamped result roughly once per
        # second.  Comparing that result to a 10-15 Hz *current* base with a
        # 150 ms window made every valid tool result disappear between worker
        # callbacks.  A 1.8 s source window covers the measured inference plus
        # one result period; ``max_layer_age_sec`` remains the independent
        # receiver-time fail-closed gate, so a stopped worker is never held
        # indefinitely.  Rendering still starts from a fresh base copy and
        # therefore cannot accumulate raster ghost trails.
        self.declare_parameter('max_source_delta_ms', 1800.0)
        self.declare_parameter('panel_width', 960)
        self.declare_parameter('panel_height', 540)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('pose_axis_length_m', 0.05)

    def _read_parameters(self) -> None:
        def text(name: str) -> str:
            value = str(self.get_parameter(name).value).strip()
            if not value.startswith('/'):
                raise ValueError(f'{name} must be an absolute ROS topic')
            return value

        self._base_topics = {camera: text(f'{camera}_color_topic') for camera in CAMERAS}
        self._camera_info_topics = {
            camera: text(f'{camera}_camera_info_topic') for camera in CAMERAS
        }
        self._tool_topics = {camera: text(f'{camera}_tool_topic') for camera in CAMERAS}
        self._pose_topics = {camera: text(f'{camera}_pose_topic') for camera in CAMERAS}
        self._hand_topic = text('hand_topic')
        self._blood_mask_topic = text('blood_mask_topic')
        self._enable_hand = bool(self.get_parameter('enable_hand').value)
        self._enable_blood = bool(self.get_parameter('enable_blood').value)
        self._output_topic = text('output_topic')
        self._status_topic = text('status_topic')
        self._output_rate_hz = max(1.0, min(30.0, float(self.get_parameter('output_rate_hz').value)))
        self._status_rate_hz = max(0.2, min(10.0, float(self.get_parameter('status_rate_hz').value)))
        self._max_base_age_sec = max(0.05, float(self.get_parameter('max_base_age_sec').value))
        self._max_layer_age_sec = max(0.05, float(self.get_parameter('max_layer_age_sec').value))
        self._max_source_delta_ns = max(
            0, int(float(self.get_parameter('max_source_delta_ms').value) * 1_000_000))
        self._panel_width = max(160, int(self.get_parameter('panel_width').value))
        self._panel_height = max(90, int(self.get_parameter('panel_height').value))
        self._jpeg_quality = max(20, min(100, int(self.get_parameter('jpeg_quality').value)))
        self._pose_axis_length_m = max(0.001, float(self.get_parameter('pose_axis_length_m').value))

    def _on_base(self, camera: str, message: CompressedImage) -> None:
        state = self._base[camera]
        state.received += 1
        image = decode_jpeg(message)
        if image is None:
            state.dropped += 1
            return
        state.message = message
        state.image = image
        state.source_stamp_ns = stamp_ns(message)
        state.freshness = Freshness(time.monotonic())

    def _on_camera_info(self, camera: str, message: CameraInfo) -> None:
        self._camera_info[camera] = message

    def _on_result(self, camera: str, layer: str, message: Any) -> None:
        state = self._layers[camera][layer]
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        if layer == 'tool':
            state.count = len(message.instances)
        elif layer == 'pose':
            state.count = len(message.tools)
        elif layer == 'hand':
            state.count = len(message.hands)

    def _on_blood_mask(self, message: Image) -> None:
        state = self._layers['cam_4']['blood']
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        mask = decode_binary_mask(message)
        if mask is None:
            state.dropped += 1
            return
        state.message = message
        state.payload = mask
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = int(np.count_nonzero(mask))

    def _base_state(self, camera: str, now: float) -> tuple[str, float | None]:
        base = self._base[camera]
        age = base.freshness.age(now)
        return (
            freshness_state(
                has_value=base.message is not None and base.image is not None,
                age_sec=age,
                max_age_sec=self._max_base_age_sec,
            ), age,
        )

    def _layer_decision(
        self, camera: str, layer: str, base_state: str, now: float
    ) -> LayerDecision:
        disabled = camera == 'cam_3' and layer in {'hand', 'blood'}
        disabled = disabled or (camera == 'cam_4' and layer == 'hand' and not self._enable_hand)
        disabled = disabled or (camera == 'cam_4' and layer == 'blood' and not self._enable_blood)
        state = self._layers[camera][layer]
        age = state.freshness.age(now)
        public_state = freshness_state(
            has_value=state.message is not None,
            age_sec=age,
            max_age_sec=self._max_layer_age_sec,
            disabled=disabled,
        )
        drawable = layer_is_drawable(
            base_stamp_ns=self._base[camera].source_stamp_ns,
            layer_stamp_ns=state.source_stamp_ns,
            base_state=base_state,
            layer_state=public_state,
            max_source_delta_ns=self._max_source_delta_ns,
        )
        if public_state == 'live' and not drawable:
            signature = (
                self._base[camera].source_stamp_ns, state.source_stamp_ns, base_state,
            )
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            public_state = 'stale'
        return LayerDecision(state=public_state, drawable=drawable, age_sec=age)

    def _camera_context(self, camera: str, now: float) -> dict[str, Any]:
        base_state, base_age = self._base_state(camera, now)
        layers = {
            name: self._layer_decision(camera, name, base_state, now)
            for name in LAYER_NAMES
        }
        return {'base_state': base_state, 'base_age': base_age, 'layers': layers}

    def _publish_if_current(self) -> None:
        now = time.monotonic()
        contexts = {camera: self._camera_context(camera, now) for camera in CAMERAS}
        anchors = [
            camera for camera in CAMERAS
            if contexts[camera]['base_state'] == 'live'
        ]
        if not anchors:
            return
        anchor = max(anchors, key=lambda camera: self._base[camera].freshness.received_monotonic or -1.0)
        signature: tuple[Any, ...] = tuple(
            item
            for camera in CAMERAS
            for item in (
                self._base[camera].source_stamp_ns if contexts[camera]['base_state'] == 'live' else None,
                *(self._layers[camera][layer].source_stamp_ns
                  if contexts[camera]['layers'][layer].drawable else None
                  for layer in LAYER_NAMES),
            )
        )
        if signature == self._last_signature:
            return
        panels = [self._render_camera_panel(camera, contexts[camera]) for camera in CAMERAS]
        image = np.hstack(panels)
        success, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not success:
            self.get_logger().warning('could not encode final Debug JPEG')
            return
        source = self._base[anchor].message
        if source is None:
            return
        output = CompressedImage()
        # The final image is a view of this particular source image.  Keep its
        # original stamp; assigning new "now" here would create a false claim
        # that stale worker evidence belongs to a newer camera frame.
        output.header = source.header
        output.format = 'jpeg'
        output.data = encoded.tobytes()
        self._image_publisher.publish(output)
        if self._last_output_at is not None:
            interval = now - self._last_output_at
            if interval > 1e-6:
                self._output_hz = 1.0 / interval
        self._last_output_at = now
        self._last_output_source_header = output.header
        self._last_output_bytes = len(output.data)
        self._last_output_height, self._last_output_width = image.shape[:2]
        self._last_signature = signature

    def _render_camera_panel(self, camera: str, context: dict[str, Any]) -> np.ndarray:
        base = self._base[camera]
        if context['base_state'] != 'live' or base.image is None:
            panel = np.zeros((self._panel_height, self._panel_width, 3), dtype=np.uint8)
            label = 'MISSING' if context['base_state'] == 'missing' else 'STALE'
            cv2.putText(panel, f'{camera.upper()} BASE {label}', (28, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (65, 180, 255), 2, cv2.LINE_AA)
            return panel
        image = base.image.copy()
        if context['layers']['tool'].drawable:
            self._draw_tool_observations(image, self._layers[camera]['tool'].message)
        if context['layers']['pose'].drawable:
            self._draw_tool_poses(image, self._layers[camera]['pose'].message, self._camera_info[camera])
        if context['layers']['hand'].drawable:
            self._draw_hands(image, self._layers[camera]['hand'].message)
        if context['layers']['blood'].drawable:
            self._draw_blood(image, self._layers[camera]['blood'])
        status = ' '.join(
            f'{name}:{context["layers"][name].state}' for name in LAYER_NAMES
            if context['layers'][name].state != 'disabled'
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 32), (12, 24, 36), -1)
        cv2.putText(image, f'{camera.upper()}  {status}', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 240, 250), 1, cv2.LINE_AA)
        return self._letterbox(image)

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(self._panel_width / width, self._panel_height / height)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        panel = np.zeros((self._panel_height, self._panel_width, 3), dtype=np.uint8)
        y = (self._panel_height - resized.shape[0]) // 2
        x = (self._panel_width - resized.shape[1]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return panel

    def _draw_tool_observations(self, image: np.ndarray, message: ToolObservation2DArray) -> None:
        for item in message.instances:
            color = TOOL_COLORS_BGR.get(str(item.class_name), (235, 235, 235))
            x0, y0, x1, y1 = (int(round(value)) for value in item.bbox_xyxy_px)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            label = f'{item.class_name} {float(item.class_confidence):.2f}'
            cv2.putText(image, label, (x0, max(18, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            if bool(item.observation_point_valid):
                u, v = (int(round(value)) for value in item.observation_point_uv_px)
                cv2.circle(image, (u, v), 4, color, -1, cv2.LINE_AA)
                cv2.circle(image, (u, v), 7, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_tool_poses(
        self, image: np.ndarray, message: ToolPoseArray, camera_info: CameraInfo | None
    ) -> None:
        if camera_info is None or len(camera_info.k) != 9:
            return
        K = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        D = np.asarray(camera_info.d, dtype=np.float64).reshape(-1, 1)
        for item in message.tools:
            if not bool(item.position_valid):
                continue
            position = np.asarray([
                item.pose.position.x, item.pose.position.y, item.pose.position.z
            ], dtype=np.float64)
            rotation = quaternion_matrix_xyzw(item.pose.orientation)
            if not np.all(np.isfinite(position)) or rotation is None:
                continue
            points = np.vstack((
                position,
                position + rotation[:, 0] * self._pose_axis_length_m,
                position + rotation[:, 1] * self._pose_axis_length_m,
                position + rotation[:, 2] * self._pose_axis_length_m,
            ))
            if np.any(points[:, 2] <= 0.0):
                continue
            try:
                projected, _ = cv2.projectPoints(
                    points, np.zeros(3), np.zeros(3), K, D)
            except cv2.error:
                continue
            uv = projected.reshape(-1, 2)
            if not np.all(np.isfinite(uv)):
                continue
            origin = tuple(np.rint(uv[0]).astype(int))
            for endpoint, axis_color in zip(uv[1:], ((40, 40, 255), (40, 230, 40), (255, 120, 40))):
                cv2.line(image, origin, tuple(np.rint(endpoint).astype(int)), axis_color, 2, cv2.LINE_AA)
            pose_label = 'VALID' if int(item.validity) == 1 else 'DEGRADED'
            color = TOOL_COLORS_BGR.get(str(item.class_name), (235, 235, 235))
            cv2.putText(image, f'{item.class_name}:{pose_label}', (origin[0] + 6, origin[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _draw_hands(self, image: np.ndarray, message: HandKeypoints) -> None:
        for hand in message.hands:
            color = (50, 220, 50) if str(hand.handedness_label).lower() == 'right' else (255, 180, 35)
            points = [
                (int(round(joint.u)), int(round(joint.v)))
                for joint in hand.joints_2d
            ]
            for start, end in HAND_EDGES:
                cv2.line(image, points[start], points[end], color, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, point, 3, (250, 250, 250), -1, cv2.LINE_AA)
            if points:
                label = str(hand.handedness_label) if bool(hand.has_handedness) else 'Hand'
                cv2.putText(image, label, (points[0][0] + 6, points[0][1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_blood(self, image: np.ndarray, state: LatestLayer) -> None:
        mask = state.payload
        if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
            signature = (state.source_stamp_ns, image.shape[:2], getattr(mask, 'shape', None))
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            return
        tinted = image.copy()
        tinted[mask] = (40, 40, 235)
        blended = cv2.addWeighted(image, 0.65, tinted, 0.35, 0.0)
        image[mask] = blended[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (35, 35, 255), 2, cv2.LINE_AA)

    def _publish_status(self) -> None:
        now = time.monotonic()
        # The Debug UI's status transport deliberately has a strict base/output
        # schema.  Until each camera has contributed one decoded base and one
        # final JPEG has actually been published, any source stamp or receiver
        # age would be unknowable.  Suppress the document rather than forge a
        # zero timestamp or make the strict consumer parse a partial object.
        if self._last_output_at is None or self._last_output_source_header is None or any(
            self._base[camera].message is None or self._base[camera].image is None
            for camera in CAMERAS
        ):
            return
        contexts = {camera: self._camera_context(camera, now) for camera in CAMERAS}
        wall_ns = time.time_ns()
        payload: dict[str, Any] = {
            'schema': STATUS_SCHEMA,
            'published_at': {'sec': wall_ns // 1_000_000_000, 'nanosec': wall_ns % 1_000_000_000},
            'output': {
                # Do not recalculate this from a newer callback.  A base can
                # arrive between image timer and status timer; the status must
                # identify the already published JPEG, not that later frame.
                'source_stamp': stamp_dict(self._last_output_source_header),
                'hz': round(float(self._output_hz), 3),
                'bytes': int(self._last_output_bytes),
                'width': int(self._last_output_width),
                'height': int(self._last_output_height),
            },
            'cameras': {},
        }
        for camera in CAMERAS:
            base = self._base[camera]
            base_state = contexts[camera]['base_state']
            layers: dict[str, Any] = {}
            for layer in LAYER_NAMES:
                current = self._layers[camera][layer]
                decision = contexts[camera]['layers'][layer]
                layers[layer] = {
                    'state': decision.state,
                    'source_stamp': stamp_dict(current.message),
                    'age_sec': None if decision.age_sec is None else round(decision.age_sec, 3),
                    'count': int(current.count),
                    'dropped': int(current.dropped),
                }
            payload['cameras'][CAMERA_STATUS_KEYS[camera]] = {
                'state': base_state,
                'base': {
                    'source_stamp': stamp_dict(base.message),
                    'age_sec': None if contexts[camera]['base_age'] is None else round(contexts[camera]['base_age'], 3),
                    'received': int(base.received),
                    'dropped': int(base.dropped),
                },
                'layers': layers,
            }
        self._status_publisher.publish(String(data=json.dumps(payload, separators=(',', ':'))))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FinalOverlayCompositor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
