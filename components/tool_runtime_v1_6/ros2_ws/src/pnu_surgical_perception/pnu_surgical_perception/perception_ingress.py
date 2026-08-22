"""Single-subscription ingress for the VIPLab camera streams.

Each instance owns one camera's external ``/synced`` subscriptions and fans
the messages out locally under ``/perception/ingress``.  It deliberately does
not decode, pair, retimestamp, or buffer frames: the source header and the
calibration payload are the contract passed to the local workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from realsense2_camera_msgs.msg import Extrinsics
from sensor_msgs.msg import CameraInfo, CompressedImage


def canonical_camera(camera: str) -> str:
    """Normalize a supported camera selector without accepting arbitrary paths."""
    raw = str(camera).strip().removeprefix('/synced/').split('/', 1)[0]
    if raw.isdigit():
        raw = f'cam_{raw}'
    elif raw.startswith('cam') and raw[3:].isdigit():
        raw = f'cam_{raw[3:]}'
    if raw not in {'cam_3', 'cam_4'}:
        raise ValueError('camera must be cam_3 or cam_4')
    return raw


@dataclass(frozen=True)
class IngressTopics:
    """The external source and local fan-out names for one camera."""

    camera: str
    remote_color: str
    remote_depth: str
    remote_color_info: str
    remote_depth_info: str
    remote_extrinsics: str
    local_color: str
    local_depth: str
    local_color_info: str
    local_depth_info: str
    local_extrinsics: str


def ingress_topics(camera: str) -> IngressTopics:
    """Return the fixed one-to-one source/fan-out mapping for ``camera``."""
    normalized = canonical_camera(camera)
    remote = f'/synced/{normalized}'
    local = f'/perception/ingress/{normalized}'
    return IngressTopics(
        camera=normalized,
        remote_color=f'{remote}/color/image_raw/compressed',
        remote_depth=f'{remote}/depth/image_rect_raw/compressedDepth',
        remote_color_info=f'{remote}/color/camera_info',
        remote_depth_info=f'{remote}/depth/camera_info',
        remote_extrinsics=f'{remote}/extrinsics/depth_to_color',
        local_color=f'{local}/color/image_raw/compressed',
        local_depth=f'{local}/depth/image_rect_raw/compressedDepth',
        local_color_info=f'{local}/color/camera_info',
        local_depth_info=f'{local}/depth/camera_info',
        local_extrinsics=f'{local}/extrinsics/depth_to_color',
    )


def image_reader_qos() -> QoSProfile:
    """The operating image-reader contract: no backlog and no retransmits."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def camera_info_qos() -> QoSProfile:
    """Keep small calibration messages reliable without image backlog."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def local_extrinsics_qos() -> QoSProfile:
    """Keep the last static factory transform available to late local workers."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class PerceptionIngress(Node):
    """Forward exactly one camera's source subscriptions to local topics."""

    def __init__(self, *, context: Context | None = None) -> None:
        super().__init__('perception_ingress', context=context)
        self.declare_parameter('camera', 'cam_4')
        camera = canonical_camera(str(self.get_parameter('camera').value))
        defaults = ingress_topics(camera)
        self._camera = camera
        for name, default in (
            ('remote_color_topic', defaults.remote_color),
            ('remote_depth_topic', defaults.remote_depth),
            ('remote_color_camera_info_topic', defaults.remote_color_info),
            ('remote_depth_camera_info_topic', defaults.remote_depth_info),
            ('remote_extrinsics_topic', defaults.remote_extrinsics),
            ('local_color_topic', defaults.local_color),
            ('local_depth_topic', defaults.local_depth),
            ('local_color_camera_info_topic', defaults.local_color_info),
            ('local_depth_camera_info_topic', defaults.local_depth_info),
            ('local_extrinsics_topic', defaults.local_extrinsics),
        ):
            self.declare_parameter(name, default)

        def topic(name: str) -> str:
            value = str(self.get_parameter(name).value).strip()
            if not value.startswith('/'):
                raise ValueError(f'{name} must be an absolute ROS topic')
            return value

        self._topics = {name: topic(name) for name in (
            'remote_color_topic', 'remote_depth_topic',
            'remote_color_camera_info_topic', 'remote_depth_camera_info_topic',
            'remote_extrinsics_topic', 'local_color_topic', 'local_depth_topic',
            'local_color_camera_info_topic', 'local_depth_camera_info_topic',
            'local_extrinsics_topic',
        )}
        self._forwarded = {
            'color': 0, 'depth': 0, 'color_info': 0,
            'depth_info': 0, 'extrinsics': 0,
        }

        image_qos = image_reader_qos()
        info_qos = camera_info_qos()
        # ``Node`` reserves ``_publishers`` and ``_subscriptions`` for its
        # lifecycle bookkeeping.  Keep ingress-owned handles separate so
        # publishing and destroy_node() both use rclpy's intact collections.
        self._ingress_publishers = {
            'color': self.create_publisher(
                CompressedImage, self._topics['local_color_topic'], image_qos),
            'depth': self.create_publisher(
                CompressedImage, self._topics['local_depth_topic'], image_qos),
            'color_info': self.create_publisher(
                CameraInfo, self._topics['local_color_camera_info_topic'], info_qos),
            'depth_info': self.create_publisher(
                CameraInfo, self._topics['local_depth_camera_info_topic'], info_qos),
            'extrinsics': self.create_publisher(
                Extrinsics, self._topics['local_extrinsics_topic'], local_extrinsics_qos()),
        }
        self._ingress_subscriptions = [
            self.create_subscription(
                CompressedImage, self._topics['remote_color_topic'],
                lambda message: self._forward('color', message), image_qos),
            self.create_subscription(
                CompressedImage, self._topics['remote_depth_topic'],
                lambda message: self._forward('depth', message), image_qos),
            # CameraInfo remains reliable for the native-depth, Hand, and
            # Blood calibration contract; it is not an image reader.
            self.create_subscription(
                CameraInfo, self._topics['remote_color_camera_info_topic'],
                lambda message: self._forward('color_info', message), info_qos),
            self.create_subscription(
                CameraInfo, self._topics['remote_depth_camera_info_topic'],
                lambda message: self._forward('depth_info', message), info_qos),
            self.create_subscription(
                Extrinsics, self._topics['remote_extrinsics_topic'],
                lambda message: self._forward('extrinsics', message), local_extrinsics_qos()),
        ]
        self.get_logger().info(
            f'{self._camera} ingress: {self._topics["remote_color_topic"]} -> '
            f'{self._topics["local_color_topic"]}; image reader QoS is '
            'BEST_EFFORT/VOLATILE/KEEP_LAST(1)')

    def _forward(self, kind: str, message: Any) -> None:
        """Publish the original object unchanged, including its source header."""
        self._ingress_publishers[kind].publish(message)
        self._forwarded[kind] += 1
        # A single startup confirmation per stream gives live cutover evidence
        # without turning a high-rate image path into a log stream.
        if self._forwarded[kind] in {1, 100}:
            header = getattr(message, 'header', None)
            stamp = getattr(header, 'stamp', None)
            self.get_logger().info(
                f'{self._camera} forwarding first {kind}: '
                f'stamp={getattr(stamp, "sec", "?")}.{getattr(stamp, "nanosec", "?")}')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PerceptionIngress()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
