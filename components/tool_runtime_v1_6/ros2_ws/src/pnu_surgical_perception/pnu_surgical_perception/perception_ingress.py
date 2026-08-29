"""Single-subscription ingress for the external camera streams.

Each instance owns one camera's external source subscriptions and fans the
messages out locally under ``/perception/ingress``.  It deliberately does
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


EIR_ALIGNED_DEPTH_CAMERAS = frozenset({'head', 'suction'})
DEPTH_CAPABLE_CAMERAS = frozenset(
    {'cam_1', 'cam_2', 'cam_3', 'cam_4', *EIR_ALIGNED_DEPTH_CAMERAS}
)
# The end-effector EIR relays currently expose RGB only. Head and suction
# publish 16UC1 depth already aligned into their color optical frames.
EIR_RGB_ONLY_CAMERAS = frozenset({'left_ee', 'right_ee'})
EIR_CAMERAS = frozenset(
    {*EIR_ALIGNED_DEPTH_CAMERAS, *EIR_RGB_ONLY_CAMERAS}
)
EXTRINSICS_CAPABLE_CAMERAS = frozenset({'cam_1', 'cam_2', 'cam_3', 'cam_4'})
SUPPORTED_CAMERAS = frozenset((*DEPTH_CAPABLE_CAMERAS, *EIR_RGB_ONLY_CAMERAS, 'flir'))


def canonical_camera(camera: str) -> str:
    """Normalize a supported camera selector without accepting arbitrary paths."""
    raw = str(camera).strip()
    for prefix in ('/synced/', '/eir/camera/'):
        if raw.startswith(prefix):
            raw = raw.removeprefix(prefix)
            break
    raw = raw.split('/', 1)[0]
    if raw.isdigit():
        raw = f'cam_{raw}'
    elif raw.startswith('cam') and raw[3:].isdigit():
        raw = f'cam_{raw[3:]}'
    if raw not in SUPPORTED_CAMERAS:
        raise ValueError(
            'camera must be cam_1, cam_2, cam_3, cam_4, flir, suction, '
            'head, left_ee, or right_ee'
        )
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
    local = f'/perception/ingress/{normalized}'
    if normalized in EIR_ALIGNED_DEPTH_CAMERAS:
        remote = f'/eir/camera/{normalized}'
        return IngressTopics(
            camera=normalized,
            remote_color=f'{remote}/color/image_raw/compressed',
            remote_depth=(
                f'{remote}/aligned_depth_to_color/image_raw/compressedDepth'
            ),
            remote_color_info=f'{remote}/color/camera_info',
            remote_depth_info=f'{remote}/aligned_depth_to_color/camera_info',
            # EIR publishes depth already aligned to color, so no extrinsics
            # message is needed or expected for this source.
            remote_extrinsics=f'{remote}/extrinsics/depth_to_color',
            local_color=f'{local}/color/image_raw/compressed',
            local_depth=f'{local}/depth/image_rect_raw/compressedDepth',
            local_color_info=f'{local}/color/camera_info',
            local_depth_info=f'{local}/depth/camera_info',
            local_extrinsics=f'{local}/extrinsics/depth_to_color',
        )
    if normalized in EIR_RGB_ONLY_CAMERAS:
        remote = f'/eir/camera/{normalized}'
        # Depth/extrinsics fields remain absolute only because the generic
        # parameter schema validates every name.  This RGB-only ingress never
        # subscribes to or advertises those unused fields.
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
    remote = f'/synced/{normalized}'
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


def camera_has_depth(camera: str) -> bool:
    """Return whether the live camera contract actually publishes depth."""
    return canonical_camera(camera) in DEPTH_CAPABLE_CAMERAS


def camera_has_extrinsics(camera: str) -> bool:
    """Return whether the source publishes a depth-to-color transform."""
    return canonical_camera(camera) in EXTRINSICS_CAPABLE_CAMERAS


def image_reader_qos() -> QoSProfile:
    """Return the image-reader contract with no backlog or retransmits."""
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


def remote_camera_info_qos(camera: str) -> QoSProfile:
    """Match each external source while retaining reliable local fan-out.

    EIR's camera publishers offer CameraInfo with the same
    sensor-data QoS as their images.  A RELIABLE reader is incompatible with
    that BEST_EFFORT writer, so only these external EIR edges use
    BEST_EFFORT.  The local CameraInfo publisher remains RELIABLE via
    :func:`camera_info_qos`.
    """
    if canonical_camera(camera) in EIR_CAMERAS:
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
    return camera_info_qos()


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
        self._has_depth = camera_has_depth(camera)
        self._has_extrinsics = camera_has_extrinsics(camera)
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
        local_info_qos = camera_info_qos()
        remote_info_qos = remote_camera_info_qos(camera)
        # ``Node`` reserves ``_publishers`` and ``_subscriptions`` for its
        # lifecycle bookkeeping.  Keep ingress-owned handles separate so
        # publishing and destroy_node() both use rclpy's intact collections.
        self._ingress_publishers = {
            'color': self.create_publisher(
                CompressedImage, self._topics['local_color_topic'], image_qos),
            'color_info': self.create_publisher(
                CameraInfo, self._topics['local_color_camera_info_topic'], local_info_qos),
        }
        if self._has_depth:
            self._ingress_publishers.update({
                'depth': self.create_publisher(
                    CompressedImage, self._topics['local_depth_topic'], image_qos),
                'depth_info': self.create_publisher(
                    CameraInfo, self._topics['local_depth_camera_info_topic'], local_info_qos),
            })
        if self._has_extrinsics:
            self._ingress_publishers.update({
                'extrinsics': self.create_publisher(
                    Extrinsics, self._topics['local_extrinsics_topic'],
                    local_extrinsics_qos()),
            })
        self._ingress_subscriptions = [
            self.create_subscription(
                CompressedImage, self._topics['remote_color_topic'],
                lambda message: self._forward('color', message), image_qos),
            # Match the external writer here; local CameraInfo fan-out stays
            # RELIABLE for the native-depth, Hand, and Blood calibration
            # contract.
            self.create_subscription(
                CameraInfo, self._topics['remote_color_camera_info_topic'],
                lambda message: self._forward('color_info', message), remote_info_qos),
        ]
        if self._has_depth:
            self._ingress_subscriptions.extend([
                self.create_subscription(
                    CompressedImage, self._topics['remote_depth_topic'],
                    lambda message: self._forward('depth', message), image_qos),
                self.create_subscription(
                    CameraInfo, self._topics['remote_depth_camera_info_topic'],
                    lambda message: self._forward('depth_info', message), remote_info_qos),
            ])
        if self._has_extrinsics:
            self._ingress_subscriptions.extend([
                self.create_subscription(
                    Extrinsics, self._topics['remote_extrinsics_topic'],
                    lambda message: self._forward('extrinsics', message),
                    local_extrinsics_qos()),
            ])
        self.get_logger().info(
            f'{self._camera} ingress: {self._topics["remote_color_topic"]} -> '
            f'{self._topics["local_color_topic"]}; image reader QoS is '
            'BEST_EFFORT/VOLATILE/KEEP_LAST(1); '
            f'depth_capable={self._has_depth}; '
            f'extrinsics_capable={self._has_extrinsics}')

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
