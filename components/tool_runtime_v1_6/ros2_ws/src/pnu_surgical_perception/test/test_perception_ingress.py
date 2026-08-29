from types import SimpleNamespace

from pnu_surgical_perception.perception_ingress import (
    camera_has_depth,
    camera_info_qos,
    canonical_camera,
    image_reader_qos,
    ingress_topics,
    local_extrinsics_qos,
    PerceptionIngress,
)
import rclpy
from rclpy.context import Context
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy


def test_cam3_mapping_uses_one_synced_source_and_local_fanout():
    topics = ingress_topics('cam3')
    assert topics.camera == 'cam_3'
    assert topics.remote_color == '/synced/cam_3/color/image_raw/compressed'
    assert topics.remote_depth == '/synced/cam_3/depth/image_rect_raw/compressedDepth'
    assert topics.remote_extrinsics == '/synced/cam_3/extrinsics/depth_to_color'
    assert topics.local_color == '/perception/ingress/cam_3/color/image_raw/compressed'
    assert topics.local_depth == '/perception/ingress/cam_3/depth/image_rect_raw/compressedDepth'


def test_cam4_mapping_preserves_synced_calibration_contract():
    topics = ingress_topics('cam_4')
    assert topics.remote_color_info == '/synced/cam_4/color/camera_info'
    assert topics.remote_depth_info == '/synced/cam_4/depth/camera_info'
    assert topics.local_extrinsics == '/perception/ingress/cam_4/extrinsics/depth_to_color'


def test_head_mapping_uses_eir_color_aligned_depth_without_extrinsics():
    topics = ingress_topics('head')
    assert topics.remote_color == '/eir/camera/head/color/image_raw/compressed'
    assert topics.remote_depth == (
        '/eir/camera/head/aligned_depth_to_color/image_raw/compressedDepth'
    )
    assert topics.remote_depth_info == (
        '/eir/camera/head/aligned_depth_to_color/camera_info'
    )
    assert topics.local_depth == (
        '/perception/ingress/head/depth/image_rect_raw/compressedDepth'
    )
    assert camera_has_depth('head') is True


def test_only_supported_cameras_are_accepted():
    assert canonical_camera('3') == 'cam_3'
    assert canonical_camera('/synced/cam4/color/image_raw/compressed') == 'cam_4'
    assert canonical_camera('cam_1') == 'cam_1'
    assert canonical_camera('cam_2') == 'cam_2'
    assert canonical_camera('flir') == 'flir'


def test_realsense_cameras_are_depth_ready_while_flir_remains_rgb_only():
    assert camera_has_depth('cam_1') is True
    assert camera_has_depth('cam_2') is True
    assert camera_has_depth('cam_3') is True
    assert camera_has_depth('cam_4') is True
    assert camera_has_depth('head') is True
    assert camera_has_depth('flir') is False


def test_image_reader_qos_is_latest_best_effort_volatile():
    qos = image_reader_qos()
    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 1
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.durability == DurabilityPolicy.VOLATILE


def test_camera_info_is_reliable_while_extrinsics_is_latched():
    assert camera_info_qos().reliability == ReliabilityPolicy.RELIABLE
    assert camera_info_qos().durability == DurabilityPolicy.VOLATILE
    assert camera_info_qos().depth == 20
    assert local_extrinsics_qos().durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_ingress_node_lifecycle_keeps_rclpy_collections_intact():
    """Node construction and destruction catch private rclpy-name collisions."""
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = PerceptionIngress(context=context)
        assert node._context is context
        assert len(node._ingress_publishers) == 5
        assert len(node._ingress_subscriptions) == 5
        assert isinstance(node._publishers, list)
        assert isinstance(node._subscriptions, list)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_flir_ingress_does_not_advertise_phantom_depth_outputs():
    context = Context()
    node = None
    rclpy.init(
        args=['--ros-args', '-p', 'camera:=flir'], context=context)
    try:
        node = PerceptionIngress(context=context)
        assert node._has_depth is False
        assert set(node._ingress_publishers) == {'color', 'color_info'}
        assert len(node._ingress_subscriptions) == 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_head_ingress_advertises_rgbd_but_no_extrinsics():
    context = Context()
    node = None
    rclpy.init(
        args=['--ros-args', '-p', 'camera:=head'], context=context)
    try:
        node = PerceptionIngress(context=context)
        assert node._has_depth is True
        assert node._has_extrinsics is False
        assert set(node._ingress_publishers) == {
            'color', 'color_info', 'depth', 'depth_info'
        }
        assert len(node._ingress_subscriptions) == 4
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_forward_uses_ingress_owned_publishers_without_changing_message():
    class Publisher:
        """Capture published messages for the forwarding contract test."""

        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    node = PerceptionIngress.__new__(PerceptionIngress)
    publisher = Publisher()
    node._ingress_publishers = {'color': publisher}
    node._forwarded = {'color': 0}
    node._camera = 'cam_3'
    node.get_logger = lambda: SimpleNamespace(info=lambda *_: None)
    message = SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=12, nanosec=34)))

    node._forward('color', message)

    assert publisher.messages == [message]
    assert node._forwarded['color'] == 1
