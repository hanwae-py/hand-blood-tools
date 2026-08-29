"""DDS compatibility checks for the one-ingress local image contract."""

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSCompatibility,
    QoSProfile,
    ReliabilityPolicy,
    qos_check_compatible,
)

from hand_keypoint_ros.hand_detection_node import (
    camera_info_qos as hand_camera_info_qos,
    image_reader_qos as hand_image_reader_qos,
)
from surgical_task_coordinator.blood_detection_node import (
    camera_info_qos as blood_camera_info_qos,
    image_reader_qos as blood_image_reader_qos,
)

from pnu_surgical_perception.final_overlay_compositor import (
    camera_info_qos as final_camera_info_qos,
    image_reader_qos as final_image_reader_qos,
)
from pnu_surgical_perception.native_depth_pose_node import (
    camera_info_qos as tool_camera_info_qos,
    depth_to_color_extrinsics_qos,
    image_reader_qos as tool_image_reader_qos,
)
from pnu_surgical_perception.perception_ingress import (
    camera_info_qos as ingress_camera_info_qos,
    image_reader_qos as ingress_image_reader_qos,
    local_extrinsics_qos,
)


def _assert_compatible(publisher, subscription):
    compatibility, reason = qos_check_compatible(publisher, subscription)
    assert compatibility != QoSCompatibility.ERROR, reason


def test_external_reliable_camera_output_can_feed_best_effort_ingress_reader():
    viplab_image_writer = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    _assert_compatible(viplab_image_writer, ingress_image_reader_qos())


def test_local_image_fanout_is_compatible_with_every_current_image_reader():
    local_writer = ingress_image_reader_qos()
    for reader in (
        tool_image_reader_qos(),
        hand_image_reader_qos(),
        blood_image_reader_qos(),
        final_image_reader_qos(),
    ):
        _assert_compatible(local_writer, reader)
        assert reader.reliability == ReliabilityPolicy.BEST_EFFORT
        assert reader.durability == DurabilityPolicy.VOLATILE
        assert reader.depth == 1


def test_camera_info_and_extrinsics_keep_their_separate_reliable_contracts():
    local_info_writer = ingress_camera_info_qos()
    for reader in (
        tool_camera_info_qos(),
        blood_camera_info_qos(),
        final_camera_info_qos(),
    ):
        _assert_compatible(local_info_writer, reader)
        assert reader.reliability == ReliabilityPolicy.RELIABLE
        assert reader.durability == DurabilityPolicy.VOLATILE
        assert reader.depth == 20
    # Hand inference is intentionally latest-frame-only. Repeated per-frame
    # CameraInfo must not accumulate behind a slow MediaPipe/depth callback.
    hand_reader = hand_camera_info_qos()
    _assert_compatible(local_info_writer, hand_reader)
    assert hand_reader.reliability == ReliabilityPolicy.RELIABLE
    assert hand_reader.durability == DurabilityPolicy.VOLATILE
    assert hand_reader.depth == 1
    _assert_compatible(local_extrinsics_qos(), depth_to_color_extrinsics_qos())


def test_test_would_reject_the_old_reliable_image_subscriber_regression():
    old_reader = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    compatibility, _ = qos_check_compatible(ingress_image_reader_qos(), old_reader)
    assert compatibility == QoSCompatibility.ERROR
