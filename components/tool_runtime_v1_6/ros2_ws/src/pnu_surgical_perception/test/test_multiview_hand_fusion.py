import time
from types import SimpleNamespace

import rclpy
from hand_keypoint_interfaces.msg import (
    HandFacingArray,
    HandGestureArray,
    HandKeypoints,
)
from rclpy.context import Context

from pnu_surgical_perception.multiview_hand_fusion import (
    MultiviewHandFusion,
    QualitySelector,
    ViewObservation,
    observation_quality,
    synchronized_cohort,
    synchronized_history_cohort,
)


def _message_pair(camera: str, source_stamp_ns: int):
    keypoints = HandKeypoints()
    gestures = HandGestureArray()
    for message in (keypoints, gestures):
        message.header.frame_id = f'{camera}_color_optical_frame'
        message.header.stamp.sec = source_stamp_ns // 1_000_000_000
        message.header.stamp.nanosec = source_stamp_ns % 1_000_000_000
    return keypoints, gestures


def _observation(
    camera: str, source_stamp_ns: int, quality: float, hand_count: int
) -> ViewObservation:
    keypoints, gestures = _message_pair(camera, source_stamp_ns)
    return ViewObservation(
        camera=camera,
        keypoints=keypoints,
        gestures=gestures,
        source_stamp_ns=source_stamp_ns,
        received_monotonic=time.monotonic(),
        quality=quality,
        hand_count=hand_count,
    )


def test_quality_prefers_positive_broad_hand_over_collapsed_none():
    broad_points = [
        SimpleNamespace(u=float(100 + (index % 5) * 45),
                        v=float(100 + (index // 5) * 50))
        for index in range(21)
    ]
    collapsed_points = [
        SimpleNamespace(u=float(300 + index), v=float(220 + index * 0.2))
        for index in range(21)
    ]
    broad_hand = SimpleNamespace(
        hand_index=0, joints_2d=broad_points, has_handedness=True,
        handedness_score=0.9, kp_valid_depth=[False] * 21)
    collapsed_hand = SimpleNamespace(
        hand_index=0, joints_2d=collapsed_points, has_handedness=True,
        handedness_score=0.9, kp_valid_depth=[False] * 21)
    positive = SimpleNamespace(
        hand_index=0, has_classification=True,
        category_name='Open_Palm', score=0.85)
    rejected = SimpleNamespace(
        hand_index=0, has_classification=False,
        category_name='', score=0.0)
    broad_quality, _ = observation_quality(
        SimpleNamespace(hands=[broad_hand]), SimpleNamespace(hands=[positive]),
        width=1280, height=720)
    collapsed_quality, _ = observation_quality(
        SimpleNamespace(hands=[collapsed_hand]), SimpleNamespace(hands=[rejected]),
        width=1280, height=720)
    assert 0.0 < collapsed_quality < broad_quality < 1.0


def test_largest_synchronized_cohort_ignores_one_frame_leader():
    observations = {
        'cam_1': _observation('cam_1', 1_067_000_000, 0.8, 1),
        'cam_3': _observation('cam_3', 1_000_000_000, 0.7, 1),
        'cam_4': _observation('cam_4', 1_006_000_000, 0.7, 1),
    }
    assert set(synchronized_cohort(observations, 20_000_000)) == {
        'cam_3', 'cam_4'}


def test_recent_history_recovers_pair_when_latest_outputs_skip_different_frames():
    histories = {
        'cam_1': [
            _observation('cam_1', 1_000_000_000, 0.7, 1),
            _observation('cam_1', 1_067_000_000, 0.8, 1),
        ],
        'cam_3': [_observation('cam_3', 1_010_000_000, 0.75, 1)],
        'cam_4': [_observation('cam_4', 1_180_000_000, 0.9, 1)],
    }
    cohort = synchronized_history_cohort(histories, 20_000_000)
    assert set(cohort) == {'cam_1', 'cam_3'}
    assert cohort['cam_1'].source_stamp_ns == 1_000_000_000


def test_selector_switches_immediately_when_current_view_has_no_hand():
    selector = QualitySelector(switch_margin=0.12, switch_frames=3)
    selector.current = 'cam_1'
    chosen = selector.choose({
        'cam_1': _observation('cam_1', 1, 0.0, 0),
        'cam_3': _observation('cam_3', 2, 0.6, 1),
    })
    assert chosen == 'cam_3'


def test_out_of_order_topic_delivery_pairs_by_stamp_and_keeps_newest():
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = MultiviewHandFusion(context=context)
        k1, g1 = _message_pair('cam_1', 1_000_000_000)
        k2, g2 = _message_pair('cam_1', 1_067_000_000)
        node._on_gestures('cam_1', g1)
        node._on_gestures('cam_1', g2)
        node._on_keypoints('cam_1', k1)
        node._on_keypoints('cam_1', k2)
        assert node._last_finalized_stamp['cam_1'] == 1_067_000_000
        assert node._observations['cam_1'].source_stamp_ns == 1_067_000_000
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_identical_timer_ticks_do_not_advance_three_frame_hysteresis():
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = MultiviewHandFusion(context=context)
        node._selector.current = 'cam_1'
        for frame_index in range(3):
            base = 2_000_000_000 + frame_index * 67_000_000
            node._observations = {
                'cam_1': _observation('cam_1', base, 0.50, 1),
                'cam_3': _observation('cam_3', base + 10_000_000, 0.80, 1),
            }
            for camera, observation in node._observations.items():
                node._observation_history[camera][
                    observation.source_stamp_ns] = observation
            node._select_and_publish()
            node._select_and_publish()
            node._select_and_publish()
            if frame_index < 2:
                assert node._selector.current == 'cam_1'
        assert node._selector.current == 'cam_3'
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_late_facing_joins_only_the_last_published_observation_stamp():
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = MultiviewHandFusion(context=context)
        published = _observation('cam_1', 3_000_000_000, 0.7, 1)
        node._last_published_observation = published
        node._last_publish_signature = ('cam_1', 3_000_000_000)

        mismatched = HandFacingArray()
        mismatched.header.frame_id = 'cam_1_color_optical_frame'
        mismatched.header.stamp.sec = 3
        mismatched.header.stamp.nanosec = 10_000_000
        node._on_facing('cam_1', mismatched)
        assert node._last_facing_publish_signature is None

        matched = HandFacingArray()
        matched.header.frame_id = 'cam_1_color_optical_frame'
        matched.header.stamp.sec = 3
        node._on_facing('cam_1', matched)
        assert node._last_facing_publish_signature == ('cam_1', 3_000_000_000)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def test_history_cohort_never_republishes_an_older_source_stamp():
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = MultiviewHandFusion(context=context)
        node._selector.current = 'cam_1'
        node._last_published_source_stamp_ns = 4_000_000_000
        old = _observation('cam_1', 3_900_000_000, 0.8, 1)
        node._observations = {'cam_1': old}
        node._observation_history['cam_1'][old.source_stamp_ns] = old

        node._select_and_publish()

        assert node._last_publish_signature is None
        assert node._last_published_source_stamp_ns == 4_000_000_000
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)
