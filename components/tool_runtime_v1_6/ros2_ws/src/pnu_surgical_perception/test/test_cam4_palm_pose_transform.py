"""Unit tests for the perception-only CAM4 palm transform contract."""

from __future__ import annotations

import math

import pytest
from geometry_msgs.msg import TransformStamped
from hand_keypoint_interfaces.msg import Hand, HandKeypoints

from pnu_surgical_perception.cam4_palm_pose_transform import (
    compose_target_palm,
    pose_stamped_from_components,
    select_camera_palm,
)


def _message(*, depth_source: str = 'real') -> HandKeypoints:
    message = HandKeypoints()
    message.header.stamp.sec = 81
    message.header.stamp.nanosec = 123_456_789
    message.header.frame_id = 'cam_4_color_optical_frame'
    message.depth_source = depth_source
    return message


def _right_palm(*, hand_index: int = 0) -> Hand:
    hand = Hand()
    hand.hand_index = hand_index
    hand.has_handedness = True
    hand.handedness_label = 'Right'
    hand.handedness_score = 0.92
    hand.has_palm_6d = True
    hand.palm_6d.translation.x = 0.10
    hand.palm_6d.translation.y = -0.20
    hand.palm_6d.translation.z = 0.80
    # Deliberately non-unit: the transform path must normalize it.
    hand.palm_6d.orientation.z = 0.0
    hand.palm_6d.orientation.w = 2.0
    return hand


def test_selects_exactly_one_metric_right_palm_and_normalizes_orientation():
    message = _message()
    message.hands = [_right_palm()]

    selected, reason = select_camera_palm(message)

    assert reason == 'SELECTED'
    assert selected is not None
    assert selected.hand_index == 0
    assert selected.translation == pytest.approx((0.10, -0.20, 0.80))
    assert selected.quaternion_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_rejects_nonmetric_or_ambiguous_palm_evidence():
    rgb_only = _message(depth_source='rgb_only')
    rgb_only.hands = [_right_palm()]
    assert select_camera_palm(rgb_only) == (None, 'DEPTH_SOURCE_NOT_REAL')

    ambiguous = _message()
    ambiguous.hands = [_right_palm(hand_index=0), _right_palm(hand_index=1)]
    assert select_camera_palm(ambiguous) == (None, 'AMBIGUOUS_SELECTED_PALMS')


def test_composes_target_from_camera_at_the_same_source_stamp():
    message = _message()
    message.hands = [_right_palm()]
    camera_palm, _ = select_camera_palm(message)
    assert camera_palm is not None
    transform = TransformStamped()
    transform.header.frame_id = 'humanoid'
    transform.child_frame_id = 'cam_4_color_optical_frame'
    transform.transform.translation.x = 1.0
    transform.transform.rotation.z = math.sqrt(0.5)
    transform.transform.rotation.w = math.sqrt(0.5)

    humanoid_palm, reason = compose_target_palm(transform, camera_palm)

    assert reason == 'COMPOSED'
    assert humanoid_palm is not None
    # +90 degrees around Z turns camera +X into target +Y.
    assert humanoid_palm.translation == pytest.approx((1.20, 0.10, 0.80))
    assert humanoid_palm.quaternion_xyzw == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    pose = pose_stamped_from_components(message.header, 'humanoid', humanoid_palm)
    assert pose.header.stamp.sec == 81
    assert pose.header.stamp.nanosec == 123_456_789
    assert pose.header.frame_id == 'humanoid'
