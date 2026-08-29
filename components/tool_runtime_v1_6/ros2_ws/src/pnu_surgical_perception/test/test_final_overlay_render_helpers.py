import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from hand_keypoint_interfaces.msg import Hand, HandKeypoints

from pnu_surgical_perception.final_overlay_compositor import (
    CAMERAS,
    LAYER_NAMES,
    FinalOverlayCompositor,
    LatestBase,
    LatestLayer,
    PanelOutput,
    RightEePalmDisplayFilter,
    ToolRoiOverlayConfig,
    camera_palm_axis_points,
    decode_binary_mask,
    draw_hand_roi_header,
    draw_hand_roi_overlay,
    draw_tool_roi_overlay,
    gesture_display_text,
    image_reader_qos,
    joined_facing_by_hand_index,
    load_tool_roi_profile,
    matched_humanoid_palm,
    normalized_polygon_pixel_points,
    normalized_roi_pixel_box,
    palm_facing_display_text,
    quaternion_matrix_xyzw,
    status_qos,
)
from pnu_surgical_perception.final_overlay_contract import Freshness


def test_binary_blood_mask_accepts_only_exact_mono8_shape():
    message = Image()
    message.height = 2
    message.width = 3
    message.step = 3
    message.encoding = 'mono8'
    message.data = bytes((0, 255, 0, 4, 0, 12))
    np.testing.assert_array_equal(
        decode_binary_mask(message),
        np.asarray([[False, True, False], [True, False, True]]),
    )
    message.encoding = 'rgb8'
    assert decode_binary_mask(message) is None


def test_pose_quaternion_identity_is_orthonormal():
    rotation = quaternion_matrix_xyzw(
        SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    )
    np.testing.assert_allclose(rotation, np.eye(3))


def test_camera_palm_axis_points_use_metric_orientation_and_reject_bad_input():
    palm = SimpleNamespace(
        translation=SimpleNamespace(x=0.10, y=-0.20, z=0.80),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    points = camera_palm_axis_points(palm, 0.08)
    assert points is not None
    np.testing.assert_allclose(points[0], (0.10, -0.20, 0.80))
    np.testing.assert_allclose(points[1], (0.18, -0.20, 0.80))
    np.testing.assert_allclose(points[2], (0.10, -0.12, 0.80))
    np.testing.assert_allclose(points[3], (0.10, -0.20, 0.88))
    assert camera_palm_axis_points(
        SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=1.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        ),
        0.08,
    ) is None


def _metric_right_hand_message() -> HandKeypoints:
    message = HandKeypoints()
    message.header.stamp.sec = 17
    message.header.stamp.nanosec = 7
    message.header.frame_id = 'cam_4_color_optical_frame'
    message.depth_source = 'real'
    hand = Hand()
    hand.hand_index = 3
    hand.has_handedness = True
    hand.handedness_label = 'Right'
    hand.handedness_score = 0.95
    hand.has_palm_6d = True
    hand.palm_6d.translation.z = 0.75
    hand.palm_6d.orientation.w = 1.0
    message.hands = [hand]
    return message


def test_humanoid_palm_hud_join_requires_same_source_and_valid_right_palm():
    hand_message = _metric_right_hand_message()
    humanoid_pose = PoseStamped()
    humanoid_pose.header.stamp.sec = hand_message.header.stamp.sec
    humanoid_pose.header.stamp.nanosec = hand_message.header.stamp.nanosec
    humanoid_pose.header.frame_id = 'humanoid'
    humanoid_pose.pose.position.x = 0.12
    humanoid_pose.pose.position.y = -0.33
    humanoid_pose.pose.position.z = 0.54
    humanoid_pose.pose.orientation.w = 1.0

    matched = matched_humanoid_palm(hand_message, humanoid_pose)
    assert matched is not None
    assert matched[0].hand_index == 3
    assert matched[1] == pytest.approx((0.12, -0.33, 0.54))

    humanoid_pose.header.stamp.nanosec += 1
    assert matched_humanoid_palm(hand_message, humanoid_pose) is None
    humanoid_pose.header.stamp.nanosec -= 1
    humanoid_pose.header.frame_id = 'tag1'
    assert matched_humanoid_palm(hand_message, humanoid_pose) is None


def test_normalized_hand_roi_maps_to_cam4_source_pixels():
    assert normalized_roi_pixel_box(
        (0.446094, 0.825, 0.108333, 0.741667), 1280, 720
    ) == (571, 78, 1056, 534)
    assert normalized_roi_pixel_box(
        (0.0, 1.0, 0.0, 1.0), 1280, 720
    ) == (0, 0, 1279, 719)


@pytest.mark.parametrize('roi', (
    (float('nan'), 0.8, 0.1, 0.7),
    (0.8, 0.4, 0.1, 0.7),
    (-0.1, 0.8, 0.1, 0.7),
    (0.4, 0.8, 0.7, 0.1),
    (0.4, 0.8, 0.1, 1.1),
))
def test_normalized_hand_roi_rejects_invalid_bounds(roi):
    with pytest.raises(ValueError):
        normalized_roi_pixel_box(roi, 1280, 720)


def test_hand_roi_overlay_draws_dashed_boundary_without_tinting_scene():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = draw_hand_roi_overlay(
        image, (0.446094, 0.825, 0.108333, 0.741667))
    assert box == (571, 78, 1056, 534)
    assert np.count_nonzero(image) > 0
    assert np.count_nonzero(image[78:82, 571:1057]) > 0
    # The ROI is an outline, not a tinted image crop.
    assert np.count_nonzero(image[300:340, 760:800]) == 0


def test_hand_roi_header_uses_free_right_side_of_status_bar():
    image = np.full((720, 1280, 3), (12, 24, 36), dtype=np.uint8)
    result = draw_hand_roi_header(
        image,
        (0.446094, 0.825, 0.108333, 0.741667),
        minimum_x=660,
    )
    assert result is not None
    label, text_x = result
    assert label.startswith('HAND ROI  x:0.446-0.825')
    assert text_x >= 660
    assert np.any(image[:46] != np.asarray((12, 24, 36), dtype=np.uint8))


def test_hand_roi_header_skips_label_when_header_has_no_free_space():
    image = np.full((720, 1280, 3), (12, 24, 36), dtype=np.uint8)
    before = image.copy()
    assert draw_hand_roi_header(
        image, (0.1, 0.9, 0.1, 0.9), minimum_x=1279) is None
    np.testing.assert_array_equal(image, before)


def test_normalized_tool_roi_polygon_maps_to_source_pixels():
    assert normalized_polygon_pixel_points(
        (0.25, 0.30, 0.75, 0.30, 0.75, 0.85, 0.20, 0.85),
        1280,
        720,
    ) == ((320, 216), (959, 216), (959, 611), (256, 611))


def test_tool_roi_overlay_draws_polygon_without_tinting_scene():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    points = draw_tool_roi_overlay(
        image,
        ToolRoiOverlayConfig(
            enabled=True,
            profile='cam3_live_tray',
            polygon_norm_xy=(
                0.25, 0.30, 0.75, 0.30, 0.75, 0.85, 0.20, 0.85,
            ),
        ),
    )
    assert points[0] == (320, 216)
    assert np.count_nonzero(image) > 0
    # The persistent guide is an outline and label, not an image tint.
    assert np.count_nonzero(image[380:420, 560:600]) == 0


def test_tool_roi_overlay_disabled_does_not_change_image():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    before = image.copy()
    assert draw_tool_roi_overlay(image, ToolRoiOverlayConfig()) == ()
    np.testing.assert_array_equal(image, before)


def test_load_tool_roi_profile_uses_worker_yaml_contract(tmp_path):
    profile = tmp_path / 'cam4_live_mayo.yaml'
    profile.write_text(
        '''/**:
  ros__parameters:
    workspace_roi_profile: cam4_live_mayo
    workspace_roi_enabled: true
    workspace_roi_polygon_norm_xy: [0.4, 0.1, 0.7, 0.1, 0.7, 0.7]
''',
        encoding='utf-8',
    )
    loaded = load_tool_roi_profile(str(profile))
    assert loaded.enabled
    assert loaded.profile == 'cam4_live_mayo'
    assert loaded.polygon_norm_xy == (0.4, 0.1, 0.7, 0.1, 0.7, 0.7)


def test_final_status_is_retained_reliable_and_final_images_are_latest_best_effort():
    status = status_qos()
    image = image_reader_qos()
    assert status.depth == 1
    assert status.reliability == ReliabilityPolicy.RELIABLE
    assert status.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert image.depth == 1
    assert image.reliability == ReliabilityPolicy.BEST_EFFORT


class _Published:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _compressed_source(sec, *, frame_id='camera_color_optical_frame'):
    message = CompressedImage()
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = 7
    message.header.frame_id = frame_id
    return message


def test_per_view_jpeg_keeps_its_own_header_and_deduplicates():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    node._jpeg_quality = 95
    node._panel_image_publishers = {'cam_3': _Published()}
    node._last_panel_signatures = {'cam_3': None}
    node._last_panel_source_headers = {'cam_3': None}
    image = np.full((24, 40, 3), 80, dtype=np.uint8)
    source = _compressed_source(101, frame_id='cam_3_color_optical_frame')

    node._publish_panel_if_changed('cam_3', image, ('live', 101), source.header)
    publisher = node._panel_image_publishers['cam_3']
    assert len(publisher.messages) == 1
    assert publisher.messages[0].header.stamp.sec == 101
    assert publisher.messages[0].header.frame_id == 'cam_3_color_optical_frame'

    # A CAM4/suction change elsewhere must not resend this CAM3 JPEG.
    node._publish_panel_if_changed('cam_3', image, ('live', 101), source.header)
    assert len(publisher.messages) == 1

    # The single stale clearing pane carries the last CAM3 source header,
    # not a synthesized timer timestamp.
    node._publish_panel_if_changed('cam_3', np.zeros_like(image), ('base', 'stale'), None)
    assert len(publisher.messages) == 2
    assert publisher.messages[-1].header.stamp.sec == 101
    assert publisher.messages[-1].header.frame_id == 'cam_3_color_optical_frame'


def test_node_context_attribute_cannot_shadow_camera_context_helper():
    """Construct a real Node: rclpy reserves ``Node._context`` for Context."""
    context = Context()
    node = None
    rclpy.init(context=context)
    try:
        node = FinalOverlayCompositor(context=context)
        assert node._context is context
        assert callable(node._camera_context)
        assert len(node._perception_subscriptions) == 19
        assert node._gesture_topic == '/perception/cam_4/hand/gestures'
        assert node._facing_topic == '/perception/cam_4/hand/facing'
        assert node._cam4_palm_pose_topic == (
            '/perception/cam_4/hand/palm_pose_humanoid')
        assert node._right_ee_hand_topic == '/perception/right_ee/hand/keypoints'
        assert node._enable_right_ee_hand
        assert node._enable_composite_output
        assert node._enable_per_view_output
        assert node._per_view_native_resolution
        assert node._per_view_jpeg_quality == 100
        assert node._panel_output_topics == {
            'cam_3': '/perception/cam_3/overlay/compressed',
            'cam_4': '/perception/cam_4/overlay/compressed',
            'suction': '/perception/suction/overlay/compressed',
            'right_ee': '/perception/right_ee/overlay/compressed',
        }
        assert set(node._panel_image_publishers) == {
            'cam_3', 'cam_4', 'suction', 'right_ee',
        }
        assert node._max_source_delta_ns == 1_800_000_000
        assert node._max_layer_age_sec == 1.5
        assert node._max_gesture_age_sec == 0.30
        assert node._max_gesture_source_delta_ns == 250_000_000
        assert node._max_right_ee_hand_age_sec == 0.30
        assert node._max_right_ee_hand_source_delta_ns == 250_000_000
        assert node._max_facing_age_sec == 0.35
        assert node._max_facing_source_delta_ns == 450_000_000
        assert node._show_cam4_hand_roi
        assert node._cam4_hand_roi == (0.0, 1.0, 0.0, 1.0)
        assert node._tool_rois == {
            camera: ToolRoiOverlayConfig() for camera in CAMERAS
        }
        # With no base images this returns early, but it must first create both
        # per-camera contexts without attempting to call ``Node._context``.
        node._publish_if_current()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown(context=context)


def _message(sec):
    return SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=7), frame_id='cam_color_optical_frame'))


def _gesture(
    hand_index=0, category='Open_Palm', score=0.875,
    has_classification=True, side='Right',
):
    return SimpleNamespace(
        hand_index=hand_index,
        has_handedness=bool(side),
        handedness_label=side,
        handedness_score=0.9,
        has_classification=has_classification,
        category_name=category,
        score=score,
    )


def _facing_hand(
    hand_index=0, label='PALM_UP', score=0.91,
    has_facing=True, side='Right',
):
    return SimpleNamespace(
        hand_index=hand_index,
        has_handedness=bool(side),
        handedness_label=side,
        has_facing=has_facing,
        facing_label=label,
        palm_up_score=score,
    )


def test_gesture_display_keeps_none_and_unclassified_distinct():
    assert gesture_display_text(_gesture(hand_index=2)) == 'H2 Right Open_Palm 0.88'
    assert gesture_display_text(_gesture(category='None')) == 'H0 Right None 0.88'
    assert gesture_display_text(
        _gesture(has_classification=False)
    ) == 'H0 Right Unclassified'


def test_palm_facing_display_keeps_unknown_and_signed_score_explicit():
    assert palm_facing_display_text(_facing_hand()) == (
        'H0 Right PALM_UP +0.91')
    assert palm_facing_display_text(
        _facing_hand(label='PALM_DOWN', score=-0.94)
    ) == 'H0 Right PALM_DOWN -0.94'
    assert palm_facing_display_text(
        _facing_hand(has_facing=False)
    ) == 'H0 Right UNKNOWN'
    assert palm_facing_display_text(
        _facing_hand(label='NOT_A_CONTRACT_LABEL')
    ) == 'H0 Right UNKNOWN'


def test_gesture_hud_renders_valid_rows_and_empty_frame_clears_it():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    image = np.zeros((240, 720, 3), dtype=np.uint8)
    message = _message(17)
    message.hands = [_gesture(), _gesture(hand_index=1, category='None')]
    node._draw_gesture_summary(image, message)
    assert np.count_nonzero(image) > 0

    empty_image = np.zeros_like(image)
    message.hands = []
    node._draw_gesture_summary(empty_image, message)
    assert np.count_nonzero(empty_image) == 0

    facing_only = np.zeros_like(image)
    node._draw_gesture_summary(
        facing_only, None, facing_rows=[_facing_hand()])
    assert np.count_nonzero(facing_only) > 0


def test_gesture_decision_uses_short_age_and_source_windows():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    node._base = {
        'cam_3': LatestBase(),
        'cam_4': LatestBase(source_stamp_ns=17_000_000_007),
    }
    node._gesture = LatestLayer(
        message=_message(17),
        source_stamp_ns=17_000_000_007,
        freshness=Freshness(now),
    )
    node._enable_gesture = True
    node._max_gesture_age_sec = 0.30
    node._max_gesture_source_delta_ns = 250_000_000

    decision = node._gesture_decision('cam_4', 'live', now)
    assert decision.state == 'live'
    assert decision.drawable
    assert node._gesture_decision('cam_3', 'live', now).state == 'disabled'

    node._gesture.freshness = Freshness(now - 0.31)
    assert node._gesture_decision('cam_4', 'live', now).state == 'stale'
    node._gesture.freshness = Freshness(now)
    node._gesture.source_stamp_ns += 250_000_001
    assert node._gesture_decision('cam_4', 'live', now).state == 'stale'


def test_facing_decision_uses_own_age_and_source_windows():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    node._base = {
        'cam_3': LatestBase(),
        'cam_4': LatestBase(source_stamp_ns=17_000_000_007),
    }
    node._facing = LatestLayer(
        message=_message(17),
        source_stamp_ns=17_000_000_007,
        freshness=Freshness(now),
    )
    node._enable_facing = True
    node._max_facing_age_sec = 0.35
    node._max_facing_source_delta_ns = 450_000_000

    decision = node._facing_decision('cam_4', 'live', now)
    assert decision.state == 'live'
    assert decision.drawable
    assert node._facing_decision('cam_3', 'live', now).state == 'disabled'

    node._facing.freshness = Freshness(now - 0.36)
    assert node._facing_decision('cam_4', 'live', now).state == 'stale'
    node._facing.freshness = Freshness(now)
    node._facing.source_stamp_ns += 450_000_001
    assert node._facing_decision('cam_4', 'live', now).state == 'stale'


def test_facing_join_requires_exact_stamp_and_unique_frame_local_index():
    hand_message = _message(17)
    hand_message.hands = [
        SimpleNamespace(hand_index=0),
        SimpleNamespace(hand_index=1),
    ]
    facing_message = _message(17)
    matching = _facing_hand(hand_index=0)
    facing_message.hands = [matching, _facing_hand(hand_index=2)]

    assert joined_facing_by_hand_index(hand_message, facing_message) == {
        0: matching,
    }

    facing_message.header.stamp.nanosec += 1
    assert joined_facing_by_hand_index(hand_message, facing_message) == {}
    facing_message.header.stamp.nanosec -= 1
    facing_message.hands = [matching, _facing_hand(hand_index=0)]
    assert joined_facing_by_hand_index(hand_message, facing_message) == {}

    hand_message.hands = [
        SimpleNamespace(hand_index=0),
        SimpleNamespace(hand_index=0),
    ]
    facing_message.hands = [matching]
    assert joined_facing_by_hand_index(hand_message, facing_message) == {}


def test_gesture_thresholds_do_not_shorten_general_result_layers():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    node._base = {
        'cam_3': LatestBase(),
        'cam_4': LatestBase(source_stamp_ns=17_000_000_007),
    }
    node._layers = {
        camera: {name: LatestLayer() for name in LAYER_NAMES}
        for camera in CAMERAS
    }
    node._layers['cam_4']['tool'] = LatestLayer(
        message=_message(18),
        source_stamp_ns=18_000_000_007,
        freshness=Freshness(now - 0.31),
    )
    node._enable_hand = True
    node._enable_blood = True
    node._max_layer_age_sec = 1.5
    node._max_source_delta_ns = 1_800_000_000

    decision = node._layer_decision('cam_4', 'tool', 'live', now)
    assert decision.state == 'live'
    assert decision.drawable


def _hand_keypoints(sec=17, *, joint_count=21):
    message = _message(sec)
    message.hands = [SimpleNamespace(
        hand_index=0,
        has_handedness=True,
        handedness_label='Right',
        joints_2d=[
            SimpleNamespace(u=90 + (index % 5) * 18, v=150 + (index // 5) * 18)
            for index in range(joint_count)
        ],
    )]
    return message


def test_right_ee_hand_layer_requires_fresh_source_aligned_skeleton():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    source_stamp = 17_000_000_007
    node._enable_right_ee_panel = True
    node._enable_right_ee_hand = True
    node._max_base_age_sec = 1.0
    node._max_gesture_age_sec = 0.30
    node._max_gesture_source_delta_ns = 250_000_000
    node._max_right_ee_hand_age_sec = 0.30
    node._max_right_ee_hand_source_delta_ns = 250_000_000
    node._right_ee_base = LatestBase(
        message=_message(17), image=np.zeros((240, 320, 3), dtype=np.uint8),
        source_stamp_ns=source_stamp, freshness=Freshness(now),
    )
    node._right_ee_hand = LatestLayer(
        message=_hand_keypoints(), source_stamp_ns=source_stamp,
        freshness=Freshness(now), count=1,
    )
    node._right_ee_gesture = LatestLayer()

    context = node._right_ee_context(now)
    assert context['hand_state'] == 'live'
    assert context['hand_drawable']

    node._right_ee_hand.source_stamp_ns += 250_000_001
    context = node._right_ee_context(now)
    assert context['hand_state'] == 'stale'
    assert not context['hand_drawable']

    node._right_ee_hand.source_stamp_ns = source_stamp
    node._right_ee_hand.freshness = Freshness(now - 0.31)
    context = node._right_ee_context(now)
    assert context['hand_state'] == 'stale'
    assert not context['hand_drawable']


def test_right_ee_panel_draws_only_a_valid_aligned_hand_skeleton():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    source_stamp = 17_000_000_007
    base = LatestBase(
        message=_message(17), image=np.zeros((240, 320, 3), dtype=np.uint8),
        source_stamp_ns=source_stamp, freshness=Freshness(now),
    )
    node._enable_right_ee_panel = True
    node._right_ee_hand = LatestLayer(
        message=_hand_keypoints(), source_stamp_ns=source_stamp,
        freshness=Freshness(now), count=1,
    )
    node._right_ee_gesture = LatestLayer()
    node._panel_width = 320
    node._panel_height = 240

    panel = node._render_right_ee_panel({
        'base_state': 'live',
        'selected': base,
        'hand_state': 'live',
        'hand_drawable': True,
        'gesture_state': 'missing',
        'gesture_drawable': False,
    })
    # The palm banner occupies y<130. A nonzero bottom-region pixel proves
    # the source-aligned skeleton, rather than only HUD text, was rendered.
    assert np.count_nonzero(panel[145:, :]) > 0

    node._right_ee_hand.message = _hand_keypoints(joint_count=20)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    node._draw_hands(blank, node._right_ee_hand.message)
    assert np.count_nonzero(blank) == 0


def test_suction_native_overlay_preserves_ingress_resolution():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    node._enable_suction_panel = True
    node._suction_mask = LatestLayer()
    node._panel_width = 960
    node._panel_height = 540
    source = _compressed_source(
        17, frame_id='eir_suction_camera_color_optical_frame')
    base = LatestBase(
        message=source,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        source_stamp_ns=17_000_000_007,
    )
    context = {
        'base_state': 'live',
        'selected': base,
        'mask_state': 'missing',
        'mask_drawable': False,
    }

    native = node._render_suction_panel(0.0, context, native=True)
    display_panel = node._render_suction_panel(0.0, context)
    assert native.shape == (720, 1280, 3)
    assert display_panel.shape == (540, 960, 3)
    assert np.count_nonzero(native) > 0


def test_right_ee_palm_filter_needs_four_unique_stamps_before_closed():
    filter_ = RightEePalmDisplayFilter(hold_sec=0.25)

    # The H/I transition was a three-frame half-curl followed by loss of the
    # skeleton. It must never present as CLOSED, even if each of those three
    # raw frames says Closed_Fist.
    for stamp, now in ((301, 20.10), (302, 20.17), (303, 20.24)):
        assert filter_.update(
            category='Closed_Fist', score=0.70, hand_present=True,
            source_stamp_ns=stamp, now=now,
        ) == ('', None, False)
    assert filter_.update(
        category='', score=None, hand_present=False,
        source_stamp_ns=304, now=20.31,
    ) == ('', None, False)

    # A renderer re-publishing one received message cannot manufacture a
    # second vote for it.
    assert filter_.update(
        category='Closed_Fist', score=0.70, hand_present=True,
        source_stamp_ns=400, now=26.00,
    ) == ('', None, False)
    assert filter_.update(
        category='Closed_Fist', score=0.70, hand_present=True,
        source_stamp_ns=400, now=26.07,
    ) == ('', None, False)
    # A late source frame is not allowed to become an additional temporal
    # sample; it clears rather than counting out-of-order evidence.
    assert filter_.update(
        category='Closed_Fist', score=0.70, hand_present=True,
        source_stamp_ns=399, now=26.08,
    ) == ('', None, False)


def test_right_ee_palm_filter_replays_e_f_g_as_closed_after_strong_rule():
    """E/F/G are all Closed once the right-EE raw rule is applied.

    The replay uses the source-stamp ordering from the user-reviewed capture:
    the strong raw classifier gives the sustained run before E and emits
    Closed for E=347, F=363, and G=364.  No long blind hold is needed.
    """
    filter_ = RightEePalmDisplayFilter(hold_sec=0.25)
    labels = []
    for stamp, score in (
        (342, 0.78), (343, 0.76), (344, 0.74), (345, 0.72),
        (347, 0.70),  # E, old geometry None
        (363, 0.79),  # F, already raw Closed
        (364, 0.73),  # G, old geometry None
    ):
        category, accepted_score, held = filter_.update(
            category='Closed_Fist', score=score, hand_present=True,
            source_stamp_ns=stamp, now=stamp / 15.0,
        )
        labels.append((stamp, category, accepted_score, held))

    assert labels[:3] == [
        (342, '', None, False),
        (343, '', None, False),
        (344, '', None, False),
    ]
    assert labels[3] == (345, 'Closed_Fist', 0.72, False)
    assert labels[4] == (347, 'Closed_Fist', 0.70, False)
    assert labels[5] == (363, 'Closed_Fist', 0.79, False)
    assert labels[6] == (364, 'Closed_Fist', 0.73, False)


def test_right_ee_palm_filter_holds_only_brief_none_and_open_or_no_hand_clear():
    filter_ = RightEePalmDisplayFilter(hold_sec=0.25)
    for stamp in range(500, 504):
        category, _, held = filter_.update(
            category='Closed_Fist', score=0.62, hand_present=True,
            source_stamp_ns=stamp, now=10.0 + (stamp - 500) * 0.06,
        )
    assert (category, held) == ('Closed_Fist', False)

    # A single geometry rejection may only retain a confirmed label for the
    # existing 250-ms display grace; it is not a 1-second blind latch.
    assert filter_.update(
        category='', score=None, hand_present=True,
        source_stamp_ns=504, now=10.20,
    ) == ('Closed_Fist', 0.62, True)
    assert filter_.update(
        category='', score=None, hand_present=True,
        source_stamp_ns=505, now=10.49,
    ) == ('', None, False)

    # Explicit OPEN overrides immediately. An explicit zero-hand frame resets
    # immediately, even if it arrives inside a formerly valid grace window.
    assert filter_.update(
        category='Open_Palm', score=0.91, hand_present=True,
        source_stamp_ns=506, now=10.55,
    ) == ('Open_Palm', 0.91, False)
    assert filter_.update(
        category='', score=None, hand_present=False,
        source_stamp_ns=507, now=10.56,
    ) == ('', None, False)


def test_right_ee_palm_renderer_requires_exact_gesture_skeleton_source_match():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    node._right_ee_palm_hold_sec = 0.25
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    hand_message = _hand_keypoints()
    closed = _message(17)
    closed.hands = [_gesture(category='Closed_Fist', score=0.62)]

    # A mismatched gesture/skeleton source pair is unknown, not an implicit
    # vote. This keeps the temporal UI filter source-frame honest.
    label, _ = node._draw_right_ee_palm_state(
        image, closed, hand_message=hand_message,
        source_stamp_ns=None, now=10.0,
    )
    assert label == 'UNKNOWN'


def test_both_expired_bases_publish_one_clearing_stale_frame():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()

    def compressed_message(sec):
        message = CompressedImage()
        message.header.stamp.sec = sec
        message.header.stamp.nanosec = 7
        message.header.frame_id = 'cam_color_optical_frame'
        return message

    cam3_source = compressed_message(31)
    cam4_source = compressed_message(30)
    node._base = {
        'cam_3': LatestBase(
            message=cam3_source,
            image=np.full((180, 320, 3), 255, dtype=np.uint8),
            source_stamp_ns=31_000_000_007,
            freshness=Freshness(now - 2.0),
        ),
        'cam_4': LatestBase(
            message=cam4_source,
            image=np.full((180, 320, 3), 255, dtype=np.uint8),
            source_stamp_ns=30_000_000_007,
            freshness=Freshness(now - 2.0),
        ),
    }
    node._layers = {
        camera: {name: LatestLayer() for name in LAYER_NAMES}
        for camera in CAMERAS
    }
    node._gesture = LatestLayer()
    node._facing = LatestLayer()
    node._enable_hand = True
    node._enable_gesture = True
    node._enable_facing = True
    node._enable_blood = True
    node._max_base_age_sec = 1.0
    node._max_layer_age_sec = 1.5
    node._max_source_delta_ns = 1_800_000_000
    node._max_gesture_age_sec = 0.30
    node._max_gesture_source_delta_ns = 250_000_000
    node._max_facing_age_sec = 0.35
    node._max_facing_source_delta_ns = 450_000_000
    node._panel_width = 320
    node._panel_height = 180
    node._jpeg_quality = 95
    node._image_publisher = _Published()
    previous_output = compressed_message(29)
    node._last_output_at = now - 0.1
    node._last_output_source_header = previous_output.header
    node._last_output_bytes = 1
    node._last_output_width = 640
    node._last_output_height = 180
    node._output_hz = 10.0
    node._last_signature = ('live',)
    node._panel_image_publishers = {
        'cam_3': _Published(),
        'cam_4': _Published(),
    }
    node._last_panel_signatures = {
        'cam_3': ('live',),
        'cam_4': ('live',),
    }
    node._last_panel_source_headers = {
        'cam_3': cam3_source.header,
        'cam_4': cam4_source.header,
    }

    node._publish_if_current()
    assert len(node._image_publisher.messages) == 1
    output = node._image_publisher.messages[0]
    assert output.header.stamp.sec == 29
    assert node._last_signature[0] == 'clear'
    decoded = cv2.imdecode(
        np.frombuffer(output.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # The current operator view is a 2x2 canvas (CAM3/CAM4/suction/right-EE),
    # so clearing stale upper panels still publishes the full two-row image.
    assert decoded.shape[:2] == (360, 640)
    # Expired white source images must be replaced by black stale panels,
    # proving that the compositor did not republish or visually hold intent.
    assert float(np.mean(decoded)) < 40.0
    # Clearing each individual panel retains its own last RGB header, rather
    # than borrowing the global 2x2 anchor (sec=29 above).
    assert len(node._panel_image_publishers['cam_3'].messages) == 1
    assert len(node._panel_image_publishers['cam_4'].messages) == 1
    assert node._panel_image_publishers['cam_3'].messages[-1].header.stamp.sec == 31
    assert node._panel_image_publishers['cam_4'].messages[-1].header.stamp.sec == 30

    node._publish_if_current()
    assert len(node._image_publisher.messages) == 1
    assert len(node._panel_image_publishers['cam_3'].messages) == 1
    assert len(node._panel_image_publishers['cam_4'].messages) == 1


def test_status_json_has_exact_compact_public_contract():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    source = _message(31)
    node._base = {
        'cam_3': LatestBase(
            message=source, image=np.zeros((2, 2, 3), dtype=np.uint8),
            source_stamp_ns=31_000_000_007, freshness=Freshness(now), received=4),
        # A known but stale CAM4 base remains representable without a null
        # stamp/age; that is different from startup, where no status is sent.
        'cam_4': LatestBase(
            message=_message(29), image=np.zeros((2, 2, 3), dtype=np.uint8),
            source_stamp_ns=29_000_000_007, freshness=Freshness(now - 2.0), received=1),
    }
    node._layers = {
        camera: {name: LatestLayer() for name in LAYER_NAMES}
        for camera in CAMERAS
    }
    node._layers['cam_3']['tool'] = LatestLayer(
        message=_message(31), source_stamp_ns=31_000_000_007,
        freshness=Freshness(now), count=2)
    node._max_base_age_sec = 1.0
    node._max_layer_age_sec = 1.0
    node._max_source_delta_ns = 1_000_000
    node._max_gesture_age_sec = 0.30
    node._max_gesture_source_delta_ns = 250_000_000
    node._max_facing_age_sec = 0.35
    node._max_facing_source_delta_ns = 450_000_000
    node._enable_hand = True
    node._enable_gesture = True
    node._gesture = LatestLayer()
    node._enable_facing = True
    node._facing = LatestLayer()
    node._enable_blood = True
    node._output_hz = 9.5
    node._last_output_at = now
    node._last_output_source_header = source.header
    node._last_output_bytes = 123
    node._last_output_width = 960
    node._last_output_height = 540
    node._status_publisher = _Published()

    node._publish_status()
    payload = json.loads(node._status_publisher.messages[-1].data)
    assert payload['schema'] == 'pnu.perception.final_overlay.v1'
    assert set(payload) == {'schema', 'published_at', 'output', 'cameras'}
    assert payload['output'] == {
        'source_stamp': {'sec': 31, 'nanosec': 7},
        'hz': 9.5, 'bytes': 123, 'width': 960, 'height': 540,
    }
    assert payload['cameras']['cam3']['base']['received'] == 4
    assert payload['cameras']['cam3']['layers']['tool']['state'] == 'live'
    assert payload['cameras']['cam3']['layers']['hand']['state'] == 'disabled'
    assert payload['cameras']['cam3']['layers']['blood']['state'] == 'disabled'
    assert payload['cameras']['cam3']['layers']['hand']['source_stamp'] is None
    assert payload['cameras']['cam3']['layers']['hand']['age_sec'] is None
    assert payload['cameras']['cam3']['layers']['blood']['source_stamp'] is None
    assert payload['cameras']['cam3']['layers']['blood']['age_sec'] is None
    assert payload['cameras']['cam4']['state'] == 'stale'
    for camera in ('cam3', 'cam4'):
        assert set(payload['cameras'][camera]['layers']) == set(LAYER_NAMES)
        assert payload['cameras'][camera]['base']['source_stamp'] is not None
        assert payload['cameras'][camera]['base']['age_sec'] is not None


def test_status_is_suppressed_until_all_base_metadata_is_truthful():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    node._base = {
        'cam_3': LatestBase(
            message=_message(31), image=np.zeros((2, 2, 3), dtype=np.uint8),
            source_stamp_ns=31_000_000_007, freshness=Freshness(now), received=1),
        'cam_4': LatestBase(),
    }
    node._layers = {
        camera: {name: LatestLayer() for name in LAYER_NAMES}
        for camera in CAMERAS
    }
    node._max_base_age_sec = 1.0
    node._max_layer_age_sec = 1.0
    node._max_source_delta_ns = 1_000_000
    node._max_gesture_age_sec = 0.30
    node._max_gesture_source_delta_ns = 250_000_000
    node._max_facing_age_sec = 0.35
    node._max_facing_source_delta_ns = 450_000_000
    node._enable_hand = True
    node._enable_gesture = True
    node._gesture = LatestLayer()
    node._enable_facing = True
    node._facing = LatestLayer()
    node._enable_blood = True
    node._last_output_at = now
    node._last_output_source_header = _message(31).header
    node._output_hz = 1.0
    node._last_output_bytes = 1
    node._last_output_width = 1
    node._last_output_height = 1
    node._status_publisher = _Published()

    node._publish_status()
    assert node._status_publisher.messages == []


def test_status_output_stamp_is_the_last_encoded_image_not_a_newer_base_callback():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    now = time.monotonic()
    node._base = {
        'cam_3': LatestBase(
            message=_message(42), image=np.zeros((2, 2, 3), dtype=np.uint8),
            source_stamp_ns=42_000_000_007, freshness=Freshness(now), received=3),
        'cam_4': LatestBase(
            message=_message(41), image=np.zeros((2, 2, 3), dtype=np.uint8),
            source_stamp_ns=41_000_000_007, freshness=Freshness(now), received=2),
    }
    node._layers = {
        camera: {name: LatestLayer() for name in LAYER_NAMES}
        for camera in CAMERAS
    }
    node._max_base_age_sec = 1.0
    node._max_layer_age_sec = 1.0
    node._max_source_delta_ns = 1_000_000
    node._max_gesture_age_sec = 0.30
    node._max_gesture_source_delta_ns = 250_000_000
    node._max_facing_age_sec = 0.35
    node._max_facing_source_delta_ns = 450_000_000
    node._enable_hand = True
    node._enable_gesture = True
    node._gesture = LatestLayer()
    node._enable_facing = True
    node._facing = LatestLayer()
    node._enable_blood = True
    node._last_output_at = now
    # Simulate a final JPEG encoded from an earlier CAM4 base.  The newer
    # CAM3 callback above must not silently overwrite this status identity.
    node._last_output_source_header = _message(40).header
    node._output_hz = 8.0
    node._last_output_bytes = 10
    node._last_output_width = 20
    node._last_output_height = 10
    node._status_publisher = _Published()

    node._publish_status()
    payload = json.loads(node._status_publisher.messages[-1].data)
    assert payload['output']['source_stamp'] == {'sec': 40, 'nanosec': 7}
    camera_palm_axis_points,

def _tiny_jpeg_source(sec: int, value: int) -> CompressedImage:
    message = _compressed_source(sec)
    success, encoded = cv2.imencode(
        '.jpg', np.full((12, 20, 3), value, dtype=np.uint8))
    assert success
    message.data = encoded.tobytes()
    return message


def test_latest_base_staging_coalesces_superseded_jpegs_before_decode():
    state = LatestBase()
    first = _tiny_jpeg_source(51, 20)
    second = _tiny_jpeg_source(52, 90)

    FinalOverlayCompositor._stage_latest_base(state, first)
    FinalOverlayCompositor._stage_latest_base(state, second)
    assert state.message is None
    assert state.received == 2

    assert FinalOverlayCompositor._drain_latest_base(state)
    assert state.message is second
    assert state.image is not None
    assert state.image.shape == (12, 20, 3)
    assert state.processed_sequence == 2
    assert state.dropped == 1


def test_status_output_falls_back_to_native_panel_without_composite():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    source = _compressed_source(71, frame_id='cam_3_color_optical_frame')
    node._last_output_at = None
    node._last_output_source_header = None
    node._last_panel_outputs = {
        'cam_3': PanelOutput(
            source_header=source.header, published_at=10.0, hz=14.8,
            bytes=321, width=1280, height=720),
    }

    output = node._status_output()
    assert output is not None
    assert output.source_header.stamp.sec == 71
    assert output.hz == pytest.approx(14.8)
    assert (output.width, output.height, output.bytes) == (1280, 720, 321)


def test_per_view_render_skips_unchanged_unsubscribed_panels():
    node = FinalOverlayCompositor.__new__(FinalOverlayCompositor)
    rendered = []
    context = {'base_state': 'missing'}
    node._drain_latest_bases = lambda: None
    node._camera_context = lambda _camera, _now: context
    node._suction_context = lambda _now: {'selected': None}
    node._right_ee_context = lambda _now: {'selected': None}
    node._camera_panel_signature = lambda camera, _context: (camera,)
    node._suction_panel_signature = lambda _context: ('suction',)
    node._right_ee_panel_signature = lambda _context: ('right_ee',)
    node._enable_composite_output = False
    node._image_publisher = None
    node._panel_image_publishers = {'cam_3': _Published()}
    node._last_panel_signatures = {'cam_3': None}
    node._last_output_at = None
    node._last_output_source_header = None
    node._render_camera_panel = lambda camera, _context, native: (
        rendered.append((camera, native)) or np.zeros((2, 2, 3), dtype=np.uint8))
    node._publish_panel_if_changed = lambda *_args: None
    node._panel_source_header = lambda *_args: None

    node._publish_if_current()
    assert rendered == [('cam_3', True)]
