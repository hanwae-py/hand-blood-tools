from collections import OrderedDict
from types import SimpleNamespace
import warnings

import numpy as np

from hand_keypoint_ros.core import (
    CANNED_GESTURE_NAMES,
    GESTURE_OUTPUT_NAMES,
    gesture_rows_from_result,
    inference_crop_box,
    normalize_canned_gesture,
    process_frame,
    remap_result_landmarks,
    sample_depth_batch,
)
from hand_keypoint_ros.topview_gesture import (
    CLASSIFIER_NAME,
    CLASSIFIER_SHA256,
    CLASSIFIER_VERSION,
    GESTURE_PROFILES,
    GESTURE_NAMES,
    OUTPUT_NAMES,
    RIGHT_EE_CLASSIFIER_VERSION,
    classify,
)
from hand_keypoint_ros.hand_detection_node import (
    newest_ready_synced_message_id,
)


def _category(name, score):
    return SimpleNamespace(category_name=name, score=score)


def test_official_contract_has_seven_positive_classes_and_none_rejection():
    assert CANNED_GESTURE_NAMES == (
        'Closed_Fist',
        'Open_Palm',
        'Pointing_Up',
        'Thumb_Down',
        'Thumb_Up',
        'Victory',
        'ILoveYou',
    )
    assert GESTURE_OUTPUT_NAMES == ('None',) + CANNED_GESTURE_NAMES


def test_parallel_join_uses_newest_ready_pair_without_chasing_latest_sync():
    stamp_1 = (11, 100, 'cam_4')
    stamp_2 = (11, 200, 'cam_4')
    pending = OrderedDict((
        (101, (stamp_1, 1, object(), object(), object())),
        (202, (stamp_2, 2, object(), object(), object())),
    ))
    recognitions = OrderedDict((
        (101, (stamp_1, object(), object(), 4.0, ('delivery', 1))),
    ))

    # Frame 1 remains processable while synchronized frame 2 is waiting for
    # its inference result; the old single-slot join returned no work here.
    assert newest_ready_synced_message_id(pending, recognitions) == 101

    recognitions[202] = (
        stamp_2, object(), object(), 4.0, ('delivery', 2))
    assert newest_ready_synced_message_id(pending, recognitions) == 202


def test_topview_public_contract_has_two_positive_classes_and_none_rejection():
    assert GESTURE_NAMES == ('Closed_Fist', 'Open_Palm')
    assert OUTPUT_NAMES == ('None', 'Closed_Fist', 'Open_Palm')
    assert CLASSIFIER_NAME == 'VIPLab Top-View Landmark Gesture Classifier'
    assert CLASSIFIER_VERSION == 'landmark-geometry-v2-world-closed'
    assert len(CLASSIFIER_SHA256) == 64


def test_rgb_only_all_nan_depth_is_invalid_without_runtime_warning():
    depth = np.full((40, 60), np.nan, dtype=np.float32)
    uv = np.asarray([[10.0, 10.0], [30.0, 20.0]], dtype=np.float32)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        values, valid = sample_depth_batch(depth, uv)
    assert not captured
    np.testing.assert_array_equal(values, [0.0, 0.0])
    np.testing.assert_array_equal(valid, [False, False])


def test_none_is_a_valid_classification_result():
    result = normalize_canned_gesture([_category('None', 0.81)])
    assert result == {
        'has_gesture': True,
        'category_name': 'None',
        'score': 0.81,
    }


def test_no_public_result_is_not_fabricated():
    result = normalize_canned_gesture([])
    assert result == {
        'has_gesture': False,
        'category_name': '',
        'score': 0.0,
    }


def test_unknown_category_is_not_published_as_official():
    result = normalize_canned_gesture([_category('Custom_Gesture', 0.99)])
    assert result['has_gesture'] is False


class _FakeMp:
    class ImageFormat:
        SRGB = 'srgb'

    class Image:
        def __init__(self, *, image_format, data):
            self.image_format = image_format
            self.data = data


def _hand_points(open_fingers=(0, 1, 2, 3)):
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = (0.0, 0.0)
    points[1:5] = np.asarray([
        (-0.18, -0.02), (-0.34, -0.12), (-0.46, -0.22),
        (-0.58, -0.30),
    ])
    for finger_index, (mcp, pip, dip, tip), (x, y) in zip(
        range(4),
        ((5, 6, 7, 8), (9, 10, 11, 12),
         (13, 14, 15, 16), (17, 18, 19, 20)),
        ((-0.32, -0.20), (0.0, -0.25),
         (0.30, -0.20), (0.55, -0.10)),
    ):
        points[mcp] = (x, y)
        points[pip] = (x, y - 0.40)
        if finger_index in open_fingers:
            points[dip] = (x, y - 0.70)
            points[tip] = (x, y - 1.00)
        else:
            points[dip] = (x + 0.25, y - 0.35)
            points[tip] = (x + 0.35, y - 0.10)
    return points * 100.0 + np.asarray((320.0, 240.0))


def _as_landmarks(points, width=640.0, height=480.0):
    return [
        SimpleNamespace(
            x=float(point[0] / width),
            y=float(point[1] / height),
            z=0.0,
        )
        for point in points
    ]


def _as_world_landmarks(points):
    return [
        SimpleNamespace(x=float(x), y=float(y), z=float(z))
        for x, y, z in points
    ]


def _world_closed_points():
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = (0.0, 0.06, 0.0)
    points[1:5] = np.asarray([
        (-0.01, 0.02, 0.0), (-0.02, 0.00, 0.0),
        (-0.025, 0.005, 0.015), (-0.015, 0.02, 0.02),
    ])
    for (mcp, pip, dip, tip), x in zip(
        ((5, 6, 7, 8), (9, 10, 11, 12),
         (13, 14, 15, 16), (17, 18, 19, 20)),
        (-0.03, -0.01, 0.01, 0.03),
    ):
        points[mcp] = (x, 0.0, 0.0)
        points[pip] = (x, -0.03, 0.0)
        points[dip] = (x, -0.031, 0.03)
        points[tip] = (x, -0.060, 0.03)
    return points


def _world_three_compact_points():
    """Three compact PIPs plus one straight finger for right-EE evidence."""
    points = _world_closed_points()
    # Keep index/middle/ring compact and make pinky anatomically straight.
    # This avoids the generic unanimous-3D override and exercises the
    # right-EE-specific three-of-four compactness rule.
    points[17] = (0.03, 0.0, 0.0)
    points[18] = (0.03, -0.03, 0.0)
    points[19] = (0.03, -0.06, 0.0)
    points[20] = (0.03, -0.09, 0.0)
    return points


def _world_noncompact_points():
    """Four anatomically straight fingers despite a curled 2-D projection."""
    points = _world_closed_points()
    for (mcp, pip, dip, tip), x in zip(
        ((5, 6, 7, 8), (9, 10, 11, 12),
         (13, 14, 15, 16), (17, 18, 19, 20)),
        (-0.03, -0.01, 0.01, 0.03),
    ):
        points[mcp] = (x, 0.0, 0.0)
        points[pip] = (x, -0.03, 0.0)
        points[dip] = (x, -0.06, 0.0)
        points[tip] = (x, -0.09, 0.0)
    return points


def _fake_result(*, include_gesture, points=None, world_points=None):
    landmarks = _as_landmarks(
        _hand_points() if points is None else points)
    result = SimpleNamespace(
        hand_landmarks=[landmarks],
        handedness=[[_category('Left', 0.91)]],
    )
    if include_gesture:
        result.gestures = [[_category('Open_Palm', 0.87)]]
    if world_points is not None:
        result.hand_world_landmarks = [_as_world_landmarks(world_points)]
    return result


def test_padded_inference_roi_is_clipped_and_full_frame_is_elided():
    assert inference_crop_box(
        (720, 1280, 3), (0.446094, 0.825, 0.108333, 0.741667), 0.08,
    ) == (468, 20, 1159, 592)
    assert inference_crop_box(
        (720, 1280, 3), (0.0, 1.0, 0.0, 1.0), 0.08,
    ) is None


def test_crop_landmarks_are_remapped_to_original_pixel_grid():
    result = SimpleNamespace(hand_landmarks=[[
        SimpleNamespace(x=0.0, y=0.0, z=0.0),
        SimpleNamespace(x=1.0, y=1.0, z=0.0),
        SimpleNamespace(x=0.5, y=0.5, z=0.0),
    ]])
    remap_result_landmarks(result, (320, 120, 960, 600), (720, 1280, 3))
    points = [(row.x, row.y) for row in result.hand_landmarks[0]]
    np.testing.assert_allclose(points, [
        (0.25, 1.0 / 6.0),
        (0.75, 5.0 / 6.0),
        (0.5, 0.5),
    ])


def _run_fake_backend(backend):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    return process_frame(
        frame,
        depth,
        backend,
        _FakeMp,
        np.eye(3, dtype=np.float32),
        1.0,
        1.0,
        0.0,
        0.0,
        640,
        480,
        1,
        draw_overlay=False,
        allow_2d_only=True,
    )[0]


def test_gesture_result_stays_aligned_with_landmark_hand_index():
    class Recognizer:
        def recognize_for_video(self, _image, _timestamp_ms):
            return _fake_result(include_gesture=True)

    hands = _run_fake_backend(Recognizer())
    assert len(hands) == 1
    assert hands[0]['hand_index'] == 0
    assert hands[0]['gesture']['has_gesture'] is True
    assert hands[0]['gesture']['category_name'] == 'Open_Palm'
    assert hands[0]['gesture']['classifier'] == CLASSIFIER_VERSION


def test_gesture_rows_do_not_require_depth_or_camera_intrinsics():
    rows = gesture_rows_from_result(
        _fake_result(include_gesture=True),
        640,
        480,
    )
    assert len(rows) == 1
    assert rows[0]['hand_index'] == 0
    assert rows[0]['handedness'] == {'label': 'Left', 'score': 0.91}
    assert rows[0]['gesture']['category_name'] == 'Open_Palm'
    # The custom result is landmark-derived; the conflicting canned output
    # embedded by _fake_result must not leak onto the ROS contract.
    assert rows[0]['gesture']['score'] != 0.87


def test_forced_handedness_overrides_mediapipe_in_gesture_path():
    rows = gesture_rows_from_result(
        _fake_result(include_gesture=True),
        640,
        480,
        forced_handedness_label='Right',
    )
    assert rows[0]['handedness'] == {'label': 'Right', 'score': 1.0}


def test_forced_handedness_reaches_depth_and_facing_paths():
    seen_handedness = []

    def facing_estimator(_joints_3d, _valid, handedness):
        seen_handedness.append(handedness)
        return {
            'has_facing': False,
            'label': 'UNKNOWN',
            'rejection_reason': 'test_probe',
        }

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    hands = process_frame(
        frame,
        depth,
        object(),
        _FakeMp,
        np.eye(3, dtype=np.float32),
        1.0,
        1.0,
        0.0,
        0.0,
        640,
        480,
        1,
        draw_overlay=False,
        recognition_result=_fake_result(include_gesture=True),
        palm_facing_estimator=facing_estimator,
        forced_handedness_label='Right',
    )[0]

    expected = {'label': 'Right', 'score': 1.0}
    assert hands[0]['handedness'] == expected
    assert seen_handedness == [expected]


def test_precomputed_result_avoids_second_mediapipe_inference():
    class MustNotRun:
        def recognize_for_video(self, _image, _timestamp_ms):
            raise AssertionError('duplicate inference')

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    hands = process_frame(
        frame,
        depth,
        MustNotRun(),
        _FakeMp,
        np.eye(3, dtype=np.float32),
        1.0,
        1.0,
        0.0,
        0.0,
        640,
        480,
        1,
        draw_overlay=False,
        recognition_result=_fake_result(include_gesture=True),
    )[0]
    assert hands[0]['gesture']['category_name'] == 'Open_Palm'


def test_legacy_landmarker_backend_remains_supported_for_offline_cli():
    class Landmarker:
        def detect_for_video(self, _image, _timestamp_ms):
            return _fake_result(include_gesture=False)

    hands = _run_fake_backend(Landmarker())
    assert len(hands) == 1
    assert hands[0]['gesture']['category_name'] == 'Open_Palm'


def test_topview_classifier_is_rotation_scale_translation_and_mirror_invariant():
    points = _hand_points()
    wrist = points[0].copy()
    for angle_deg in range(0, 360, 45):
        radians = np.radians(angle_deg)
        rotation = np.asarray([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ], dtype=np.float32)
        transformed = (points - wrist) @ rotation.T
        transformed = transformed * 1.7 + np.asarray((83.0, -41.0))
        for candidate in (transformed, transformed * np.asarray((-1.0, 1.0))):
            result = classify(candidate)
            assert result['category_name'] == 'Open_Palm'
            assert result['extended_fingers'] == 4


def test_topview_classifier_maps_closed_and_partial_shapes_conservatively():
    closed = classify(_hand_points(open_fingers=()))
    assert closed['category_name'] == 'Closed_Fist'
    assert closed['curled_fingers'] == 4

    partial = classify(_hand_points(open_fingers=(0, 1)))
    assert partial['category_name'] == 'None'
    assert partial['extended_fingers'] == 2
    assert partial['curled_fingers'] == 2

    pointing = classify(_hand_points(open_fingers=(0,)))
    assert pointing['category_name'] == 'None'
    assert pointing['extended_fingers'] == 1
    assert pointing['curled_fingers'] == 3


def test_right_ee_profile_requires_three_compact_world_pips_for_closed_hand():
    # A right-EE fist can have one apparent image-plane extension due to
    # perspective/self-occlusion, but the 3-D PIP compactness must be present.
    one_spurious_extension = _hand_points(open_fingers=(0,))
    topview = classify(one_spurious_extension)
    right_ee_without_world = classify(
        one_spurious_extension, profile='right_ee')
    right_ee = classify(
        one_spurious_extension, _world_three_compact_points(),
        profile='right_ee')

    assert GESTURE_PROFILES == ('topview', 'right_ee')
    assert topview['category_name'] == 'None'
    assert right_ee_without_world['category_name'] == 'None'
    assert right_ee['category_name'] == 'Closed_Fist'
    assert right_ee['classifier'] == RIGHT_EE_CLASSIFIER_VERSION
    assert right_ee['decision_path'] == 'right_ee_world_pip_compact'
    assert right_ee['extended_fingers'] == 1
    assert right_ee['world_compact_pip_fingers'] == 3


def test_right_ee_profile_rejects_a_2d_half_curl_without_world_compactness():
    # User-labelled H is a half-curl: 2-D geometry sees a fist, while the
    # anatomical PIPs are too open. CAM4 keeps its original result; the EE
    # profile must fail closed rather than reuse the 2-D image rule.
    curled_image = _hand_points(open_fingers=())
    noncompact_world = _world_noncompact_points()
    topview = classify(curled_image, noncompact_world)
    right_ee = classify(curled_image, noncompact_world, profile='right_ee')

    assert topview['category_name'] == 'Closed_Fist'
    assert right_ee['category_name'] == 'None'
    assert right_ee['decision_path'] == 'ambiguous'
    assert right_ee['world_compact_pip_fingers'] == 0


def test_right_ee_profile_preserves_open_palm_precedence():
    open_hand = _hand_points()
    result = classify(open_hand, profile='right_ee')
    assert result['category_name'] == 'Open_Palm'
    assert result['decision_path'] == 'image_open'


def test_right_ee_profile_reaches_rgb_only_gesture_rows():
    image_only_result = _fake_result(
        include_gesture=True,
        points=_hand_points(open_fingers=(0,)),
    )
    assert gesture_rows_from_result(image_only_result, 640, 480)[0][
        'gesture']['category_name'] == 'None'
    assert gesture_rows_from_result(
        image_only_result, 640, 480, gesture_profile='right_ee')[0][
        'gesture']['category_name'] == 'None'

    result = _fake_result(
        include_gesture=True,
        points=_hand_points(open_fingers=(0,)),
        world_points=_world_three_compact_points(),
    )
    assert gesture_rows_from_result(
        result, 640, 480, gesture_profile='right_ee')[0][
        'gesture']['category_name'] == 'Closed_Fist'


def test_world_landmarks_recover_z_axis_closed_fist_projection():
    # This is one physically consistent 3-D skeleton and its orthographic XY
    # projection. Flexion is mostly along Z, so the old image-only path sees
    # straight fingers while the anatomical PIP/DIP angles remain about 92°.
    world = _world_closed_points()
    image_projection = (
        world[:, :2] * 1000.0 + np.asarray((320.0, 240.0)))
    assert classify(image_projection)['category_name'] == 'Open_Palm'
    result = classify(image_projection, world)
    assert result['category_name'] == 'Closed_Fist'
    assert result['world_geometry_valid'] is True
    assert result['world_curled_fingers'] == 4
    assert result['world_strongly_curled_fingers'] == 4
    assert result['decision_path'] == 'world_closed_strong_unanimous'


def test_three_world_curls_conflicting_with_image_open_fail_closed():
    world = _world_closed_points()
    # Keep one anatomically straight finger so the 3-D cue is only 3/4
    # consensus, not the strict unanimous override used for the known
    # back-of-hand fist failure mode.
    world[17] = (0.03, 0.0, 0.0)
    world[18] = (0.03, -0.03, 0.0)
    world[19] = (0.03, -0.06, 0.0)
    world[20] = (0.03, -0.09, 0.0)

    result = classify(_hand_points(), world)
    assert result['category_name'] == 'None'
    assert result['world_curled_fingers'] == 3
    assert result['decision_path'] == 'world_image_conflict'


def test_invalid_world_landmarks_fall_back_to_existing_image_geometry():
    world = _world_closed_points()
    world[8, 2] = np.nan
    result = classify(_hand_points(open_fingers=()), world)
    assert result['category_name'] == 'Closed_Fist'
    assert result['world_geometry_valid'] is False
    assert result['decision_path'] == 'image_closed'


def test_two_hand_world_landmarks_remain_index_aligned():
    open_image = _hand_points()
    ambiguous_image = _hand_points(open_fingers=(0, 1))
    open_world = np.column_stack((
        open_image[:, 0] * 0.001,
        open_image[:, 1] * 0.001,
        np.zeros(21, dtype=np.float32),
    ))
    result = SimpleNamespace(
        hand_landmarks=[
            _as_landmarks(open_image),
            _as_landmarks(ambiguous_image),
        ],
        hand_world_landmarks=[
            _as_world_landmarks(open_world),
            _as_world_landmarks(_world_closed_points()),
        ],
        handedness=[
            [_category('Left', 0.91)],
            [_category('Right', 0.92)],
        ],
    )
    rows = gesture_rows_from_result(result, 640, 480)
    assert [row['hand_index'] for row in rows] == [0, 1]
    assert rows[0]['gesture']['category_name'] == 'Open_Palm'
    assert rows[1]['gesture']['category_name'] == 'Closed_Fist'


def test_depth_path_uses_world_geometry_and_isolates_facing_failure():
    world = _world_closed_points()
    image_projection = (
        world[:, :2] * 1000.0 + np.asarray((320.0, 240.0)))
    result = _fake_result(
        include_gesture=True,
        points=image_projection,
        world_points=world,
    )

    def broken_facing_estimator(_joints, _valid, _handedness):
        raise RuntimeError('synthetic facing failure')

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32)
    hands = process_frame(
        frame,
        depth,
        object(),
        _FakeMp,
        np.eye(3, dtype=np.float32),
        1.0,
        1.0,
        0.0,
        0.0,
        640,
        480,
        1,
        draw_overlay=False,
        recognition_result=result,
        palm_facing_estimator=broken_facing_estimator,
    )[0]

    assert len(hands) == 1
    assert hands[0]['gesture']['category_name'] == 'Closed_Fist'
    assert hands[0]['palm_facing']['label'] == 'UNKNOWN'
    assert hands[0]['palm_facing']['rejection_reason'] == (
        'estimator_exception:RuntimeError')


def test_topview_classifier_rejects_invalid_geometry_as_explicit_none():
    invalid = classify(np.zeros((21, 2), dtype=np.float32))
    assert invalid['has_gesture'] is True
    assert invalid['category_name'] == 'None'
    assert invalid['quality_valid'] is False

    nan_points = _hand_points()
    nan_points[8, 0] = np.nan
    invalid = classify(nan_points)
    assert invalid['category_name'] == 'None'
    assert invalid['quality_valid'] is False
