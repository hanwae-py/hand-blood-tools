from types import SimpleNamespace

from hand_keypoint_ros.hand_detection_node import (
    HandDetectionNode,
    _row_facing_to_msg,
    camera_info_qos,
    camera_infos_share_pixel_grid,
    image_reader_qos,
)
from rclpy.qos import ReliabilityPolicy


def _camera_info(*, frame_id, fx=100.0):
    return SimpleNamespace(
        width=5,
        height=4,
        distortion_model='plumb_bob',
        k=[fx, 0.0, 2.0, 0.0, fx, 1.5, 0.0, 0.0, 1.0],
        d=[0.0, 0.0, 0.0, 0.0, 0.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[
            fx, 0.0, 2.0, 0.0,
            0.0, fx, 1.5, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        header=SimpleNamespace(frame_id=frame_id),
    )


def test_equal_resolution_does_not_prove_equal_pixel_grid():
    color = _camera_info(frame_id='color_optical_frame', fx=100.0)
    depth = _camera_info(frame_id='depth_optical_frame', fx=110.0)

    assert not camera_infos_share_pixel_grid(color, depth, (4, 5), 4, 5)


def test_live_image_and_camera_info_qos_keep_only_the_latest_sample():
    image_qos = image_reader_qos()
    info_qos = camera_info_qos()

    assert image_qos.depth == 1
    assert info_qos.depth == 1
    assert image_qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert info_qos.reliability == ReliabilityPolicy.RELIABLE


def test_matching_frame_and_calibration_prove_aligned_color_grid():
    color = _camera_info(frame_id='color_optical_frame')
    aligned_depth = _camera_info(frame_id='color_optical_frame')

    assert camera_infos_share_pixel_grid(
        color, aligned_depth, (4, 5), 4, 5,
    )


def test_repeated_depth_camera_info_keeps_cached_registrar():
    class Registrar:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class Logger:
        def warn(self, _message):
            raise AssertionError('registrar close should not fail')

    class Subject:
        _depth_camera_info_key = staticmethod(
            HandDetectionNode._depth_camera_info_key)
        _discard_depth_registrar = HandDetectionNode._discard_depth_registrar
        _on_depth_info = HandDetectionNode._on_depth_info

        def __init__(self):
            self._depth_info = None
            self._depth_info_key = None
            self._registrar = None
            self._registrar_key = None

        def get_logger(self):
            return Logger()

    subject = Subject()
    first = _camera_info(frame_id='depth_optical_frame', fx=100.0)
    subject._on_depth_info(first)
    cached = Registrar()
    subject._registrar = cached
    subject._registrar_key = ('cached',)

    subject._on_depth_info(
        _camera_info(frame_id='depth_optical_frame', fx=100.0))
    assert subject._registrar is cached
    assert cached.close_calls == 0

    subject._on_depth_info(
        _camera_info(frame_id='depth_optical_frame', fx=101.0))
    assert subject._registrar is None
    assert cached.close_calls == 1


def test_depth_facing_row_serializes_to_generated_ros_message():
    message = _row_facing_to_msg({
        'hand_index': 2,
        'handedness': {'label': 'Left', 'score': 0.93},
        'palm_facing': {
            'has_facing': True,
            'label': 'PALM_DOWN',
            'palm_up_score': -0.88,
            'normal_cam': [0.1, -0.2, 0.97],
            'plane_residual_m': 0.004,
            'support_height_m': 0.06,
            'valid_depth_points': 5,
            'rejection_reason': '',
        },
    })

    assert message.hand_index == 2
    assert message.has_handedness
    assert message.handedness_label == 'Left'
    assert message.has_facing
    assert message.facing_label == 'PALM_DOWN'
    assert message.palm_up_score < -0.87
    assert message.palm_normal_cam.z > 0.96
    assert message.valid_depth_points == 5
    assert message.rejection_reason == ''


def test_malformed_facing_normal_serializes_as_fail_closed_unknown():
    message = _row_facing_to_msg({
        'hand_index': 0,
        'handedness': None,
        'palm_facing': {
            'has_facing': True,
            'label': 'PALM_UP',
            'palm_up_score': float('nan'),
            'normal_cam': [0.0],
            'plane_residual_m': float('nan'),
            'support_height_m': 0.05,
            'valid_depth_points': 4,
            'rejection_reason': '',
        },
    })

    assert not message.has_facing
    assert message.facing_label == 'UNKNOWN'
    assert message.palm_up_score == 0.0
    assert message.palm_normal_cam.x == 0.0
    assert message.palm_normal_cam.y == 0.0
    assert message.palm_normal_cam.z == 0.0
    assert message.rejection_reason == 'serialization_input_invalid'
