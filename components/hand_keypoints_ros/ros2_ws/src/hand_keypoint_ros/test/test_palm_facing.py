import numpy as np

from hand_keypoint_ros.palm_facing import (
    PalmFacingEstimator,
    PalmFacingTemporalFilter,
)


def _estimator():
    return PalmFacingEstimator(
        table_up_normal=[0.0, 0.0, -1.0],
        support_plane_offset_m=0.84,
        handedness_signs={'Left': -1.0, 'Right': 1.0},
        enter_cosine=0.75,
        max_plane_residual_m=0.012,
        min_palm_span_m=0.025,
        min_handedness_score=0.60,
        calibration_version='synthetic-table-v1',
        handedness_mapping_version='synthetic-nonmirrored-v1',
    )


def _metric_hand(*, index_on_left=True):
    points = np.zeros((21, 3), dtype=np.float32)
    if index_on_left:
        palm = {
            0: (0.0, 0.08, 0.80),
            5: (-0.04, 0.00, 0.80),
            9: (-0.012, -0.018, 0.80),
            13: (0.016, -0.014, 0.80),
            17: (0.045, 0.005, 0.80),
        }
    else:
        palm = {
            0: (0.0, 0.08, 0.80),
            5: (0.04, 0.00, 0.80),
            9: (0.012, -0.018, 0.80),
            13: (-0.016, -0.014, 0.80),
            17: (-0.045, 0.005, 0.80),
        }
    for index, point in palm.items():
        points[index] = point
    valid = np.zeros(21, dtype=bool)
    valid[list(palm)] = True
    return points, valid


def test_left_and_right_handedness_produce_same_semantic_palm_up():
    estimator = _estimator()
    left_points, left_valid = _metric_hand(index_on_left=True)
    right_points, right_valid = _metric_hand(index_on_left=False)

    left = estimator.estimate(
        left_points, left_valid, {'label': 'Left', 'score': 0.95})
    right = estimator.estimate(
        right_points, right_valid, {'label': 'Right', 'score': 0.95})

    assert left['label'] == 'PALM_UP'
    assert right['label'] == 'PALM_UP'
    assert left['palm_up_score'] > 0.99
    assert right['palm_up_score'] > 0.99


def test_topology_flip_produces_palm_down():
    estimator = _estimator()
    points, valid = _metric_hand(index_on_left=False)
    result = estimator.estimate(
        points, valid, {'label': 'Left', 'score': 0.95})
    assert result['label'] == 'PALM_DOWN'
    assert result['palm_up_score'] < -0.99


def test_edge_and_invalid_depth_fail_closed():
    estimator = _estimator()
    points, valid = _metric_hand(index_on_left=True)
    centre = points[0].copy()
    radians = np.pi / 2.0
    rotation = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(radians), -np.sin(radians)],
        [0.0, np.sin(radians), np.cos(radians)],
    ])
    transformed = points.copy()
    transformed[valid] = (points[valid] - centre) @ rotation.T + centre
    edge = estimator.estimate(
        transformed, valid, {'label': 'Left', 'score': 0.95})
    assert edge['has_facing'] is True
    assert edge['label'] == 'EDGE'

    valid[5] = False
    unknown = estimator.estimate(
        transformed, valid, {'label': 'Left', 'score': 0.95})
    assert unknown['has_facing'] is False
    assert unknown['label'] == 'UNKNOWN'
    assert unknown['rejection_reason'] == 'insufficient_palm_depth'


def test_large_palm_plane_residual_is_unknown():
    estimator = _estimator()
    points, valid = _metric_hand(index_on_left=True)
    points[9, 2] += 0.08
    result = estimator.estimate(
        points, valid, {'label': 'Left', 'score': 0.95})
    assert result['has_facing'] is False
    assert result['label'] == 'UNKNOWN'
    assert result['rejection_reason'] == 'palm_plane_residual_too_large'
    assert result['plane_residual_m'] > 0.012


def test_table_or_background_depth_is_rejected_by_support_height():
    estimator = _estimator()
    points, valid = _metric_hand(index_on_left=True)
    points[valid, 2] = 0.839
    result = estimator.estimate(
        points, valid, {'label': 'Left', 'score': 0.95})

    assert result['has_facing'] is False
    assert result['label'] == 'UNKNOWN'
    assert result['rejection_reason'] == 'palm_support_height_out_of_range'
    assert result['support_height_m'] < 0.008


def test_temporal_filter_holds_then_releases_facing_label():
    temporal = PalmFacingTemporalFilter(
        enter_cosine=0.75, hold_cosine=0.60, alpha=0.50)
    base = {
        'has_facing': True,
        'label': 'PALM_UP',
        'palm_up_score': 0.90,
        'normal_cam': [np.sqrt(1.0 - 0.90 ** 2), 0.0, -0.90],
    }
    first = temporal.update(
        base, centroid_norm=(0.4, 0.5), handedness={'label': 'Left'})
    assert first['label'] == 'PALM_UP'

    weaker = dict(
        base,
        palm_up_score=0.50,
        normal_cam=[np.sqrt(1.0 - 0.50 ** 2), 0.0, -0.50],
    )
    held = temporal.update(
        weaker, centroid_norm=(0.41, 0.5), handedness={'label': 'Left'})
    assert held['label'] == 'PALM_UP'
    assert 0.60 <= held['palm_up_score'] < 0.75
    assert np.isclose(
        held['palm_up_score'],
        np.dot(held['normal_cam'], [0.0, 0.0, -1.0]),
    )

    edge = dict(
        base,
        palm_up_score=-0.20,
        normal_cam=[np.sqrt(1.0 - 0.20 ** 2), 0.0, 0.20],
    )
    released = temporal.update(
        edge, centroid_norm=(0.42, 0.5), handedness={'label': 'Left'})
    assert released['label'] == 'EDGE'
    assert np.isclose(
        released['palm_up_score'],
        np.dot(released['normal_cam'], [0.0, 0.0, -1.0]),
    )
