"""Pure contract tests for ROS class masks and color-aligned depth."""

from types import SimpleNamespace
import threading

import numpy as np
from pnu_surgical_perception.native_depth_pose_node import (
    LatestOnlyOutputSlot,
    aligned_depth_to_meters,
    class_mask_messages,
    class_mask_slug,
    union_class_masks,
)
import pytest
from std_msgs.msg import Header


def test_class_mask_slug_matches_public_topic_contract():
    assert class_mask_slug('Adson Forceps') == 'adson_forceps'
    assert class_mask_slug('Army-Navy Retractor') == 'army_navy_retractor'
    with pytest.raises(ValueError, match='ASCII'):
        class_mask_slug('---')


def test_class_masks_union_instances_and_publish_empty_classes():
    first = np.zeros((4, 6), dtype=bool)
    second = np.zeros((4, 6), dtype=bool)
    first[1, 2] = True
    second[2, 4] = True
    detections = SimpleNamespace(instances=[
        SimpleNamespace(class_name='Scalpel', mask=first),
        SimpleNamespace(class_name='Scalpel', mask=second),
    ])

    masks = union_class_masks(
        detections, ('Scalpel', 'Bovie'), (4, 6)
    )

    assert masks['Scalpel'].dtype == np.uint8
    assert masks['Scalpel'][1, 2] == 255
    assert masks['Scalpel'][2, 4] == 255
    assert np.count_nonzero(masks['Scalpel']) == 2
    assert np.count_nonzero(masks['Bovie']) == 0


def test_class_mask_bundle_has_one_exact_source_stamp_and_frame():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 2] = True
    detections = SimpleNamespace(instances=[
        SimpleNamespace(class_name='Scalpel', mask=mask)
    ])
    header = Header()
    header.stamp.sec = 123
    header.stamp.nanosec = 456789
    header.frame_id = 'cam_3_color_optical_frame'

    messages = class_mask_messages(
        detections,
        tuple(f'class_{index}' for index in range(7)) + ('Scalpel',),
        (4, 6),
        header,
    )

    assert len(messages) == 8
    assert {message.header.stamp.sec for _, message in messages} == {123}
    assert {message.header.stamp.nanosec for _, message in messages} == {456789}
    assert {message.header.frame_id for _, message in messages} == {
        'cam_3_color_optical_frame'
    }


def test_latest_output_slot_overwrites_only_not_yet_taken_bundle():
    slot = LatestOnlyOutputSlot()
    first_taken = threading.Event()
    release_first = threading.Event()
    second_taken = threading.Event()
    processed = []

    def consume():
        processed.append(slot.take())
        first_taken.set()
        assert release_first.wait(timeout=1.0)
        processed.append(slot.take())
        second_taken.set()

    slot.put('A')
    worker = threading.Thread(target=consume)
    worker.start()
    assert first_taken.wait(timeout=1.0)
    slot.put('B')
    slot.put('C')
    release_first.set()
    assert second_taken.wait(timeout=1.0)
    slot.stop()
    worker.join(timeout=1.0)

    assert processed == ['A', 'C']
    assert slot.overwritten_total == 1
    assert not worker.is_alive()


def test_class_mask_rejects_geometry_mismatch():
    detections = SimpleNamespace(instances=[SimpleNamespace(
        class_name='Scalpel', mask=np.zeros((2, 3), dtype=bool)
    )])
    with pytest.raises(ValueError, match='RGB shape'):
        union_class_masks(detections, ('Scalpel',), (3, 2))


def test_aligned_depth_conversion_filters_invalid_samples():
    native = np.array([[0, 49, 50], [750, 10001, 1000]], dtype=np.uint16)

    depth_m = aligned_depth_to_meters(
        native,
        (2, 3),
        depth_scale_m_per_unit=0.001,
        minimum_depth_m=0.05,
        maximum_depth_m=10.0,
    )

    assert np.isnan(depth_m[0, 0])
    assert np.isnan(depth_m[0, 1])
    assert depth_m[0, 2] == pytest.approx(0.05)
    assert depth_m[1, 0] == pytest.approx(0.75)
    assert np.isnan(depth_m[1, 1])
    assert depth_m[1, 2] == pytest.approx(1.0)


def test_aligned_depth_rejects_nonmatching_rgb_shape():
    with pytest.raises(ValueError, match='RGB shape'):
        aligned_depth_to_meters(
            np.zeros((2, 3), dtype=np.uint16),
            (3, 2),
            0.001,
            0.05,
            10.0,
        )
