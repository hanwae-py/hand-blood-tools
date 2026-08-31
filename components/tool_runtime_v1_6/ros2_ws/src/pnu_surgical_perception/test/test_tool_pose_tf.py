"""Tests for constrained planar-tool TF construction and temporal tracking."""

from __future__ import annotations

import math

import pytest

from std_msgs.msg import Header

from surgical_perception_msgs.msg import ToolPose

from pnu_surgical_perception.tool_pose_tf import (
    spatial_tool_child_frames,
    ToolSpatialTfSelector,
    ToolTfTracker,
    constrained_tool_child_frame,
    constrained_transform_from_tool_pose,
    source_age_seconds,
)


def _header(seconds: int = 100, nanoseconds: int = 250_000_000) -> Header:
    header = Header()
    header.stamp.sec = seconds
    header.stamp.nanosec = nanoseconds
    header.frame_id = 'cam_4_color_optical_frame'
    return header


def _valid_planar_tool(
    *,
    local_id: int = 7,
    position: tuple[float, float, float] = (0.1, -0.2, 0.8),
    confidence: float = 0.9,
    horizontal_u_px: float = 100.0,
    class_name: str = 'Army-Navy Retractor',
    canonical_class_id: int = 6,
) -> ToolPose:
    tool = ToolPose()
    tool.frame_local_instance_id = local_id
    tool.canonical_class_id = canonical_class_id
    tool.class_name = class_name
    tool.class_confidence = confidence
    tool.pose_mode = ToolPose.POSE_MODE_PLANAR_4DOF_WITH_NORMAL_PRIOR
    tool.validity = ToolPose.VALIDITY_VALID
    tool.position_valid = True
    tool.orientation_valid = True
    tool.dof_observed = [True, True, True, False, False, True]
    tool.pose.position.x, tool.pose.position.y, tool.pose.position.z = position
    tool.observation_point_uv_px = [horizontal_u_px, 240.0]
    # Deliberately non-unit: TF output must normalize it before publication.
    tool.pose.orientation.x = 0.0
    tool.pose.orientation.y = 0.0
    tool.pose.orientation.z = 2.0
    tool.pose.orientation.w = 2.0
    return tool


def _tracker() -> ToolTfTracker:
    return ToolTfTracker(
        max_displacement_m=0.05,
        ttl_sec=1.0,
        max_tracks_per_class=2,
        reset_stamp_jump_sec=5.0,
    )


def _spatial_selector() -> ToolSpatialTfSelector:
    return ToolSpatialTfSelector(
        max_tools_per_class=8,
        reset_stamp_jump_sec=5.0,
    )


def test_constrained_planar_pose_becomes_timestamped_unit_se3_transform():
    tool = _valid_planar_tool()
    decision = constrained_transform_from_tool_pose(
        _header(), tool, 'cam_4', track_id=4
    )

    assert decision.reason == 'PUBLISHED'
    assert decision.transform is not None
    transform = decision.transform
    assert transform.header.frame_id == 'cam_4_color_optical_frame'
    assert transform.header.stamp.sec == 100
    assert transform.header.stamp.nanosec == 250_000_000
    assert transform.child_frame_id == (
        'cam_4_army#4'
    )
    assert 'frame_local' not in transform.child_frame_id
    assert transform.transform.translation.x == pytest.approx(0.1)
    assert transform.transform.translation.y == pytest.approx(-0.2)
    assert transform.transform.translation.z == pytest.approx(0.8)
    orientation = transform.transform.rotation
    norm = math.sqrt(
        orientation.x**2
        + orientation.y**2
        + orientation.z**2
        + orientation.w**2
    )
    assert norm == pytest.approx(1.0)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('validity', ToolPose.VALIDITY_DEGRADED),
        ('validity', ToolPose.VALIDITY_INVALID),
        ('position_valid', False),
        ('orientation_valid', False),
        ('dof_observed', [True, True, True, False, False, False]),
    ],
)
def test_quality_flags_do_not_suppress_a_numeric_measured_tf(
    field: str, value: object
):
    tool = _valid_planar_tool()
    setattr(tool, field, value)

    decision = constrained_transform_from_tool_pose(
        _header(), tool, 'cam_4', track_id=1
    )

    assert decision.transform is not None
    assert decision.reason == 'PUBLISHED'


def test_tf_accepts_numeric_pose_when_reported_dof_quality_differs():
    tool = _valid_planar_tool()
    tool.dof_observed = [True, True, True, True, True, True]

    decision = constrained_transform_from_tool_pose(
        _header(), tool, 'cam_4', track_id=1
    )

    assert decision.transform is not None
    assert decision.reason == 'PUBLISHED'


def test_tf_still_rejects_a_pose_without_a_numeric_orientation():
    tool = _valid_planar_tool()
    tool.validity = ToolPose.VALIDITY_INVALID
    tool.position_valid = False
    tool.orientation_valid = False
    tool.dof_observed = [False] * 6
    tool.pose.orientation.x = 0.0
    tool.pose.orientation.y = 0.0
    tool.pose.orientation.z = 0.0
    tool.pose.orientation.w = 0.0

    decision = constrained_transform_from_tool_pose(
        _header(), tool, 'cam_4', track_id=1
    )

    assert decision.transform is None
    assert decision.reason == 'ORIENTATION_ZERO_NORM'


def test_track_name_is_camera_qualified_and_never_contains_frame_local_id():
    tool = _valid_planar_tool(local_id=99)

    assert constrained_tool_child_frame('cam_3', tool, 5) == (
        'cam_3_army#5'
    )


def test_same_physical_tool_keeps_track_when_frame_local_id_changes():
    tracker = _tracker()
    first = tracker.assign(
        _header(100, 0),
        [_valid_planar_tool(local_id=0, position=(0.10, 0.0, 0.8))],
        'cam_4',
    )[0]
    second = tracker.assign(
        _header(100, 100_000_000),
        [_valid_planar_tool(local_id=19, position=(0.12, 0.0, 0.8))],
        'cam_4',
    )[0]

    assert first.reason == 'PUBLISHED'
    assert second.reason == 'PUBLISHED'
    assert first.child_frame_id == second.child_frame_id
    assert first.child_frame_id.endswith('#1')
    assert tracker.active_track_count == 1
    assert tracker.created_total == 1


def test_expired_reappearance_gets_new_track_id_without_accumulating_active_tracks():
    tracker = _tracker()
    first = tracker.assign(
        _header(100, 0), [_valid_planar_tool()], 'cam_4'
    )[0]
    second = tracker.assign(
        _header(102, 0), [_valid_planar_tool(local_id=1)], 'cam_4'
    )[0]

    assert first.reason == 'PUBLISHED'
    assert second.reason == 'PUBLISHED'
    assert first.child_frame_id.endswith('#1')
    assert second.child_frame_id.endswith('#2')
    assert tracker.active_track_count == 1
    assert tracker.expired_total == 1
    assert tracker.created_total == 2


def test_repeated_frame_local_churn_does_not_allocate_unbounded_tracks():
    tracker = _tracker()
    child_frames = set()
    for frame in range(30):
        decision = tracker.assign(
            _header(100, frame * 10_000_000),
            [
                _valid_planar_tool(
                    local_id=frame,
                    position=(0.10 + 0.001 * frame, 0.0, 0.8),
                )
            ],
            'cam_4',
        )[0]
        assert decision.reason == 'PUBLISHED'
        child_frames.add(decision.child_frame_id)

    assert child_frames == {
        'cam_4_army#1'
    }
    assert tracker.active_track_count == 1
    assert tracker.created_total == 1


def test_large_displacement_is_rejected_until_old_track_expires():
    tracker = _tracker()
    initial = tracker.assign(
        _header(100, 0), [_valid_planar_tool(position=(0.0, 0.0, 0.8))], 'cam_4'
    )[0]
    jumped = tracker.assign(
        _header(100, 100_000_000),
        [_valid_planar_tool(local_id=1, position=(0.20, 0.0, 0.8))],
        'cam_4',
    )[0]
    reappeared = tracker.assign(
        _header(102, 0),
        [_valid_planar_tool(local_id=2, position=(0.20, 0.0, 0.8))],
        'cam_4',
    )[0]

    assert initial.reason == 'PUBLISHED'
    assert jumped.transform is None
    assert jumped.reason == 'TRACK_DISPLACEMENT_EXCEEDED'
    assert reappeared.reason == 'PUBLISHED'
    assert reappeared.child_frame_id.endswith('#2')


def test_small_out_of_order_stamp_is_rejected_and_large_clock_reset_is_explicit():
    tracker = _tracker()
    first = tracker.assign(
        _header(100, 0), [_valid_planar_tool()], 'cam_4'
    )[0]
    out_of_order = tracker.assign(
        _header(99, 900_000_000), [_valid_planar_tool(local_id=1)], 'cam_4'
    )[0]
    reset = tracker.assign(
        _header(90, 0), [_valid_planar_tool(local_id=2)], 'cam_4'
    )[0]

    assert first.reason == 'PUBLISHED'
    assert out_of_order.transform is None
    assert out_of_order.reason == 'TRACK_SOURCE_STAMP_OUT_OF_ORDER'
    assert reset.reason == 'PUBLISHED'
    assert reset.child_frame_id.endswith('#2')
    assert tracker.reset_total == 1


def test_source_age_uses_the_header_timebase():
    assert source_age_seconds(_header(), 101_750_000_000) == pytest.approx(1.5)
    assert source_age_seconds(_header(), 99_750_000_000) == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ('class_name', 'canonical_id', 'expected'),
    [
        ('Bovie', 6, 'cam_4_bovie#2'),
        ('Bipolar Forceps', 5, 'cam_4_bipolar#2'),
    ],
)
def test_child_frame_starts_with_human_readable_tool_name(
    class_name: str, canonical_id: int, expected: str
):
    tool = _valid_planar_tool()
    tool.class_name = class_name
    tool.canonical_class_id = canonical_id

    assert constrained_tool_child_frame('cam_4', tool, 2) == expected


def test_spatial_names_are_per_class_and_left_to_right_in_input_snapshot():
    tools = [
        _valid_planar_tool(local_id=7, horizontal_u_px=620.0),
        _valid_planar_tool(local_id=2, horizontal_u_px=110.0),
        _valid_planar_tool(
            local_id=9,
            horizontal_u_px=50.0,
            class_name='Mosquito',
            canonical_class_id=2,
        ),
    ]

    assert spatial_tool_child_frames('mayo', tools) == [
        'mayo_army#2',
        'mayo_army#1',
        'mayo_mosquito#1',
    ]


def test_spatial_selector_publishes_zone_names_not_physical_track_ids():
    selector = _spatial_selector()
    decisions = selector.assign(
        _header(100, 0),
        [
            _valid_planar_tool(local_id=1, horizontal_u_px=700.0),
            _valid_planar_tool(local_id=2, horizontal_u_px=100.0),
        ],
        'tray',
    )

    assert [decision.child_frame_id for decision in decisions] == [
        'tray_army#2',
        'tray_army#1',
    ]
    assert all(decision.reason == 'PUBLISHED' for decision in decisions)
    assert selector.active_track_count == 2


def test_spatial_selector_reindexes_after_left_item_disappears_by_design():
    selector = _spatial_selector()
    first = selector.assign(
        _header(100, 0),
        [
            _valid_planar_tool(local_id=1, horizontal_u_px=100.0),
            _valid_planar_tool(local_id=2, horizontal_u_px=700.0),
        ],
        'mayo',
    )
    second = selector.assign(
        _header(100, 100_000_000),
        [_valid_planar_tool(local_id=99, horizontal_u_px=700.0)],
        'mayo',
    )

    assert [decision.child_frame_id for decision in first] == [
        'mayo_army#1',
        'mayo_army#2',
    ]
    assert second[0].child_frame_id == 'mayo_army#1'
    assert selector.expired_total == 1


def test_degraded_left_selector_reserves_ordinal_and_emits_measured_tf():
    selector = _spatial_selector()
    left = _valid_planar_tool(local_id=1, horizontal_u_px=100.0)
    left.validity = ToolPose.VALIDITY_DEGRADED
    right = _valid_planar_tool(local_id=2, horizontal_u_px=700.0)

    decisions = selector.assign(_header(100, 0), [left, right], 'tray')

    assert decisions[0].child_frame_id == 'tray_army#1'
    assert decisions[0].transform is not None
    assert decisions[0].reason == 'PUBLISHED'
    assert decisions[1].child_frame_id == 'tray_army#2'
    assert decisions[1].transform is not None


def test_observation_u_override_keeps_bbox_and_pose_selector_order_identical():
    selector = _spatial_selector()
    tools = [
        _valid_planar_tool(local_id=10, horizontal_u_px=100.0),
        _valid_planar_tool(local_id=20, horizontal_u_px=700.0),
    ]

    decisions = selector.assign(
        _header(100, 0),
        tools,
        'mayo',
        horizontal_u_by_instance_id={10: 800.0, 20: 20.0},
    )

    assert [decision.child_frame_id for decision in decisions] == [
        'mayo_army#2',
        'mayo_army#1',
    ]


def test_spatial_selector_outputs_stabilized_translation_when_enabled():
    selector = ToolSpatialTfSelector(
        max_tools_per_class=8,
        reset_stamp_jump_sec=5.0,
        position_stabilization_enabled=True,
        position_deadband_m=0.008,
        position_smoothing_alpha=0.20,
        position_max_jump_m=0.04,
        position_relocation_confirmation_frames=2,
        position_relocation_consistency_m=0.015,
        position_max_missed_frames=3,
    )
    first = selector.assign(
        _header(100, 0),
        [_valid_planar_tool(position=(0.10, 0.20, 0.80))],
        'mayo',
    )[0]
    jittered = selector.assign(
        _header(100, 100_000_000),
        [_valid_planar_tool(position=(0.105, 0.20, 0.80))],
        'mayo',
    )[0]

    assert first.transform is not None
    assert jittered.transform is not None
    assert jittered.transform.transform.translation.x == pytest.approx(0.10)
    assert selector.position_filter_held_total == 1


def test_position_filter_resets_when_spatial_selector_cardinality_changes():
    selector = ToolSpatialTfSelector(
        max_tools_per_class=8,
        reset_stamp_jump_sec=5.0,
        position_stabilization_enabled=True,
    )
    selector.assign(
        _header(100, 0),
        [
            _valid_planar_tool(
                local_id=1,
                position=(0.10, 0.20, 0.80),
                horizontal_u_px=100.0,
            ),
            _valid_planar_tool(
                local_id=2,
                position=(0.30, 0.20, 0.80),
                horizontal_u_px=700.0,
            ),
        ],
        'mayo',
    )
    remaining = selector.assign(
        _header(100, 100_000_000),
        [
            _valid_planar_tool(
                local_id=9,
                position=(0.30, 0.20, 0.80),
                horizontal_u_px=700.0,
            )
        ],
        'mayo',
    )[0]

    assert remaining.transform is not None
    assert remaining.child_frame_id == 'mayo_army#1'
    assert remaining.transform.transform.translation.x == pytest.approx(0.30)
    assert selector.position_filter_association_reset_total == 2
