"""ROS-independent tests for control-facing tool position stabilization."""

from __future__ import annotations

import pytest

from pnu_surgical_perception.tool_position_stabilizer import (
    ToolPositionStabilizer,
)


def _stabilizer(**overrides: object) -> ToolPositionStabilizer:
    options = {
        'enabled': True,
        'deadband_m': 0.008,
        'smoothing_alpha': 0.20,
        'max_jump_m': 0.04,
        'relocation_confirmation_frames': 2,
        'relocation_consistency_m': 0.015,
        'max_missed_frames': 3,
    }
    options.update(overrides)
    return ToolPositionStabilizer(**options)


def test_holds_single_pixel_scale_jitter():
    stabilizer = _stabilizer()
    first = stabilizer.update('mayo_army#1', (0.10, 0.20, 0.80))
    jittered = stabilizer.update(
        'mayo_army#1', (0.105, 0.20, 0.80)
    )

    assert first.reason == 'INITIALIZED'
    assert jittered.reason == 'DEADBAND_HELD'
    assert jittered.position_m == pytest.approx(first.position_m)
    assert stabilizer.held_total == 1


def test_smooths_moderate_motion_without_frame_delay():
    stabilizer = _stabilizer()
    stabilizer.update('tray_bovie#1', (0.10, 0.20, 0.80))
    smoothed = stabilizer.update('tray_bovie#1', (0.12, 0.20, 0.80))

    assert smoothed.reason == 'EMA_SMOOTHED'
    assert smoothed.position_m == pytest.approx((0.104, 0.20, 0.80))
    assert stabilizer.smoothed_total == 1


def test_holds_one_jump_then_accepts_consistent_relocation():
    stabilizer = _stabilizer()
    stabilizer.update('mayo_army#1', (0.10, 0.20, 0.80))
    jumped = stabilizer.update('mayo_army#1', (0.20, 0.20, 0.80))
    confirmed = stabilizer.update(
        'mayo_army#1', (0.205, 0.20, 0.80)
    )

    assert jumped.reason == 'OUTLIER_HELD'
    assert jumped.position_m == pytest.approx((0.10, 0.20, 0.80))
    assert confirmed.reason == 'RELOCATION_CONFIRMED'
    assert confirmed.position_m == pytest.approx((0.2025, 0.20, 0.80))
    assert stabilizer.outlier_held_total == 1
    assert stabilizer.relocation_total == 1


def test_expires_missing_slot_before_reappearance():
    stabilizer = _stabilizer(max_missed_frames=1)
    stabilizer.update('mayo_army#1', (0.10, 0.20, 0.80))
    stabilizer.finish_frame(set())
    stabilizer.finish_frame(set())
    reappeared = stabilizer.update(
        'mayo_army#1', (0.30, 0.20, 0.80)
    )

    assert reappeared.reason == 'INITIALIZED'
    assert reappeared.position_m == pytest.approx((0.30, 0.20, 0.80))


def test_explicit_association_reset_drops_reindexed_slot_state():
    stabilizer = _stabilizer()
    stabilizer.update('mayo_army#1', (0.10, 0.20, 0.80))
    stabilizer.reset_keys({'mayo_army#1'})
    reassigned = stabilizer.update(
        'mayo_army#1', (0.30, 0.20, 0.80)
    )

    assert reassigned.reason == 'INITIALIZED'
    assert reassigned.position_m == pytest.approx((0.30, 0.20, 0.80))
    assert stabilizer.association_reset_total == 1


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'deadband_m': -0.1}, 'deadband_m'),
        ({'smoothing_alpha': 0.0}, 'smoothing_alpha'),
        ({'max_jump_m': 0.005}, 'max_jump_m'),
        ({'relocation_confirmation_frames': 0}, 'confirmation_frames'),
        ({'relocation_consistency_m': -0.1}, 'consistency_m'),
        ({'max_missed_frames': -1}, 'max_missed_frames'),
    ],
)
def test_rejects_invalid_configuration(
    override: dict[str, object], message: str
):
    with pytest.raises(ValueError, match=message):
        _stabilizer(**override)
