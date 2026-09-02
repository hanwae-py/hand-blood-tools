"""Tests for class-independent surgical-tool axis stabilization."""

from __future__ import annotations

import pytest

from pnu_surgical_perception.tool_axis_stabilizer import ToolAxisStabilizer


IDENTITY = (0.0, 0.0, 0.0, 1.0)
PLANAR_FLIP = (0.0, 0.0, 1.0, 0.0)


def _stabilizer(**overrides: object) -> ToolAxisStabilizer:
    parameters = {
        'enabled': True,
        'flip_confirmation_frames': 3,
        'flip_dot_threshold': 0.0,
        'pending_consistency_dot': 0.85,
        'max_missed_frames': 3,
    }
    parameters.update(overrides)
    return ToolAxisStabilizer(**parameters)


def test_single_opposite_axis_measurement_is_held():
    stabilizer = _stabilizer()
    stabilizer.update('mayo_adson#1', IDENTITY)

    decision = stabilizer.update('mayo_adson#1', PLANAR_FLIP)

    assert decision.reason == 'AXIS_FLIP_HELD'
    assert decision.quaternion_xyzw == pytest.approx(IDENTITY)
    assert decision.y_axis_dot == pytest.approx(-1.0)


def test_consistent_opposite_axis_is_accepted_after_confirmation():
    stabilizer = _stabilizer()
    stabilizer.update('mayo_adson#1', IDENTITY)

    first = stabilizer.update('mayo_adson#1', PLANAR_FLIP)
    second = stabilizer.update('mayo_adson#1', PLANAR_FLIP)
    third = stabilizer.update('mayo_adson#1', PLANAR_FLIP)

    assert first.quaternion_xyzw == pytest.approx(IDENTITY)
    assert second.quaternion_xyzw == pytest.approx(IDENTITY)
    assert third.reason == 'AXIS_FLIP_CONFIRMED'
    assert third.quaternion_xyzw == pytest.approx(PLANAR_FLIP)
    assert stabilizer.flip_held_total == 2
    assert stabilizer.flip_confirmed_total == 1


def test_return_to_original_axis_cancels_pending_flip():
    stabilizer = _stabilizer()
    stabilizer.update('tray_bovie#1', IDENTITY)
    stabilizer.update('tray_bovie#1', PLANAR_FLIP)

    recovered = stabilizer.update('tray_bovie#1', IDENTITY)
    next_flip = stabilizer.update('tray_bovie#1', PLANAR_FLIP)

    assert recovered.reason == 'AXIS_ACCEPTED'
    assert next_flip.reason == 'AXIS_FLIP_HELD'
    assert next_flip.quaternion_xyzw == pytest.approx(IDENTITY)


def test_quaternion_sign_equivalence_is_not_treated_as_axis_flip():
    stabilizer = _stabilizer()
    stabilizer.update('tray_mosquito#1', IDENTITY)

    decision = stabilizer.update('tray_mosquito#1', (0.0, 0.0, 0.0, -1.0))

    assert decision.reason == 'AXIS_ACCEPTED'
    assert decision.quaternion_xyzw == pytest.approx(IDENTITY)
    assert decision.y_axis_dot == pytest.approx(1.0)


def test_filter_is_keyed_by_tool_slot_not_class_policy():
    stabilizer = _stabilizer()
    stabilizer.update('mayo_adson#1', IDENTITY)
    stabilizer.update('mayo_bipolar#1', PLANAR_FLIP)

    adson = stabilizer.update('mayo_adson#1', PLANAR_FLIP)
    bipolar = stabilizer.update('mayo_bipolar#1', IDENTITY)

    assert adson.quaternion_xyzw == pytest.approx(IDENTITY)
    assert bipolar.quaternion_xyzw == pytest.approx(PLANAR_FLIP)


def test_spatial_selector_reset_discards_stale_axis_state():
    stabilizer = _stabilizer()
    stabilizer.update('mayo_adson#1', IDENTITY)
    stabilizer.reset_keys({'mayo_adson#1'})

    decision = stabilizer.update('mayo_adson#1', PLANAR_FLIP)

    assert decision.reason == 'INITIALIZED'
    assert decision.quaternion_xyzw == pytest.approx(PLANAR_FLIP)
    assert stabilizer.association_reset_total == 1
