"""Tests for class-independent surgical-tool axis stabilization."""

from __future__ import annotations

import math

import pytest

from pnu_surgical_perception.tool_axis_stabilizer import ToolAxisStabilizer


IDENTITY = (0.0, 0.0, 0.0, 1.0)
PLANAR_FLIP = (0.0, 0.0, 1.0, 0.0)


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    half_angle = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))


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


def test_stationary_angular_jitter_inside_deadband_is_held():
    stabilizer = _stabilizer(angular_deadband_rad=math.radians(1.5))
    stabilizer.update('mayo_adson#1', IDENTITY)

    decision = stabilizer.update('mayo_adson#1', _yaw_quaternion(1.0))

    assert decision.reason == 'AXIS_ANGULAR_DEADBAND_HELD'
    assert decision.quaternion_xyzw == pytest.approx(IDENTITY)
    assert stabilizer.angular_held_total == 1


def test_ordinary_rotation_is_slerp_smoothed():
    stabilizer = _stabilizer(smoothing_alpha=0.25)
    stabilizer.update('mayo_adson#1', IDENTITY)

    decision = stabilizer.update('mayo_adson#1', _yaw_quaternion(20.0))

    assert decision.reason == 'AXIS_SLERP_SMOOTHED'
    assert decision.quaternion_xyzw == pytest.approx(_yaw_quaternion(5.0))
    assert stabilizer.angular_smoothed_total == 1


def test_opposite_axis_lock_never_accepts_persistent_static_flip():
    stabilizer = _stabilizer(lock_opposite_axis=True)
    stabilizer.update('mayo_adson#1', IDENTITY)

    decisions = [
        stabilizer.update('mayo_adson#1', PLANAR_FLIP)
        for _ in range(10)
    ]

    assert all(item.reason == 'AXIS_FLIP_HELD' for item in decisions)
    assert all(
        item.quaternion_xyzw == pytest.approx(IDENTITY)
        for item in decisions
    )
    assert stabilizer.flip_held_total == 10
    assert stabilizer.flip_confirmed_total == 0


def test_confirmed_relocation_can_establish_a_new_axis_sign():
    stabilizer = _stabilizer(lock_opposite_axis=True)
    stabilizer.update('mayo_adson#1', IDENTITY)
    stabilizer.update('mayo_adson#1', PLANAR_FLIP)

    stabilizer.reset_for_relocation('mayo_adson#1')
    decision = stabilizer.update('mayo_adson#1', PLANAR_FLIP)

    assert decision.reason == 'INITIALIZED'
    assert decision.quaternion_xyzw == pytest.approx(PLANAR_FLIP)
    assert stabilizer.relocation_reset_total == 1


def test_low_confidence_opposite_axis_does_not_advance_confirmation():
    stabilizer = _stabilizer(flip_confirmation_frames=3)
    stabilizer.update('mayo_adson#1', IDENTITY)

    first = stabilizer.update('mayo_adson#1', PLANAR_FLIP)
    invalid = stabilizer.update(
        'mayo_adson#1',
        PLANAR_FLIP,
        allow_flip_confirmation=False,
    )
    after_reset = stabilizer.update('mayo_adson#1', PLANAR_FLIP)

    assert first.reason == 'AXIS_FLIP_HELD'
    assert invalid.reason == 'AXIS_FLIP_LOW_CONFIDENCE_HELD'
    assert after_reset.reason == 'AXIS_FLIP_HELD'
    assert invalid.quaternion_xyzw == pytest.approx(IDENTITY)
    assert stabilizer.flip_confirmed_total == 0
