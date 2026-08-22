from types import SimpleNamespace

from pnu_surgical_perception.final_overlay_contract import (
    Freshness,
    freshness_state,
    layer_is_drawable,
    stamp_dict,
    stamp_ns,
)


def _header(sec=12, nanosec=34):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def test_status_stamp_is_source_stamp_and_not_a_new_timestamp():
    header = _header(42, 123)
    assert stamp_ns(header) == 42_000_000_123
    assert stamp_dict(header) == {'sec': 42, 'nanosec': 123}


def test_freshness_state_has_only_public_contract_values():
    assert freshness_state(has_value=False, age_sec=None, max_age_sec=1.0) == 'missing'
    assert freshness_state(has_value=True, age_sec=1.1, max_age_sec=1.0) == 'stale'
    assert freshness_state(has_value=True, age_sec=0.1, max_age_sec=1.0) == 'live'
    assert freshness_state(has_value=True, age_sec=0.1, max_age_sec=1.0, disabled=True) == 'disabled'


def test_layer_can_never_overlay_an_unrelated_or_stale_base_frame():
    assert layer_is_drawable(
        base_stamp_ns=10_000, layer_stamp_ns=10_500, base_state='live',
        layer_state='live', max_source_delta_ns=1_000)
    assert not layer_is_drawable(
        base_stamp_ns=10_000, layer_stamp_ns=12_000, base_state='live',
        layer_state='live', max_source_delta_ns=1_000)
    assert not layer_is_drawable(
        base_stamp_ns=10_000, layer_stamp_ns=10_500, base_state='stale',
        layer_state='live', max_source_delta_ns=1_000)


def test_receiver_age_is_monotonic_and_never_negative():
    freshness = Freshness(100.0)
    assert freshness.age(100.25) == 0.25
    assert freshness.age(99.0) == 0.0
    assert Freshness(None).age(101.0) is None
