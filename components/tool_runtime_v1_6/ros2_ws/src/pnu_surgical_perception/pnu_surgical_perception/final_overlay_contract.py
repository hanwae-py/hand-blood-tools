"""Pure freshness and status helpers for the single final Debug overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STATUS_SCHEMA = 'pnu.perception.final_overlay.v1'
CAMERA_STATUS_KEYS = {'cam_3': 'cam3', 'cam_4': 'cam4'}
LAYER_NAMES = ('tool', 'pose', 'hand', 'blood')


def stamp_ns(header_or_message: Any) -> int:
    """Return a source stamp without interpreting it as the local clock."""
    header = getattr(header_or_message, 'header', header_or_message)
    stamp = header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_dict(header_or_message: Any | None) -> dict[str, int] | None:
    """Make the JSON representation required by the Debug status contract."""
    if header_or_message is None:
        return None
    header = getattr(header_or_message, 'header', header_or_message)
    return {
        'sec': int(header.stamp.sec),
        'nanosec': int(header.stamp.nanosec),
    }


def freshness_state(
    *, has_value: bool, age_sec: float | None, max_age_sec: float, disabled: bool = False
) -> str:
    """Return the intentionally small public state enum for one input."""
    if disabled:
        return 'disabled'
    if not has_value:
        return 'missing'
    if age_sec is None or age_sec > max_age_sec:
        return 'stale'
    return 'live'


@dataclass(frozen=True)
class Freshness:
    """Receiver-age metadata; source timestamps remain separately lossless."""

    received_monotonic: float | None

    def age(self, now_monotonic: float) -> float | None:
        if self.received_monotonic is None:
            return None
        return max(0.0, float(now_monotonic) - self.received_monotonic)


def layer_is_drawable(
    *, base_stamp_ns: int | None, layer_stamp_ns: int | None, base_state: str,
    layer_state: str, max_source_delta_ns: int,
) -> bool:
    """Only draw data belonging to the current base frame's time neighborhood."""
    return (
        base_state == 'live'
        and layer_state == 'live'
        and base_stamp_ns is not None
        and layer_stamp_ns is not None
        and abs(base_stamp_ns - layer_stamp_ns) <= max_source_delta_ns
    )
