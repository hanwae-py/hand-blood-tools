"""ROS-independent temporal stabilization for tool-axis direction flips."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass
class _AxisFilterState:
    """Last accepted orientation and an unconfirmed opposite-axis sample."""

    quaternion_xyzw: tuple[float, float, float, float]
    y_axis: tuple[float, float, float]
    pending_y_axis: tuple[float, float, float] | None = None
    pending_count: int = 0
    missed_frames: int = 0


@dataclass(frozen=True)
class AxisStabilizationDecision:
    """One control-facing quaternion and how its axis sign was handled."""

    quaternion_xyzw: tuple[float, float, float, float]
    reason: str
    y_axis_dot: float


class ToolAxisStabilizer:
    """Reject isolated 180-degree longitudinal-axis reversals.

    The constrained surgical-tool frame uses +Y as its longitudinal signed
    axis. A segmentation or endpoint-sign error reverses both X and Y while
    preserving Z, producing a physically different 180-degree planar pose.
    Opposite-hemisphere measurements must therefore remain mutually
    consistent for several frames before replacing the last accepted pose.

    This filter is class-independent and keyed by the same spatial selector
    used for control-facing TF publication. It deliberately does not low-pass
    ordinary angular motion, so genuine gradual rotations remain responsive.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        flip_confirmation_frames: int = 3,
        flip_dot_threshold: float = 0.0,
        pending_consistency_dot: float = 0.85,
        max_missed_frames: int = 3,
    ) -> None:
        if int(flip_confirmation_frames) < 1:
            raise ValueError('flip_confirmation_frames must be at least one')
        if not -1.0 <= float(flip_dot_threshold) <= 1.0:
            raise ValueError('flip_dot_threshold must be in [-1, 1]')
        if not -1.0 <= float(pending_consistency_dot) <= 1.0:
            raise ValueError('pending_consistency_dot must be in [-1, 1]')
        if int(max_missed_frames) < 0:
            raise ValueError('max_missed_frames must be non-negative')

        self.enabled = bool(enabled)
        self.flip_confirmation_frames = int(flip_confirmation_frames)
        self.flip_dot_threshold = float(flip_dot_threshold)
        self.pending_consistency_dot = float(pending_consistency_dot)
        self.max_missed_frames = int(max_missed_frames)
        self._states: dict[str, _AxisFilterState] = {}
        self._flip_held_total = 0
        self._flip_confirmed_total = 0
        self._association_reset_total = 0

    @property
    def active_count(self) -> int:
        return len(self._states)

    @property
    def flip_held_total(self) -> int:
        return self._flip_held_total

    @property
    def flip_confirmed_total(self) -> int:
        return self._flip_confirmed_total

    @property
    def association_reset_total(self) -> int:
        return self._association_reset_total

    def reset(self) -> None:
        self._states.clear()

    def reset_keys(self, keys: set[str]) -> None:
        """Drop slots whose spatial-selector meaning has just changed."""
        removed = 0
        for key in keys:
            if self._states.pop(key, None) is not None:
                removed += 1
        self._association_reset_total += removed

    @staticmethod
    def _normalized_quaternion(
        quaternion_xyzw: Sequence[float],
    ) -> tuple[float, float, float, float]:
        quaternion = tuple(float(value) for value in quaternion_xyzw)
        if len(quaternion) != 4 or not all(
            math.isfinite(value) for value in quaternion
        ):
            raise ValueError('quaternion_xyzw must contain four finite values')
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise ValueError('quaternion_xyzw must have non-zero norm')
        return tuple(value / norm for value in quaternion)

    @staticmethod
    def _y_axis(
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z, w = quaternion_xyzw
        return (
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        )

    @staticmethod
    def _dot(left: Sequence[float], right: Sequence[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right, strict=True)))

    @classmethod
    def _same_quaternion_hemisphere(
        cls,
        quaternion: tuple[float, float, float, float],
        reference: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if cls._dot(quaternion, reference) < 0.0:
            return tuple(-value for value in quaternion)
        return quaternion

    def update(
        self,
        key: str,
        quaternion_xyzw: Sequence[float],
    ) -> AxisStabilizationDecision:
        """Update one selector slot and return its control-facing orientation."""
        measurement = self._normalized_quaternion(quaternion_xyzw)
        measurement_y = self._y_axis(measurement)
        if not self.enabled:
            return AxisStabilizationDecision(measurement, 'DISABLED', 1.0)

        state = self._states.get(key)
        if state is None:
            self._states[key] = _AxisFilterState(measurement, measurement_y)
            return AxisStabilizationDecision(measurement, 'INITIALIZED', 1.0)

        state.missed_frames = 0
        measurement = self._same_quaternion_hemisphere(
            measurement, state.quaternion_xyzw
        )
        axis_dot = self._dot(measurement_y, state.y_axis)
        if axis_dot >= self.flip_dot_threshold:
            state.quaternion_xyzw = measurement
            state.y_axis = measurement_y
            state.pending_y_axis = None
            state.pending_count = 0
            return AxisStabilizationDecision(
                measurement, 'AXIS_ACCEPTED', axis_dot
            )

        if (
            state.pending_y_axis is None
            or self._dot(measurement_y, state.pending_y_axis)
            < self.pending_consistency_dot
        ):
            state.pending_count = 1
        else:
            state.pending_count += 1
        state.pending_y_axis = measurement_y

        if state.pending_count >= self.flip_confirmation_frames:
            state.quaternion_xyzw = measurement
            state.y_axis = measurement_y
            state.pending_y_axis = None
            state.pending_count = 0
            self._flip_confirmed_total += 1
            return AxisStabilizationDecision(
                measurement, 'AXIS_FLIP_CONFIRMED', axis_dot
            )

        self._flip_held_total += 1
        return AxisStabilizationDecision(
            state.quaternion_xyzw, 'AXIS_FLIP_HELD', axis_dot
        )

    def finish_frame(self, active_keys: set[str]) -> None:
        """Expire slots absent for more than the configured frame allowance."""
        if not self.enabled:
            return
        expired: list[str] = []
        for key, state in self._states.items():
            if key in active_keys:
                continue
            state.missed_frames += 1
            if state.missed_frames > self.max_missed_frames:
                expired.append(key)
        for key in expired:
            del self._states[key]
