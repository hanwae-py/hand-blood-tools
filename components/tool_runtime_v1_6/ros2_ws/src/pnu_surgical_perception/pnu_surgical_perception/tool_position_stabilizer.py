"""ROS-independent temporal stabilization for control-facing tool positions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass
class _PositionFilterState:
    """Filtered translation and any unconfirmed relocation candidate."""

    position_m: tuple[float, float, float]
    pending_position_m: tuple[float, float, float] | None = None
    pending_count: int = 0
    missed_frames: int = 0


@dataclass(frozen=True)
class PositionStabilizationDecision:
    """One filtered translation and how the current sample was handled."""

    position_m: tuple[float, float, float]
    reason: str
    raw_displacement_m: float


class ToolPositionStabilizer:
    """Suppress camera-pixel pose jitter without hiding real relocation.

    Small motion inside ``deadband_m`` is held exactly. Moderate motion is
    blended with an exponential moving average. A jump larger than
    ``max_jump_m`` is treated as a possible segmentation outlier and must be
    observed consistently in consecutive frames before it replaces the
    filtered position. State is keyed by the operator-facing spatial selector
    (for example ``mayo_army#1``), never by frame-local detector IDs.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        deadband_m: float = 0.0,
        smoothing_alpha: float = 0.20,
        max_jump_m: float = 0.04,
        relocation_confirmation_frames: int = 2,
        relocation_consistency_m: float = 0.015,
        max_missed_frames: int = 3,
    ) -> None:
        if not math.isfinite(deadband_m) or deadband_m < 0.0:
            raise ValueError('deadband_m must be finite and non-negative')
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError('smoothing_alpha must be in (0, 1]')
        if not math.isfinite(max_jump_m) or max_jump_m <= deadband_m:
            raise ValueError('max_jump_m must be greater than deadband_m')
        if int(relocation_confirmation_frames) < 1:
            raise ValueError(
                'relocation_confirmation_frames must be at least one'
            )
        if (
            not math.isfinite(relocation_consistency_m)
            or relocation_consistency_m < 0.0
        ):
            raise ValueError(
                'relocation_consistency_m must be finite and non-negative'
            )
        if int(max_missed_frames) < 0:
            raise ValueError('max_missed_frames must be non-negative')

        self.enabled = bool(enabled)
        self.deadband_m = float(deadband_m)
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_jump_m = float(max_jump_m)
        self.relocation_confirmation_frames = int(
            relocation_confirmation_frames
        )
        self.relocation_consistency_m = float(relocation_consistency_m)
        self.max_missed_frames = int(max_missed_frames)
        self._states: dict[str, _PositionFilterState] = {}
        self._held_total = 0
        self._smoothed_total = 0
        self._outlier_held_total = 0
        self._relocation_total = 0
        self._association_reset_total = 0

    @property
    def active_count(self) -> int:
        return len(self._states)

    @property
    def held_total(self) -> int:
        return self._held_total

    @property
    def smoothed_total(self) -> int:
        return self._smoothed_total

    @property
    def outlier_held_total(self) -> int:
        return self._outlier_held_total

    @property
    def relocation_total(self) -> int:
        return self._relocation_total

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
    def _finite_position(
        position_m: Sequence[float],
    ) -> tuple[float, float, float]:
        position = tuple(float(value) for value in position_m)
        if len(position) != 3 or not all(
            math.isfinite(value) for value in position
        ):
            raise ValueError('position_m must contain three finite values')
        return position

    def update(
        self, key: str, position_m: Sequence[float]
    ) -> PositionStabilizationDecision:
        """Update one selector slot and return its control-facing position."""
        measurement = self._finite_position(position_m)
        if not self.enabled:
            return PositionStabilizationDecision(measurement, 'DISABLED', 0.0)

        state = self._states.get(key)
        if state is None:
            self._states[key] = _PositionFilterState(measurement)
            return PositionStabilizationDecision(
                measurement, 'INITIALIZED', 0.0
            )

        state.missed_frames = 0
        displacement = math.dist(measurement, state.position_m)
        if displacement <= self.deadband_m:
            state.pending_position_m = None
            state.pending_count = 0
            self._held_total += 1
            return PositionStabilizationDecision(
                state.position_m, 'DEADBAND_HELD', displacement
            )

        if displacement <= self.max_jump_m:
            alpha = self.smoothing_alpha
            state.position_m = tuple(
                previous + alpha * (current - previous)
                for previous, current in zip(
                    state.position_m, measurement, strict=True
                )
            )
            state.pending_position_m = None
            state.pending_count = 0
            self._smoothed_total += 1
            return PositionStabilizationDecision(
                state.position_m, 'EMA_SMOOTHED', displacement
            )

        pending = state.pending_position_m
        if (
            pending is None
            or math.dist(measurement, pending) > self.relocation_consistency_m
        ):
            state.pending_position_m = measurement
            state.pending_count = 1
        else:
            count = state.pending_count + 1
            state.pending_position_m = tuple(
                previous + (current - previous) / count
                for previous, current in zip(pending, measurement, strict=True)
            )
            state.pending_count = count

        if state.pending_count >= self.relocation_confirmation_frames:
            state.position_m = state.pending_position_m
            state.pending_position_m = None
            state.pending_count = 0
            self._relocation_total += 1
            return PositionStabilizationDecision(
                state.position_m, 'RELOCATION_CONFIRMED', displacement
            )

        self._outlier_held_total += 1
        return PositionStabilizationDecision(
            state.position_m, 'OUTLIER_HELD', displacement
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
