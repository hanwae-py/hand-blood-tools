"""Small one-to-one timestamp matcher for RGB and native-depth messages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


def stamp_ns(message: Any) -> int:
    """Return a ROS message header stamp as integer nanoseconds."""
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class RgbDepthPair:
    """One RGB/depth pair whose source stamps are within tolerance."""

    rgb: Any
    depth: Any
    delta_ns: int


class ApproximateRgbDepthPairer:
    """Pair each RGB/depth message at most once using closest timestamps."""

    def __init__(self, maximum_delta_ns: int, queue_size: int = 8) -> None:
        """Initialize a bounded matcher with nanosecond tolerance."""
        if maximum_delta_ns < 0:
            raise ValueError('maximum_delta_ns must be non-negative')
        if queue_size < 1:
            raise ValueError('queue_size must be positive')
        self.maximum_delta_ns = int(maximum_delta_ns)
        self.queue_size = int(queue_size)
        self._rgb: deque[Any] = deque()
        self._depth: deque[Any] = deque()
        self.dropped_unmatched = 0

    def add_rgb(self, message: Any) -> RgbDepthPair | None:
        """Add an RGB message and return a pair when one is available."""
        return self._add(message, self._rgb, self._depth, incoming_is_rgb=True)

    def add_depth(self, message: Any) -> RgbDepthPair | None:
        """Add a depth message and return a pair when one is available."""
        return self._add(
            message, self._depth, self._rgb, incoming_is_rgb=False
        )

    def _add(
        self,
        message: Any,
        own: deque[Any],
        opposite: deque[Any],
        *,
        incoming_is_rgb: bool,
    ) -> RgbDepthPair | None:
        own.append(message)
        while len(own) > self.queue_size:
            own.popleft()
            self.dropped_unmatched += 1
        if not opposite:
            return None

        incoming_stamp = stamp_ns(message)
        deltas = [abs(incoming_stamp - stamp_ns(item)) for item in opposite]
        best_index = min(range(len(deltas)), key=deltas.__getitem__)
        delta_ns = int(deltas[best_index])
        if delta_ns > self.maximum_delta_ns:
            return None

        own.pop()
        matched = opposite[best_index]
        del opposite[best_index]
        if incoming_is_rgb:
            return RgbDepthPair(message, matched, delta_ns)
        return RgbDepthPair(matched, message, delta_ns)

    @property
    def queued_rgb(self) -> int:
        """Return the number of unmatched RGB messages."""
        return len(self._rgb)

    @property
    def queued_depth(self) -> int:
        """Return the number of unmatched depth messages."""
        return len(self._depth)
