"""Binary open/closed classification from anatomical landmark ordering."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
# The thumb lane is represented by its MCP; other lanes use their MCP.
LANE_ANCHORS = (2, 5, 9, 13, 17)


@dataclass(frozen=True)
class AnatomicalRuleConfig:
    # A distal point must reach this close to a second-neighbor lane before it
    # counts as crossing beyond the immediate neighbor.
    second_neighbor_margin_ratio: float = 0.03
    # The tip must remain at least this far distal of the second joint.
    minimum_tip_beyond_second_ratio: float = 0.05


def classify_anatomical_rules(
    landmarks: Iterable[Iterable[float]],
    config: AnatomicalRuleConfig = AnatomicalRuleConfig(),
) -> dict:
    """Classify a MediaPipe hand as OPEN unless an anatomical rule is broken.

    Rule 1: DIP/TIP points may enter an immediate neighbor's lane, but cannot
    reach or cross the next (second-neighbor) finger lane.

    Rule 2: a fingertip cannot fold proximally back to or past its second joint
    (thumb MCP; PIP for the other fingers).
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 2):
        raise ValueError("landmarks must have shape (21, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("landmarks must contain only finite coordinates")

    lateral_vector = points[17] - points[5]
    palm_width = float(np.linalg.norm(lateral_vector))
    if palm_width < 1e-6:
        return {
            "state": "CLOSED",
            "palm_width_px": round(palm_width, 4),
            "lateral_crossings": [],
            "proximal_folds": [],
            "reason": "degenerate palm width",
        }
    lateral_axis = lateral_vector / palm_width
    lateral = points @ lateral_axis
    lane_positions = [float(lateral[index]) for index in LANE_ANCHORS]

    lateral_crossings = []
    margin = config.second_neighbor_margin_ratio * palm_width
    for finger_index, (name, chain) in enumerate(zip(FINGER_NAMES, FINGER_CHAINS)):
        current_lane = lane_positions[finger_index]
        # A second neighbor exists two fingers to either side. The immediate
        # neighbor between them may be touched/crossed without closing the hand.
        for neighbor_index in (finger_index - 2, finger_index + 2):
            if not 0 <= neighbor_index < len(FINGER_NAMES):
                continue
            boundary_lane = lane_positions[neighbor_index]
            direction = float(np.sign(boundary_lane - current_lane))
            if direction == 0.0:
                continue
            for point_index in chain[-2:]:  # DIP/IP and fingertip
                signed_remaining = direction * (boundary_lane - float(lateral[point_index]))
                if signed_remaining <= margin:
                    lateral_crossings.append(
                        {
                            "finger": name,
                            "point_index": point_index,
                            "crossed_toward": FINGER_NAMES[neighbor_index],
                            "second_neighbor_anchor": LANE_ANCHORS[neighbor_index],
                            "remaining_ratio": round(signed_remaining / palm_width, 4),
                        }
                    )

    proximal_folds = []
    minimum_distal_gap = config.minimum_tip_beyond_second_ratio * palm_width
    for name, (base, second, immediate, tip) in zip(FINGER_NAMES, FINGER_CHAINS):
        distal_vector = points[immediate] - points[base]
        distal_norm = float(np.linalg.norm(distal_vector))
        if distal_norm < 1e-6:
            proximal_folds.append(
                {
                    "finger": name,
                    "tip_index": tip,
                    "second_joint_index": second,
                    "reason": "degenerate finger axis",
                }
            )
            continue
        distal_axis = distal_vector / distal_norm
        second_projection = float(np.dot(points[second] - points[base], distal_axis))
        tip_projection = float(np.dot(points[tip] - points[base], distal_axis))
        gap = tip_projection - second_projection
        if gap <= minimum_distal_gap:
            proximal_folds.append(
                {
                    "finger": name,
                    "tip_index": tip,
                    "second_joint_index": second,
                    "distal_gap_ratio": round(gap / palm_width, 4),
                }
            )

    state = "OPEN" if not lateral_crossings and not proximal_folds else "CLOSED"
    return {
        "state": state,
        "palm_width_px": round(palm_width, 4),
        "lateral_crossings": lateral_crossings,
        "proximal_folds": proximal_folds,
    }
