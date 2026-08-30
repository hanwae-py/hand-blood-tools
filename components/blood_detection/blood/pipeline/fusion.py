"""Rule-based detector vs tracker update policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blood.pipeline.postprocess import filter_components


@dataclass
class FusionConfig:
    iou_agree: float = 0.5
    iou_disagree: float = 0.2
    min_new_area: int = 400
    min_det_area: int = 200
    min_conf: float = 0.4
    persist_disagree: int = 3


@dataclass
class FusionState:
    disagree_streak: int = 0


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def update(
    track: np.ndarray,
    det: np.ndarray,
    det_conf: float,
    cfg: FusionConfig,
    state: FusionState,
) -> tuple[np.ndarray, str]:
    det_f = filter_components(det, min_area=cfg.min_det_area)
    track_b = track.astype(bool)
    det_b = det_f.astype(bool)

    if det_conf < cfg.min_conf or not det_b.any():
        return track_b, "keep_cutie_miss"

    iou = _iou(track_b, det_b)
    new_area = int(np.logical_and(det_b, ~track_b).sum())

    if iou >= cfg.iou_agree:
        state.disagree_streak = 0
        return track_b, "agree_correct"

    if new_area >= cfg.min_new_area:
        state.disagree_streak = 0
        return np.logical_or(track_b, det_b), "add_new"

    if iou < cfg.iou_disagree:
        state.disagree_streak += 1
        if state.disagree_streak >= cfg.persist_disagree:
            state.disagree_streak = 0
            return det_b, "reinit"
        return track_b, "disagree_hold"

    return track_b, "keep_cutie"
