"""End-to-end RGB -> RF-DETR -> Cutie -> stable mask -> centroids."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from blood.pipeline.detector import RFDETRDetector
from blood.pipeline.fusion import FusionConfig, FusionState, update as fusion_update
from blood.pipeline.postprocess import filter_components, region_centroids
from blood.pipeline.tracker import CutieTracker


@dataclass
class PipelineConfig:
    redetect_interval: int = 1
    score_thr: float = 0.5
    min_area: int = 400
    max_internal_size: int = 480
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class FrameOutput:
    mask: np.ndarray
    centroids: list[dict]
    action: str
    ran_detector: bool
    detector_ms: float
    tracker_ms: float
    total_ms: float


class BloodPipeline:
    def __init__(
        self,
        rfdetr_ckpt: str | Path | None,
        cutie_ckpt: str | Path | None,
        cfg: PipelineConfig | None = None,
    ) -> None:
        self.cfg = cfg or PipelineConfig()
        self.detector = RFDETRDetector(rfdetr_ckpt, score_thr=self.cfg.score_thr)
        self.tracker = CutieTracker(cutie_ckpt, max_internal_size=self.cfg.max_internal_size)
        self.state = FusionState()
        self.initialized = False
        self.t = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.state = FusionState()
        self.initialized = False
        self.t = 0

    @torch.inference_mode()
    def step(self, rgb: np.ndarray) -> FrameOutput:
        import time

        t0 = time.perf_counter()
        run_det = (self.t % self.cfg.redetect_interval == 0) or (not self.initialized)
        det_ms = 0.0
        det_mask = None
        det_conf = 0.0
        if run_det:
            td = time.perf_counter()
            det = self.detector.detect(rgb)
            det_ms = (time.perf_counter() - td) * 1000
            det_mask = filter_components(det["mask"], min_area=self.cfg.min_area)
            det_conf = det["confidence"]

        tt = time.perf_counter()
        action = "propagate"
        if not self.initialized:
            if det_mask is not None and det_mask.any() and det_conf >= self.cfg.fusion.min_conf:
                pred = self.tracker.init(rgb, det_mask)
                self.initialized = True
                action = "init"
            else:
                pred = np.zeros(rgb.shape[:2], dtype=bool)
                action = "wait_detection"
            track_ms = (time.perf_counter() - tt) * 1000
        else:
            track = self.tracker.step(rgb)
            track_ms = (time.perf_counter() - tt) * 1000
            if det_mask is None:
                pred = track
                action = "propagate"
            else:
                pred, action = fusion_update(track, det_mask, det_conf, self.cfg.fusion, self.state)
                if action in {"add_new", "reinit"}:
                    tr = time.perf_counter()
                    pred = self.tracker.reinit(rgb, pred)
                    track_ms += (time.perf_counter() - tr) * 1000

        pred = filter_components(pred, min_area=self.cfg.min_area)
        cents = region_centroids(pred, min_area=self.cfg.min_area)
        self.t += 1
        total_ms = (time.perf_counter() - t0) * 1000
        return FrameOutput(
            mask=pred,
            centroids=cents,
            action=action,
            ran_detector=run_det,
            detector_ms=det_ms,
            tracker_ms=track_ms,
            total_ms=total_ms,
        )
