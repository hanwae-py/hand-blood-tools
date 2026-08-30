"""RF-DETR frame detector producing a union binary blood mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from blood.pipeline.loaders import load_rfdetr, resolve_rfdetr_checkpoint, union_detections


class RFDETRDetector:
    def __init__(self, checkpoint: str | Path | None, score_thr: float = 0.5, device: str = "cuda") -> None:
        ckpt = resolve_rfdetr_checkpoint(checkpoint)
        self.checkpoint = ckpt
        self.score_thr = score_thr
        self.model = load_rfdetr(ckpt, device=device)

    def detect(self, rgb: np.ndarray) -> dict:
        dets = self.model.predict(Image.fromarray(rgb), threshold=self.score_thr)
        union, instances = union_detections(dets, self.score_thr)
        h, w = rgb.shape[:2]
        if union is None:
            union = np.zeros((h, w), dtype=bool)
        elif union.shape != (h, w):
            import cv2

            union = cv2.resize(union.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        conf = max((s for _, s in instances), default=0.0)
        return {
            "mask": union,
            "confidence": conf,
            "n_instances": len(instances),
            "area": int(union.sum()),
        }
