"""Cutie tracker wrapper for a single binary object."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from blood.pipeline.loaders import load_cutie, resolve_cutie_checkpoint


class CutieTracker:
    def __init__(self, checkpoint: str | Path | None, max_internal_size: int = 480) -> None:
        ckpt = resolve_cutie_checkpoint(checkpoint)
        self.checkpoint = ckpt
        self.max_internal_size = max_internal_size
        self.model, self.cfg = load_cutie(ckpt)
        self.processor = None

    def reset(self) -> None:
        from cutie.inference.inference_core import InferenceCore

        self.processor = InferenceCore(self.model, cfg=self.cfg)
        self.processor.max_internal_size = self.max_internal_size

    def _image(self, rgb: np.ndarray) -> torch.Tensor:
        return to_tensor(Image.fromarray(rgb)).cuda().float()

    def init(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.reset()
        if not mask.any():
            return np.zeros(mask.shape, dtype=bool)
        image = self._image(rgb)
        tmask = torch.from_numpy(mask.astype(np.uint8)).cuda()
        prob = self.processor.step(image, tmask, objects=[1])
        out = self.processor.output_prob_to_mask(prob).cpu().numpy() > 0
        return out.astype(bool)

    def step(self, rgb: np.ndarray) -> np.ndarray:
        if self.processor is None or self.processor.object_manager.num_obj == 0:
            h, w = rgb.shape[:2]
            return np.zeros((h, w), dtype=bool)
        image = self._image(rgb)
        prob = self.processor.step(image)
        out = self.processor.output_prob_to_mask(prob).cpu().numpy() > 0
        return out.astype(bool)

    def reinit(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.init(rgb, mask)
