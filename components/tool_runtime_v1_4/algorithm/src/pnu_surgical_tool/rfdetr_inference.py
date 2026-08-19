"""Fine-tuned RF-DETR segmentation adapter with an explicit color contract."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .types import DetectionBatch, DetectionInstance


@dataclass(frozen=True)
class DetectorConfig:
    checkpoint_path: str | Path
    ontology_path: str | Path
    confidence_threshold: float = 0.5
    optimize: bool = True
    jit_compile: bool = True
    fp16: bool = True
    model_version: str = "cam4-rfdetr-seg-small-v1"
    checkpoint_color_order: Literal["RGB", "BGR"] = "BGR"


class SurgicalToolDetector:
    """Load the supplied checkpoint and return class/bbox/instance-mask results.

    Input arrays may be RGB or BGR. They are normalized to the color order used
    by the supplied checkpoint's validated inference path (BGR for v1).
    Loading is intentionally lazy so pose-only users do not need RF-DETR.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model = None
        payload = json.loads(Path(config.ontology_path).read_text(encoding="utf-8"))
        classes = payload["canonical_tool_classes"]
        self._classes = sorted(classes, key=lambda item: int(item["canonical_id"]))
        self.ontology_version = str(payload.get("schema", "unknown"))
        if [int(item["canonical_id"]) for item in self._classes] != list(range(1, 9)):
            raise ValueError("Expected frozen canonical IDs 1..8")
        if not 0.0 <= config.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if config.checkpoint_color_order not in ("RGB", "BGR"):
            raise ValueError("checkpoint_color_order must be 'RGB' or 'BGR'")

    def load(self) -> None:
        if self._model is not None:
            return
        checkpoint = Path(self.config.checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        import torch
        from rfdetr import RFDETRSegSmall

        model = RFDETRSegSmall.from_checkpoint(str(checkpoint))
        if self.config.optimize:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "optimized mode requires CUDA; set optimize=False for a CPU smoke test"
                )
            dtype = torch.float16 if self.config.fp16 else torch.float32
            model.optimize_for_inference(
                compile=self.config.jit_compile,
                batch_size=1,
                dtype=dtype,
                inplace=False,
            )
        self._model = model

    def predict(
        self,
        image: np.ndarray,
        color_order: Literal["RGB", "BGR"],
        confidence_threshold: float | None = None,
    ) -> DetectionBatch:
        frame = np.asarray(image)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("image must be uint8 HxWx3")
        if color_order not in ("RGB", "BGR"):
            raise ValueError("color_order must be exactly 'RGB' or 'BGR'")
        threshold = (
            self.config.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.load()
        if color_order == self.config.checkpoint_color_order:
            model_image = frame
        else:
            model_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        detections = self._model.predict(
            model_image,
            threshold=threshold,
            include_source_image=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0

        height, width = frame.shape[:2]
        masks = getattr(detections, "mask", None)
        if masks is None:
            raise RuntimeError("Checkpoint did not return segmentation masks")
        instances: list[DetectionInstance] = []
        for index, (box, model_index, confidence) in enumerate(
            zip(
                np.asarray(detections.xyxy),
                np.asarray(detections.class_id, dtype=int),
                np.asarray(detections.confidence),
                strict=True,
            )
        ):
            if model_index < 0 or model_index >= len(self._classes):
                raise RuntimeError(f"Unexpected model class index: {model_index}")
            mask = np.asarray(masks[index], dtype=bool)
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            class_record = self._classes[model_index]
            instances.append(
                DetectionInstance(
                    frame_local_instance_id=index,
                    canonical_class_id=int(class_record["canonical_id"]),
                    model_class_index=int(model_index),
                    class_name=str(class_record["canonical_name"]),
                    class_confidence=float(confidence),
                    bbox_xyxy_px=tuple(float(value) for value in box),
                    mask=mask,
                )
            )
        return DetectionBatch(
            image_width=width,
            image_height=height,
            model_version=self.config.model_version,
            ontology_version=self.ontology_version,
            instances=instances,
            inference_latency_ms=latency_ms,
        )
