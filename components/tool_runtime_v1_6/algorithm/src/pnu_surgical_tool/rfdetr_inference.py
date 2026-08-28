"""Fine-tuned RF-DETR segmentation adapter with an explicit color contract."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .types import DetectionBatch, DetectionInstance


ModelSize = Literal["small", "medium", "large", "xlarge"]
ColorOrder = Literal["RGB", "BGR"]


@dataclass(frozen=True)
class _ModelRuntimeSpec:
    class_name: str
    checkpoint_color_order: ColorOrder
    enable_class_agnostic_nms: bool
    model_version: str


_MODEL_RUNTIME_SPECS: dict[ModelSize, _ModelRuntimeSpec] = {
    "small": _ModelRuntimeSpec(
        class_name="RFDETRSegSmall",
        checkpoint_color_order="BGR",
        enable_class_agnostic_nms=True,
        model_version="cam4-rfdetr-seg-small-regular-resume-best",
    ),
    "medium": _ModelRuntimeSpec(
        class_name="RFDETRSegMedium",
        checkpoint_color_order="RGB",
        enable_class_agnostic_nms=False,
        model_version="cam4-rfdetr-seg-medium-20260825-best",
    ),
    "large": _ModelRuntimeSpec(
        class_name="RFDETRSegLarge",
        checkpoint_color_order="RGB",
        enable_class_agnostic_nms=False,
        model_version="cam4-rfdetr-seg-large-20260825-best",
    ),
    "xlarge": _ModelRuntimeSpec(
        class_name="RFDETRSegXLarge",
        checkpoint_color_order="RGB",
        enable_class_agnostic_nms=False,
        model_version="rfdetr-seg-xlarge-selected-external-0825-conf030",
    ),
}


@dataclass(frozen=True)
class DetectorConfig:
    checkpoint_path: str | Path
    ontology_path: str | Path
    confidence_threshold: float = 0.3
    class_confidence_thresholds: Mapping[str, float] | None = None
    enable_class_agnostic_nms: bool | None = None
    class_agnostic_nms_iou: float | None = 0.8
    optimize: bool = True
    jit_compile: bool = True
    fp16: bool = True
    model_size: ModelSize = "small"
    model_version: str | None = None
    checkpoint_color_order: ColorOrder | None = None


class SurgicalToolDetector:
    """Load the supplied checkpoint and return class/bbox/instance-mask results.

    Input arrays may be RGB or BGR. They are normalized to the selected model's
    validated contract: legacy Small uses BGR; larger variants use RGB.
    Loading is intentionally lazy so pose-only users do not need RF-DETR.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model = None
        if config.model_size not in _MODEL_RUNTIME_SPECS:
            supported = ", ".join(_MODEL_RUNTIME_SPECS)
            raise ValueError(
                f"model_size must be one of {supported}; got {config.model_size!r}"
            )
        self._runtime_spec = _MODEL_RUNTIME_SPECS[config.model_size]
        self.model_version = config.model_version or self._runtime_spec.model_version
        self.checkpoint_color_order = (
            config.checkpoint_color_order
            or self._runtime_spec.checkpoint_color_order
        )
        self.enable_class_agnostic_nms = (
            self._runtime_spec.enable_class_agnostic_nms
            if config.enable_class_agnostic_nms is None
            else config.enable_class_agnostic_nms
        )
        payload = json.loads(Path(config.ontology_path).read_text(encoding="utf-8"))
        classes = payload["canonical_tool_classes"]
        self._classes = sorted(classes, key=lambda item: int(item["canonical_id"]))
        self.ontology_version = str(payload.get("schema", "unknown"))
        if [int(item["canonical_id"]) for item in self._classes] != list(range(1, 9)):
            raise ValueError("Expected frozen canonical IDs 1..8")
        if not 0.0 <= config.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        known_class_names = {
            str(item["canonical_name"]) for item in self._classes
        }
        self.class_confidence_thresholds = {
            str(class_name): float(class_threshold)
            for class_name, class_threshold in (
                config.class_confidence_thresholds or {}
            ).items()
        }
        unknown_class_names = (
            self.class_confidence_thresholds.keys() - known_class_names
        )
        if unknown_class_names:
            names = ", ".join(sorted(unknown_class_names))
            raise ValueError(f"Unknown class threshold: {names}")
        for class_name, class_threshold in self.class_confidence_thresholds.items():
            if not 0.0 <= class_threshold <= 1.0:
                raise ValueError(
                    f"class threshold for {class_name!r} must be in [0, 1]"
                )
        if config.class_agnostic_nms_iou is not None and not (
            0.0 <= config.class_agnostic_nms_iou <= 1.0
        ):
            raise ValueError("class_agnostic_nms_iou must be in [0, 1]")
        if self.checkpoint_color_order not in ("RGB", "BGR"):
            raise ValueError("checkpoint_color_order must be 'RGB' or 'BGR'")

    def load(self) -> None:
        if self._model is not None:
            return
        checkpoint = Path(self.config.checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        import torch
        import rfdetr

        model_class = getattr(rfdetr, self._runtime_spec.class_name)
        model = model_class.from_checkpoint(str(checkpoint))
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
        color_order: ColorOrder,
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
        candidate_threshold = min(
            (threshold, *self.class_confidence_thresholds.values())
        )
        self.load()
        if color_order == self.checkpoint_color_order:
            model_image = frame
        elif color_order == "BGR":
            model_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            model_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        detections = self._model.predict(
            model_image,
            threshold=candidate_threshold,
            include_source_image=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0

        height, width = frame.shape[:2]
        masks = getattr(detections, "mask", None)
        if masks is None:
            raise RuntimeError("Checkpoint did not return segmentation masks")
        boxes = np.asarray(detections.xyxy)
        model_indices = np.asarray(detections.class_id, dtype=int)
        confidences = np.asarray(detections.confidence, dtype=float)
        detection_thresholds = np.empty(len(model_indices), dtype=float)
        for index, model_index in enumerate(model_indices):
            if model_index < 0 or model_index >= len(self._classes):
                raise RuntimeError(f"Unexpected model class index: {model_index}")
            class_name = str(self._classes[model_index]["canonical_name"])
            detection_thresholds[index] = self.class_confidence_thresholds.get(
                class_name,
                threshold,
            )
        threshold_keep_indices = np.flatnonzero(
            confidences > detection_thresholds
        )
        keep_indices = (
            threshold_keep_indices[
                class_agnostic_nms_indices(
                    boxes[threshold_keep_indices],
                    confidences[threshold_keep_indices],
                    self.config.class_agnostic_nms_iou,
                )
            ]
            if self.enable_class_agnostic_nms
            and self.config.class_agnostic_nms_iou is not None
            else threshold_keep_indices
        )
        instances: list[DetectionInstance] = []
        for frame_local_id, index in enumerate(keep_indices):
            box = boxes[index]
            model_index = model_indices[index]
            confidence = confidences[index]
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
                    frame_local_instance_id=frame_local_id,
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
            model_version=self.model_version,
            ontology_version=self.ontology_version,
            instances=instances,
            inference_latency_ms=latency_ms,
        )


def class_agnostic_nms_indices(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return confidence-ordered indices after class-agnostic bbox NMS.

    Class IDs are intentionally absent from this function. Detections of
    different classes suppress one another when their bbox IoU is greater than
    ``iou_threshold``; the higher-confidence candidate is retained.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("boxes_xyxy must be Nx4")
    if len(boxes) != len(scores):
        raise ValueError("boxes and confidences must have equal length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if not len(boxes):
        return np.empty(0, dtype=int)

    x0, y0, x1, y1 = boxes.T
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        inter_x0 = np.maximum(x0[current], x0[remaining])
        inter_y0 = np.maximum(y0[current], y0[remaining])
        inter_x1 = np.minimum(x1[current], x1[remaining])
        inter_y1 = np.minimum(y1[current], y1[remaining])
        intersections = np.maximum(0.0, inter_x1 - inter_x0) * np.maximum(
            0.0, inter_y1 - inter_y0
        )
        unions = areas[current] + areas[remaining] - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=unions > 0,
        )
        order = remaining[ious <= iou_threshold]
    return np.asarray(keep, dtype=int)
