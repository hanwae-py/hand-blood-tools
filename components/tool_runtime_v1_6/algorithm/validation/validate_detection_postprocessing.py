#!/usr/bin/env python3
"""Clinical-data-free class-agnostic NMS contract test."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from pnu_surgical_tool import (
    class_agnostic_nms_indices,
    DetectorConfig,
    SurgicalToolDetector,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeModel:
    def predict(self, image, threshold, include_source_image):
        del image, threshold, include_source_image
        masks = np.zeros((2, 12, 12), dtype=bool)
        masks[:, 2:10, 2:10] = True
        return SimpleNamespace(
            xyxy=np.asarray([[2, 2, 10, 10], [2, 2, 10, 10]], dtype=float),
            class_id=np.asarray([0, 1], dtype=int),
            confidence=np.asarray([0.9, 0.8], dtype=float),
            mask=masks,
        )


def main() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [1.0, 1.0, 9.0, 9.0],
        ]
    )
    confidences = np.asarray([0.90, 0.80, 0.70, 0.85])
    keep = class_agnostic_nms_indices(boxes, confidences, 0.8)
    assert keep.tolist() == [0, 3, 2]

    empty = class_agnostic_nms_indices(
        np.empty((0, 4)), np.empty(0), 0.8
    )
    assert empty.shape == (0,)
    defaults = DetectorConfig("checkpoint", "ontology")
    assert defaults.confidence_threshold == 0.3
    assert defaults.enable_class_agnostic_nms is True
    assert defaults.class_agnostic_nms_iou == 0.8
    disabled = DetectorConfig(
        "checkpoint", "ontology", enable_class_agnostic_nms=False
    )
    assert disabled.enable_class_agnostic_nms is False

    sys.modules.setdefault(
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    observed_counts = []
    for enabled in (True, False):
        detector = SurgicalToolDetector(
            DetectorConfig(
                checkpoint_path="unused",
                ontology_path=ROOT / "model/ontology.json",
                enable_class_agnostic_nms=enabled,
                optimize=False,
            )
        )
        detector._model = _FakeModel()
        observed_counts.append(len(detector.predict(image, "BGR").instances))
    assert observed_counts == [1, 2]
    print("PASS: confidence-ordered class-agnostic bbox NMS IoU 0.8")


if __name__ == "__main__":
    main()
