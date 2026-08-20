#!/usr/bin/env python3
"""Optional model-load/detection acceptance on a caller-approved image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int)
    parser.add_argument("--color-order", choices=("RGB", "BGR"), default="BGR")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)
    supplied = bgr if args.color_order == "BGR" else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detector = SurgicalToolDetector(
        DetectorConfig(
            ROOT / "model/cam4_rfdetr_seg_small_regular_resume_e13_best.pth",
            ROOT / "model/ontology.json",
            optimize=not args.no_optimize,
        )
    )
    result = detector.predict(supplied, args.color_order)
    if args.expected_instances is not None:
        assert len(result.instances) == args.expected_instances, (
            len(result.instances),
            args.expected_instances,
        )
    assert all(item.mask.shape == bgr.shape[:2] for item in result.instances)
    assert all(1 <= item.canonical_class_id <= 8 for item in result.instances)
    print(f"PASS: model loaded; {len(result.instances)} instances; {result.inference_latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
