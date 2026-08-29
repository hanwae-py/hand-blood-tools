#!/usr/bin/env python3
"""Run standalone detection and write JSON plus an optional overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector
from pnu_surgical_tool.rle import encode_uncompressed_coco_rle
from pnu_surgical_tool.visualization import draw_detections_bgr


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "model/checkpoint_selected_external_0825_holdout_conf030.pth",
    )
    parser.add_argument("--ontology", type=Path, default=ROOT / "model/ontology.json")
    parser.add_argument(
        "--model-size",
        choices=("small", "medium", "large", "xlarge"),
        default="xlarge",
    )
    parser.add_argument("--color-order", choices=("RGB", "BGR"), required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path)
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    supplied = bgr if args.color_order == "BGR" else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=args.checkpoint,
            ontology_path=args.ontology,
            model_size=args.model_size,
            confidence_threshold=args.threshold,
            optimize=not args.no_optimize,
        )
    )
    detections = detector.predict(supplied, color_order=args.color_order)
    payload = {
        "schema": "pnu.surgical_tool.detection.v1",
        "model_version": detections.model_version,
        "ontology_version": detections.ontology_version,
        "image": {"width": detections.image_width, "height": detections.image_height},
        "inference_latency_ms": detections.inference_latency_ms,
        "instances": [
            {
                "frame_local_instance_id": item.frame_local_instance_id,
                "canonical_class_id": item.canonical_class_id,
                "model_class_index": item.model_class_index,
                "class_name": item.class_name,
                "class_confidence": item.class_confidence,
                "bbox_xyxy_px": item.bbox_xyxy_px,
                "mask_rle": encode_uncompressed_coco_rle(item.mask),
            }
            for item in detections.instances
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.output_overlay is not None:
        args.output_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output_overlay), draw_detections_bgr(bgr, detections))
    print(json.dumps({"instances": len(detections.instances), "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
