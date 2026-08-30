#!/usr/bin/env python3
"""Offline fused RF-DETR Seg-Small + Cutie blood segmentation.

One explicit class: ``blood``. Background is implicit. The live overlay is the
fused mask, not detector-only instances.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


COMPONENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = COMPONENT_ROOT / "pretrained" / "blood_detection_full_all.pth"
DEFAULT_CUTIE = COMPONENT_ROOT / "pretrained" / "cutie_blood_full_all.pth"
DEFAULT_IMAGES = Path.home() / "data" / "blood" / "imgs"


def encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=np.uint8)
    flat = binary.reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in flat:
        current = int(pixel != 0)
        if current == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = current
    counts.append(run_length)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def centroid(mask: np.ndarray) -> list[float] | None:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]


def draw_fused_overlay(image_bgr: np.ndarray, mask: np.ndarray, regions: list[dict]) -> np.ndarray:
    output = image_bgr.copy()
    if mask.any():
        layer = output.copy()
        layer[mask] = (230, 80, 30)
        output = cv2.addWeighted(output, 0.70, layer, 0.30, 0.0)
    for region in regions:
        cx, cy = region["centroid_xy"]
        cv2.circle(output, (int(round(cx)), int(round(cy))), 5, (230, 80, 30), -1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cutie-checkpoint", type=Path, default=DEFAULT_CUTIE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--redetect-interval", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all images")
    parser.add_argument("--fps", type=float, default=15.0, help="FPS used when writing MP4 videos")
    parser.add_argument("--no-video", action="store_true", help="Do not create overlay.mp4 and mask.mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.redetect_interval < 1:
        raise ValueError("--redetect-interval must be >= 1")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.cutie_checkpoint.is_file():
        raise FileNotFoundError(args.cutie_checkpoint)

    image_paths = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise RuntimeError("No PNG/JPEG images found")

    from blood.pipeline.runner import BloodPipeline, PipelineConfig

    print("Loading fused RF-DETR Seg-Small + Cutie Blood pipeline...")
    pipe = BloodPipeline(
        args.checkpoint,
        args.cutie_checkpoint,
        PipelineConfig(redetect_interval=args.redetect_interval, score_thr=args.threshold),
    )

    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "blood_results.jsonl"
    first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Cannot read {image_paths[0]}")
    frame_height, frame_width = first_frame.shape[:2]
    overlay_video = None
    mask_video = None
    if not args.no_video:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_video = cv2.VideoWriter(
            str(args.output_dir / "overlay.mp4"), codec, args.fps, (frame_width, frame_height)
        )
        mask_video = cv2.VideoWriter(
            str(args.output_dir / "mask.mp4"), codec, args.fps, (frame_width, frame_height), isColor=False
        )
        if not overlay_video.isOpened() or not mask_video.isOpened():
            raise RuntimeError("Cannot open MP4 output writer")

    with results_path.open("w", encoding="utf-8") as results_file:
        for sequence, image_path in enumerate(image_paths):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Cannot read {image_path}")
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            result = pipe.step(rgb)
            latency_ms = (time.perf_counter() - started) * 1000.0
            height, width = image_bgr.shape[:2]
            union_mask = np.asarray(result.mask, dtype=bool)
            overlay = draw_fused_overlay(image_bgr, union_mask, result.centroids)
            instances = []
            for item_id, region in enumerate(result.centroids):
                x, y, w, h = (int(value) for value in region["bbox_xywh"])
                instances.append(
                    {
                        "instance_id": item_id,
                        "class_id": 1,
                        "class_name": "blood",
                        "bbox_xyxy_px": [float(x), float(y), float(x + w), float(y + h)],
                        "centroid_xy_px": [float(value) for value in region["centroid_xy"]],
                        "area": int(region["area"]),
                    }
                )
            cv2.imwrite(str(masks_dir / f"{image_path.stem}_blood_mask.png"), (union_mask * 255).astype(np.uint8))
            cv2.imwrite(str(overlays_dir / f"{image_path.stem}_overlay.jpg"), overlay)
            if overlay_video is not None and mask_video is not None:
                overlay_video.write(overlay)
                mask_video.write((union_mask * 255).astype(np.uint8))
            payload = {
                "schema": "pnu.surgical_blood_observations.v1",
                "sequence": sequence,
                "source_image": str(image_path),
                "image": {"width": width, "height": height},
                "model": "RF-DETR Seg-Small + Cutie",
                "checkpoint": str(args.checkpoint),
                "cutie_checkpoint": str(args.cutie_checkpoint),
                "classes": ["blood"],
                "confidence_threshold": args.threshold,
                "fusion_action": result.action,
                "ran_detector": result.ran_detector,
                "inference_latency_ms": latency_ms,
                "instances": instances,
                "combined_blood_mask_rle": encode_coco_rle(union_mask),
                "combined_blood_centroid_xy_px": centroid(union_mask),
            }
            results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                f"[{sequence + 1}/{len(image_paths)}] {image_path.name}: "
                f"{len(instances)} regions, {result.action}, {latency_ms:.1f} ms"
            )

    if overlay_video is not None:
        overlay_video.release()
    if mask_video is not None:
        mask_video.release()
    print(f"Saved masks, overlays, and JSONL results to: {args.output_dir}")


if __name__ == "__main__":
    main()
