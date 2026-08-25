#!/usr/bin/env python3
"""Compare RF-DETR model sizes on exact RGB frames extracted from a rosbag."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector
from pnu_surgical_tool import (
    DetectionPostprocessor,
    DetectionPostprocessorConfig,
    TemporalClassConfig,
    WorkspaceRoiConfig,
)
from pnu_surgical_tool.types import DetectionBatch
from pnu_surgical_tool.visualization import draw_detections_bgr


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for a non-empty list."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def read_frame_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Index the rosbag extraction CSV by relative frame path."""
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["relative_path"]: row for row in csv.DictReader(stream)}


def filtered_batch(batch: DetectionBatch, threshold: float) -> DetectionBatch:
    """Return a shallow batch filtered at a higher confidence threshold."""
    return DetectionBatch(
        image_width=batch.image_width,
        image_height=batch.image_height,
        model_version=batch.model_version,
        ontology_version=batch.ontology_version,
        instances=[
            item
            for item in batch.instances
            if item.class_confidence >= threshold
        ],
        inference_latency_ms=batch.inference_latency_ms,
    )


def frame_record(
    frame_index: int,
    relative_path: str,
    metadata: dict[str, str],
    batch: DetectionBatch,
) -> dict[str, object]:
    """Build a mask-free JSON record for one model/frame result."""
    return {
        "frame_index": frame_index,
        "relative_path": relative_path,
        "stamp_ns": int(metadata["stamp_ns"]),
        "source_jpeg_sha256": metadata["sha256"],
        "inference_latency_ms": batch.inference_latency_ms,
        "instances": [
            {
                "canonical_class_id": item.canonical_class_id,
                "model_class_index": item.model_class_index,
                "class_name": item.class_name,
                "confidence": item.class_confidence,
                "bbox_xyxy_px": list(item.bbox_xyxy_px),
                "mask_area_px": int(np.count_nonzero(item.mask)),
            }
            for item in batch.instances
        ],
    }


def summarize_records(
    records: list[dict[str, object]], thresholds: list[float]
) -> dict[str, object]:
    """Summarize prediction counts and latency without claiming GT accuracy."""
    latencies = [float(record["inference_latency_ms"]) for record in records]
    by_threshold: dict[str, object] = {}
    for threshold in thresholds:
        class_counts: Counter[str] = Counter()
        confidence_values: list[float] = []
        per_frame_counts: list[int] = []
        detection_frames = 0
        for record in records:
            instances = [
                item
                for item in record["instances"]
                if float(item["confidence"]) >= threshold
            ]
            per_frame_counts.append(len(instances))
            detection_frames += bool(instances)
            class_counts.update(str(item["class_name"]) for item in instances)
            confidence_values.extend(float(item["confidence"]) for item in instances)
        adjacent_count_changes = sum(
            left != right
            for left, right in zip(
                per_frame_counts,
                per_frame_counts[1:],
                strict=False,
            )
        )
        by_threshold[f"{threshold:.2f}"] = {
            "detection_frames": detection_frames,
            "detection_frame_ratio": detection_frames / len(records),
            "total_instances": sum(per_frame_counts),
            "instances_per_frame_mean": statistics.fmean(per_frame_counts),
            "instances_per_frame_min": min(per_frame_counts),
            "instances_per_frame_max": max(per_frame_counts),
            "adjacent_frame_count_changes": adjacent_count_changes,
            "class_counts": dict(sorted(class_counts.items())),
            "confidence_mean": (
                statistics.fmean(confidence_values)
                if confidence_values
                else None
            ),
            "confidence_p50": percentile(confidence_values, 50),
            "confidence_p95": percentile(confidence_values, 95),
        }
    return {
        "frames": len(records),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "fps_from_mean": 1000.0 / statistics.fmean(latencies),
        },
        "thresholds": by_threshold,
    }


def add_overlay_header(
    image: np.ndarray,
    model_size: str,
    frame_index: int,
    instance_count: int,
    threshold: float,
) -> np.ndarray:
    """Add model/frame context above the standard mask overlay."""
    output = np.asarray(image).copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (18, 18, 18), -1)
    label = (
        f"RF-DETR Seg {model_size.title()} | frame {frame_index:03d} | "
        f"threshold {threshold:.2f} | instances {instance_count}"
    )
    cv2.putText(
        output,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def evaluate_model(
    model_size: str,
    checkpoint: Path,
    ontology: Path,
    frame_root: Path,
    frames: list[Path],
    metadata_by_path: dict[str, dict[str, str]],
    output_dir: Path,
    inference_threshold: float,
    overlay_threshold: float,
    summary_thresholds: list[float],
    fps: float,
    optimize: bool,
    warmup_iterations: int,
    postprocessor_config: DetectionPostprocessorConfig | None,
) -> tuple[list[dict[str, object]], dict[str, object], Path]:
    """Run one model over all frames and write JSONL plus an overlay video."""
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=checkpoint,
            ontology_path=ontology,
            model_size=model_size,
            confidence_threshold=inference_threshold,
            optimize=optimize,
        )
    )
    detector.load()
    postprocessor = (
        DetectionPostprocessor(postprocessor_config)
        if postprocessor_config is not None
        else None
    )
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to decode {frames[0]}")
    for _ in range(warmup_iterations):
        detector.predict(first, color_order="BGR")
    height, width = first.shape[:2]
    video_path = output_dir / f"{model_size}_overlay_t{overlay_threshold:.2f}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {video_path}")

    records: list[dict[str, object]] = []
    postprocessing_totals = Counter()
    jsonl_path = output_dir / f"{model_size}_predictions.jsonl"
    started = time.perf_counter()
    try:
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for frame_index, frame_path in enumerate(frames):
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"failed to decode {frame_path}")
                batch = detector.predict(bgr, color_order="BGR")
                if postprocessor is not None:
                    batch = postprocessor.process(batch)
                    for key, value in postprocessor.last_diagnostics.items():
                        if isinstance(value, int) and not isinstance(value, bool):
                            postprocessing_totals[key] += value
                relative_path = frame_path.relative_to(frame_root).as_posix()
                record = frame_record(
                    frame_index,
                    relative_path,
                    metadata_by_path[relative_path],
                    batch,
                )
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

                visible = filtered_batch(batch, overlay_threshold)
                overlay = draw_detections_bgr(bgr, visible)
                if postprocessor is not None:
                    polygon = postprocessor.roi_polygon_pixels(width, height)
                    if polygon is not None:
                        cv2.polylines(
                            overlay,
                            [polygon],
                            isClosed=True,
                            color=(30, 230, 30),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )
                writer.write(
                    add_overlay_header(
                        overlay,
                        model_size,
                        frame_index,
                        len(visible.instances),
                        overlay_threshold,
                    )
                )
                if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(frames):
                    elapsed = time.perf_counter() - started
                    print(
                        f"{model_size}: {frame_index + 1}/{len(frames)} "
                        f"frames, wall {elapsed:.1f}s",
                        flush=True,
                    )
    finally:
        writer.release()

    summary = summarize_records(records, summary_thresholds)
    summary.update(
        {
            "model_size": model_size,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_color_order": detector.checkpoint_color_order,
            "enable_class_agnostic_nms": detector.enable_class_agnostic_nms,
            "inference_threshold": inference_threshold,
            "overlay_threshold": overlay_threshold,
            "optimized": optimize,
            "warmup_iterations": warmup_iterations,
            "wall_duration_sec": time.perf_counter() - started,
            "predictions_jsonl": str(jsonl_path),
            "overlay_video": str(video_path),
            "postprocessing": (
                {
                    "workspace_roi_enabled": (
                        postprocessor_config.workspace_roi.enabled
                    ),
                    "workspace_roi_polygon_norm_xy": list(
                        postprocessor_config.workspace_roi.polygon_norm_xy
                    ),
                    "workspace_roi_minimum_mask_overlap": (
                        postprocessor_config.workspace_roi.minimum_mask_overlap
                    ),
                    "workspace_roi_require_mask_centroid_inside": (
                        postprocessor_config.workspace_roi.require_mask_centroid_inside
                    ),
                    "temporal_class_smoothing_enabled": (
                        postprocessor_config.temporal_class.enabled
                    ),
                    "temporal_class_history_size": (
                        postprocessor_config.temporal_class.history_size
                    ),
                    "temporal_class_minimum_switch_frames": (
                        postprocessor_config.temporal_class.minimum_switch_frames
                    ),
                    "temporal_class_switch_score_margin": (
                        postprocessor_config.temporal_class.switch_score_margin
                    ),
                    "temporal_association_minimum_mask_iou": (
                        postprocessor_config.temporal_class.minimum_mask_iou
                    ),
                    "temporal_association_minimum_bbox_iou": (
                        postprocessor_config.temporal_class.minimum_bbox_iou
                    ),
                    "temporal_association_maximum_centroid_distance_norm": (
                        postprocessor_config.temporal_class.maximum_centroid_distance_norm
                    ),
                    "temporal_association_maximum_mask_area_ratio": (
                        postprocessor_config.temporal_class.maximum_mask_area_ratio
                    ),
                    "temporal_track_max_missed_frames": (
                        postprocessor_config.temporal_class.max_missed_frames
                    ),
                    "totals": dict(postprocessing_totals),
                    "final_active_tracks": postprocessor.active_track_count,
                }
                if postprocessor is not None
                else {"enabled": False}
            ),
        }
    )
    return records, summary, video_path


def compare_model_pair(
    left_name: str,
    left_records: list[dict[str, object]],
    right_name: str,
    right_records: list[dict[str, object]],
    thresholds: list[float],
) -> dict[str, object]:
    """Compare prediction counts between models without treating either as GT."""
    comparison: dict[str, object] = {}
    for threshold in thresholds:
        left_counts = [
            sum(
                float(item["confidence"]) >= threshold
                for item in record["instances"]
            )
            for record in left_records
        ]
        right_counts = [
            sum(
                float(item["confidence"]) >= threshold
                for item in record["instances"]
            )
            for record in right_records
        ]
        differences = [
            right_count - left_count
            for left_count, right_count in zip(
                left_counts, right_counts, strict=True
            )
        ]
        comparison[f"{threshold:.2f}"] = {
            "frames_with_equal_instance_count": sum(
                difference == 0 for difference in differences
            ),
            f"frames_{right_name}_has_more": sum(
                difference > 0 for difference in differences
            ),
            f"frames_{left_name}_has_more": sum(
                difference < 0 for difference in differences
            ),
            f"{right_name}_minus_{left_name}_instances": sum(differences),
            "absolute_count_difference_per_frame_mean": statistics.fmean(
                abs(difference) for difference in differences
            ),
        }
    return comparison


def combine_videos(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    fps: float,
) -> None:
    """Write a side-by-side two-model review video."""
    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not left.isOpened() or not right.isOpened() or not writer.isOpened():
        raise RuntimeError("failed to open comparison video streams")
    try:
        while True:
            left_ok, left_frame = left.read()
            right_ok, right_frame = right.read()
            if not left_ok or not right_ok:
                if left_ok != right_ok:
                    raise RuntimeError("model overlay videos have unequal lengths")
                break
            writer.write(np.hstack([left_frame, right_frame]))
    finally:
        left.release()
        right.release()
        writer.release()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--small-checkpoint", type=Path)
    parser.add_argument("--medium-checkpoint", type=Path)
    parser.add_argument("--large-checkpoint", type=Path)
    parser.add_argument("--xlarge-checkpoint", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("small", "medium", "large", "xlarge"),
        default=["medium", "large"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-threshold", type=float, default=0.2)
    parser.add_argument("--overlay-threshold", type=float, default=0.3)
    parser.add_argument(
        "--summary-thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.5],
    )
    parser.add_argument("--fps", type=float, default=14.990429585087846)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument(
        "--workspace-roi-polygon-norm-xy",
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--workspace-roi-minimum-mask-overlap", type=float, default=0.5
    )
    parser.add_argument(
        "--workspace-roi-require-mask-centroid-inside",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temporal-class-smoothing", action="store_true")
    parser.add_argument("--temporal-class-history-size", type=int, default=7)
    parser.add_argument(
        "--temporal-class-minimum-switch-frames", type=int, default=3
    )
    parser.add_argument(
        "--temporal-class-switch-score-margin", type=float, default=0.2
    )
    parser.add_argument(
        "--temporal-association-minimum-mask-iou", type=float, default=0.10
    )
    parser.add_argument(
        "--temporal-association-minimum-bbox-iou", type=float, default=0.20
    )
    parser.add_argument(
        "--temporal-association-maximum-centroid-distance-norm",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--temporal-association-maximum-mask-area-ratio",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--temporal-track-max-missed-frames", type=int, default=3
    )
    return parser.parse_args()


def main() -> None:
    """Run selected models and write individual plus comparative artifacts."""
    args = parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must not be negative")
    if args.inference_threshold > min(args.summary_thresholds):
        raise ValueError(
            "inference-threshold must not exceed any summary threshold"
        )
    frames = sorted(args.frames_dir.glob("frame_*.jpg"))[:: args.stride]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("no input frames found")
    checkpoint_by_model = {
        "small": args.small_checkpoint,
        "medium": args.medium_checkpoint,
        "large": args.large_checkpoint,
        "xlarge": args.xlarge_checkpoint,
    }
    for model_size in args.models:
        if checkpoint_by_model[model_size] is None:
            raise ValueError(f"--{model_size}-checkpoint is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_root = args.frames_dir.parent
    metadata_by_path = read_frame_metadata(args.frames_csv)
    use_postprocessing = bool(
        args.workspace_roi_polygon_norm_xy
        or args.temporal_class_smoothing
    )
    postprocessor_config = (
        DetectionPostprocessorConfig(
            workspace_roi=WorkspaceRoiConfig(
                enabled=bool(args.workspace_roi_polygon_norm_xy),
                polygon_norm_xy=tuple(
                    args.workspace_roi_polygon_norm_xy or ()
                ),
                minimum_mask_overlap=(
                    args.workspace_roi_minimum_mask_overlap
                ),
                require_mask_centroid_inside=(
                    args.workspace_roi_require_mask_centroid_inside
                ),
            ),
            temporal_class=TemporalClassConfig(
                enabled=args.temporal_class_smoothing,
                history_size=args.temporal_class_history_size,
                minimum_switch_frames=(
                    args.temporal_class_minimum_switch_frames
                ),
                switch_score_margin=(
                    args.temporal_class_switch_score_margin
                ),
                minimum_mask_iou=(
                    args.temporal_association_minimum_mask_iou
                ),
                minimum_bbox_iou=(
                    args.temporal_association_minimum_bbox_iou
                ),
                maximum_centroid_distance_norm=(
                    args.temporal_association_maximum_centroid_distance_norm
                ),
                maximum_mask_area_ratio=(
                    args.temporal_association_maximum_mask_area_ratio
                ),
                max_missed_frames=args.temporal_track_max_missed_frames,
            ),
        )
        if use_postprocessing
        else None
    )

    records_by_model: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, object] = {}
    videos: dict[str, Path] = {}
    for model_size in args.models:
        checkpoint = checkpoint_by_model[model_size]
        records, summary, video = evaluate_model(
            model_size=model_size,
            checkpoint=checkpoint,
            ontology=args.ontology,
            frame_root=frame_root,
            frames=frames,
            metadata_by_path=metadata_by_path,
            output_dir=args.output_dir,
            inference_threshold=args.inference_threshold,
            overlay_threshold=args.overlay_threshold,
            summary_thresholds=args.summary_thresholds,
            fps=args.fps / args.stride,
            optimize=args.optimize,
            warmup_iterations=args.warmup_iterations,
            postprocessor_config=postprocessor_config,
        )
        records_by_model[model_size] = records
        summaries[model_size] = summary
        videos[model_size] = video

    comparisons: dict[str, object] = {}
    comparison_videos: dict[str, str] = {}
    for left_name, right_name in zip(
        args.models, args.models[1:], strict=False
    ):
        pair_name = f"{left_name}_vs_{right_name}"
        comparison_video = (
            args.output_dir
            / f"{pair_name}_overlay_t{args.overlay_threshold:.2f}.mp4"
        )
        combine_videos(
            videos[left_name],
            videos[right_name],
            comparison_video,
            args.fps / args.stride,
        )
        comparisons[pair_name] = compare_model_pair(
            left_name,
            records_by_model[left_name],
            right_name,
            records_by_model[right_name],
            args.summary_thresholds,
        )
        comparison_videos[pair_name] = str(comparison_video)
    payload = {
        "schema": "pnu.cam4.rfdetrseg.scale_rosbag_comparison.v2",
        "interpretation": (
            "unlabeled external-scene prediction comparison; not accuracy, "
            "precision, recall, IoU, or mAP"
        ),
        "source": {
            "frames_dir": str(args.frames_dir),
            "frames_csv": str(args.frames_csv),
            "frame_count": len(frames),
            "stride": args.stride,
            "fps": args.fps / args.stride,
        },
        "models": summaries,
        "comparisons": comparisons,
        "comparison_videos": comparison_videos,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
