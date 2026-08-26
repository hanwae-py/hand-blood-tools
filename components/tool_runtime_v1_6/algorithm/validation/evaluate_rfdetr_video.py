#!/usr/bin/env python3
"""Evaluate one RF-DETR checkpoint on contiguous windows from RGB videos."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import yaml

from pnu_surgical_tool import (
    DetectionPostprocessor,
    DetectionPostprocessorConfig,
    DetectorConfig,
    SurgicalToolDetector,
    TemporalClassConfig,
    WorkspaceRoiConfig,
)
from pnu_surgical_tool.types import DetectionBatch
from pnu_surgical_tool.visualization import draw_detections_bgr


DIAGNOSTIC_TOTAL_KEYS = (
    "input_instances",
    "roi_rejected_instances",
    "output_instances",
    "tracks_created",
    "tracks_matched",
    "raw_class_transitions",
    "class_overrides",
    "class_switches",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--camera", choices=("cam3", "cam4"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument(
        "--model-size",
        choices=("small", "medium", "large", "xlarge"),
        required=True,
    )
    parser.add_argument("--roi-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--window-duration-sec", type=float, default=30.0)
    parser.add_argument(
        "--window-centers", type=float, nargs="+", default=(0.25, 0.5, 0.75)
    )
    parser.add_argument("--full-video", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    return parser.parse_args()


def load_roi_profile(path: Path) -> tuple[str, WorkspaceRoiConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = payload["/**"]["ros__parameters"]
    if not bool(parameters["workspace_roi_enabled"]):
        raise ValueError(f"ROI profile is disabled: {path}")
    profile = str(parameters["workspace_roi_profile"])
    return profile, WorkspaceRoiConfig(
        enabled=True,
        polygon_norm_xy=tuple(
            float(value)
            for value in parameters["workspace_roi_polygon_norm_xy"]
        ),
        minimum_mask_overlap=float(
            parameters["workspace_roi_minimum_mask_overlap"]
        ),
        require_mask_centroid_inside=bool(
            parameters["workspace_roi_require_mask_centroid_inside"]
        ),
    )


def make_postprocessor(roi: WorkspaceRoiConfig) -> DetectionPostprocessor:
    return DetectionPostprocessor(
        DetectionPostprocessorConfig(
            workspace_roi=roi,
            temporal_class=TemporalClassConfig(
                enabled=True,
                history_size=7,
                minimum_switch_frames=3,
                switch_score_margin=0.2,
                minimum_mask_iou=0.1,
                minimum_bbox_iou=0.2,
                maximum_centroid_distance_norm=0.06,
                maximum_mask_area_ratio=3.0,
                max_missed_frames=3,
            ),
        )
    )


def selected_windows(
    frame_count: int,
    fps: float,
    centers: list[float],
    duration_sec: float,
    full_video: bool,
) -> list[tuple[int, int]]:
    if full_video:
        return [(0, frame_count)]
    length = max(1, int(round(duration_sec * fps)))
    windows = []
    for center in centers:
        if not 0.0 <= center <= 1.0:
            raise ValueError("window centers must lie in [0, 1]")
        start = int(round(center * max(frame_count - 1, 0) - length / 2))
        start = min(max(0, start), max(0, frame_count - length))
        windows.append((start, min(frame_count, start + length)))
    return sorted(set(windows))


def instance_records(batch: DetectionBatch) -> list[dict[str, object]]:
    return [
        {
            "class_name": item.class_name,
            "canonical_class_id": item.canonical_class_id,
            "confidence": item.class_confidence,
            "bbox_xyxy_px": list(item.bbox_xyxy_px),
            "mask_area_px": int(np.count_nonzero(item.mask)),
        }
        for item in batch.instances
    ]


def draw_header(
    image: np.ndarray,
    camera: str,
    profile: str,
    source_frame: int,
    source_fps: float,
    raw_count: int,
    output_count: int,
    threshold: float,
) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 56), (18, 18, 18), -1)
    cv2.putText(
        output,
        (
            f"{camera.upper()} | t={source_frame / source_fps:.1f}s "
            f"frame={source_frame} | threshold={threshold:.2f} | "
            f"raw={raw_count} post={output_count}"
        ),
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"ROI profile: {profile}",
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (40, 235, 40),
        2,
        cv2.LINE_AA,
    )
    return output


def evaluate_video(
    path: Path,
    camera: str,
    detector: SurgicalToolDetector,
    postprocessor: DetectionPostprocessor,
    profile: str,
    output_dir: Path,
    threshold: float,
    centers: list[float],
    duration_sec: float,
    full_video: bool,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    windows = selected_windows(frame_count, fps, centers, duration_sec, full_video)
    stem = f"{path.parent.name}_{camera}"
    overlay_path = output_dir / f"{stem}_post_t{threshold:.2f}.mp4"
    jsonl_path = output_dir / f"{stem}_predictions.jsonl"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create video: {overlay_path}")

    raw_classes: Counter[str] = Counter()
    output_classes: Counter[str] = Counter()
    diagnostic_totals: Counter[str] = Counter()
    inference_latencies: list[float] = []
    pipeline_latencies: list[float] = []
    raw_detection_frames = 0
    output_detection_frames = 0
    processed = 0
    started = time.perf_counter()
    polygon = postprocessor.roi_polygon_pixels(width, height)
    try:
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for window_index, (start, end) in enumerate(windows):
                postprocessor.reset()
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                for source_frame in range(start, end):
                    ok, bgr = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"decode failed: {path} frame {source_frame}"
                        )
                    frame_started = time.perf_counter()
                    raw = detector.predict(bgr, "BGR", threshold)
                    output = postprocessor.process(raw)
                    pipeline_latencies.append(
                        (time.perf_counter() - frame_started) * 1000.0
                    )
                    inference_latencies.append(raw.inference_latency_ms)
                    raw_detection_frames += bool(raw.instances)
                    output_detection_frames += bool(output.instances)
                    raw_classes.update(item.class_name for item in raw.instances)
                    output_classes.update(item.class_name for item in output.instances)
                    diagnostics = dict(postprocessor.last_diagnostics)
                    diagnostic_totals.update(
                        {
                            key: int(diagnostics.get(key, 0))
                            for key in DIAGNOSTIC_TOTAL_KEYS
                        }
                    )
                    overlay = draw_detections_bgr(bgr, output)
                    if polygon is not None:
                        cv2.polylines(
                            overlay,
                            [polygon],
                            True,
                            (30, 235, 30),
                            3,
                            cv2.LINE_AA,
                        )
                    overlay = draw_header(
                        overlay,
                        camera,
                        profile,
                        source_frame,
                        fps,
                        len(raw.instances),
                        len(output.instances),
                        threshold,
                    )
                    writer.write(overlay)
                    stream.write(
                        json.dumps(
                            {
                                "window_index": window_index,
                                "source_frame": source_frame,
                                "source_time_sec": source_frame / fps,
                                "raw_instances": instance_records(raw),
                                "output_instances": instance_records(output),
                                "postprocessing": diagnostics,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    processed += 1
                    if processed % 100 == 0:
                        print(
                            f"{stem}: {processed} selected frames, "
                            f"wall {time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
    finally:
        capture.release()
        writer.release()

    return {
        "video": str(path),
        "camera": camera,
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": frame_count / fps,
        },
        "selection": {
            "full_video": full_video,
            "window_duration_sec": duration_sec,
            "window_centers": centers,
            "windows_frame_start_end_exclusive": [list(item) for item in windows],
            "selected_frames": processed,
            "temporal_state_reset_at_each_window": True,
        },
        "raw": {
            "detection_frames": raw_detection_frames,
            "instances": int(diagnostic_totals["input_instances"]),
            "class_counts": dict(sorted(raw_classes.items())),
        },
        "postprocessed": {
            "detection_frames": output_detection_frames,
            "instances": int(diagnostic_totals["output_instances"]),
            "class_counts": dict(sorted(output_classes.items())),
            "diagnostic_totals": dict(diagnostic_totals),
        },
        "latency_ms": {
            "inference_mean": statistics.fmean(inference_latencies),
            "inference_p95": float(np.percentile(inference_latencies, 95)),
            "pipeline_mean": statistics.fmean(pipeline_latencies),
            "pipeline_p95": float(np.percentile(pipeline_latencies, 95)),
        },
        "artifacts": {
            "overlay_video": str(overlay_path),
            "predictions_jsonl": str(jsonl_path),
        },
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if args.window_duration_sec <= 0.0:
        raise ValueError("window-duration-sec must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must not be negative")
    for path in (*args.videos, args.checkpoint, args.ontology, args.roi_profile):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile, roi = load_roi_profile(args.roi_profile)
    if not profile.startswith(f"{args.camera}_"):
        raise ValueError(f"ROI profile {profile!r} does not match {args.camera}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=args.checkpoint,
            ontology_path=args.ontology,
            model_size=args.model_size,
            confidence_threshold=args.threshold,
            optimize=args.optimize,
        )
    )
    detector.load()
    warmup_capture = cv2.VideoCapture(str(args.videos[0]))
    ok, warmup_frame = warmup_capture.read()
    warmup_capture.release()
    if not ok:
        raise RuntimeError(f"failed to decode warmup frame: {args.videos[0]}")
    for _ in range(args.warmup_iterations):
        detector.predict(warmup_frame, "BGR", args.threshold)

    results = []
    for video in args.videos:
        results.append(
            evaluate_video(
                video,
                args.camera,
                detector,
                make_postprocessor(roi),
                profile,
                args.output_dir,
                args.threshold,
                list(args.window_centers),
                args.window_duration_sec,
                args.full_video,
            )
        )
    payload = {
        "schema": "pnu.surgical_tool.rgb_video_postprocessing_evaluation.v1",
        "interpretation": (
            "unlabeled prediction and temporal-consistency comparison; not "
            "accuracy, precision, recall, IoU, or mAP"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_size": args.model_size,
        "checkpoint_color_order": detector.checkpoint_color_order,
        "confidence_threshold": args.threshold,
        "roi_profile": profile,
        "roi_polygon_norm_xy": list(roi.polygon_norm_xy),
        "videos": results,
    }
    summary_path = args.output_dir / f"{args.camera}_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
