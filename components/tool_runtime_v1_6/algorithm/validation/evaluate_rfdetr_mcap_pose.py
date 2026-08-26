#!/usr/bin/env python3
"""Evaluate RF-DETR segmentation and constrained pose on a native-depth MCAP."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import struct
import time

import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import CameraInfo, CompressedImage

from pnu_surgical_tool import (
    CameraCalibration,
    decode_compressed_depth_16uc1,
    DepthToColorRegistrar,
    DetectionPostprocessor,
    DetectionPostprocessorConfig,
    DetectorConfig,
    PlanarPoseEstimator,
    RigidTransform,
    SurgicalToolAlgorithm,
    SurgicalToolDetector,
    SupportPlane,
    TemporalClassConfig,
    WorkspaceRoiConfig,
)
from pnu_surgical_tool.types import DetectionBatch, result_to_dict
from pnu_surgical_tool.visualization import draw_detections_bgr, draw_pose_axes_bgr


DIAGNOSTIC_KEYS = (
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
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--camera", default="cam_4")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument(
        "--model-size",
        choices=("small", "medium", "large", "xlarge"),
        required=True,
    )
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--roi-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--maximum-delta-ns", type=int, default=1_000_000)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def stamp_ns(message: CompressedImage) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def calibration(message: CameraInfo, version: str) -> CameraCalibration:
    return CameraCalibration(
        width=int(message.width),
        height=int(message.height),
        k=np.asarray(message.k, dtype=np.float64).reshape(3, 3),
        distortion=np.asarray(message.d, dtype=np.float64),
        frame_name=str(message.header.frame_id),
        calibration_version=version,
    )


def parameters(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["/**"]["ros__parameters"]


def make_postprocessor(roi_parameters: dict[str, object]) -> DetectionPostprocessor:
    if not bool(roi_parameters["workspace_roi_enabled"]):
        raise ValueError("ROI profile must be enabled")
    return DetectionPostprocessor(
        DetectionPostprocessorConfig(
            workspace_roi=WorkspaceRoiConfig(
                enabled=True,
                polygon_norm_xy=tuple(
                    float(value)
                    for value in roi_parameters["workspace_roi_polygon_norm_xy"]
                ),
                minimum_mask_overlap=float(
                    roi_parameters["workspace_roi_minimum_mask_overlap"]
                ),
                require_mask_centroid_inside=bool(
                    roi_parameters["workspace_roi_require_mask_centroid_inside"]
                ),
            ),
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


def read_calibration_inputs(
    bag: Path, camera: str
) -> tuple[CameraInfo, CameraInfo, tuple[float, ...]]:
    prefix = f"/synced/{camera}"
    color_info_topic = f"{prefix}/color/camera_info"
    depth_info_topic = f"{prefix}/depth/camera_info"
    extrinsics_topic = f"{prefix}/extrinsics/depth_to_color"
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag), storage_id="mcap"),
        ConverterOptions("", ""),
    )
    color_info = None
    depth_info = None
    extrinsics = None
    while reader.has_next() and (
        color_info is None or depth_info is None or extrinsics is None
    ):
        topic, serialized, _ = reader.read_next()
        if topic == color_info_topic and color_info is None:
            color_info = deserialize_message(serialized, CameraInfo)
        elif topic == depth_info_topic and depth_info is None:
            depth_info = deserialize_message(serialized, CameraInfo)
        elif topic == extrinsics_topic and extrinsics is None:
            if len(serialized) != 100:
                raise ValueError(
                    f"unexpected RealSense Extrinsics CDR size: {len(serialized)}"
                )
            extrinsics = struct.unpack_from("<12d", serialized, 4)
    if color_info is None or depth_info is None or extrinsics is None:
        raise RuntimeError("bag is missing camera info or depth-to-color extrinsics")
    return color_info, depth_info, extrinsics


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def add_header(
    image: np.ndarray,
    label: str,
    frame_index: int,
    result_count: int,
    valid_count: int,
    threshold: float,
) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 56), (18, 18, 18), -1)
    cv2.putText(
        output,
        (
            f"{label} | frame={frame_index} threshold={threshold:.2f} | "
            f"instances={result_count} valid_pose={valid_count}"
        ),
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "pose=PLANAR_4DOF_WITH_NORMAL_PRIOR (not unconstrained 6D)",
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (60, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    for path in (
        args.bag,
        args.checkpoint,
        args.ontology,
        args.pose_config,
        args.roi_profile,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if args.maximum_delta_ns < 0 or args.warmup_iterations < 0:
        raise ValueError("delta and warmup values must not be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pose_parameters = parameters(args.pose_config)
    roi_parameters = parameters(args.roi_profile)
    color_info_message, depth_info_message, extrinsics = read_calibration_inputs(
        args.bag, args.camera
    )
    calibration_version = str(pose_parameters["calibration_version"])
    color_camera = calibration(
        color_info_message, f"{calibration_version}:color"
    )
    depth_camera = calibration(
        depth_info_message, f"{calibration_version}:depth"
    )
    # realsense2_camera_msgs/Extrinsics.rotation is explicitly column-major.
    rotation = np.asarray(extrinsics[:9], dtype=np.float64).reshape(3, 3, order="F")
    translation = np.asarray(extrinsics[9:], dtype=np.float64)
    transform = RigidTransform(
        rotation=rotation,
        translation_m=translation,
        source_frame=depth_camera.frame_name,
        target_frame=color_camera.frame_name,
        calibration_version=f"{calibration_version}:depth_to_color_topic",
    )
    registrar = DepthToColorRegistrar(depth_camera, color_camera, transform)
    support_plane = SupportPlane(
        normal=np.asarray(pose_parameters["support_plane_normal"], dtype=np.float64),
        offset_m=float(pose_parameters["support_plane_offset_m"]),
        config_version=str(pose_parameters["support_plane_config_version"]),
        inlier_ratio=float(pose_parameters["support_plane_inlier_ratio"]),
        residual_p95_m=float(pose_parameters["support_plane_residual_p95_m"]),
    )
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
    postprocessor = make_postprocessor(roi_parameters)
    algorithm = SurgicalToolAlgorithm(
        detector=detector,
        pose_estimator=PlanarPoseEstimator(),
        postprocessor=postprocessor,
    )

    prefix = f"/synced/{args.camera}"
    color_topic = f"{prefix}/color/image_raw/compressed"
    depth_topic = f"{prefix}/depth/image_rect_raw/compressedDepth"
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id="mcap"),
        ConverterOptions("", ""),
    )
    pending_color: dict[int, CompressedImage] = {}
    pending_depth: dict[int, CompressedImage] = {}
    overlay_path = (
        args.output_dir
        / f"{args.label}_pose_overlay_t{args.threshold:.2f}.mp4"
    )
    jsonl_path = args.output_dir / f"{args.label}_pose_predictions.jsonl"
    writer = None
    pair_deltas_ns: list[int] = []
    registration_ms: list[float] = []
    detection_pose_ms: list[float] = []
    pipeline_ms: list[float] = []
    validity_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    valid_pose_by_class: Counter[str] = Counter()
    diagnostic_totals: Counter[str] = Counter()
    registration_valid_ratios: list[float] = []
    paired = 0
    first_bgr = None
    started = time.perf_counter()

    def process_pair(
        color_message: CompressedImage,
        depth_message: CompressedImage,
        stream: object,
    ) -> None:
        nonlocal paired, writer, first_bgr
        bgr = cv2.imdecode(
            np.frombuffer(color_message.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None:
            raise ValueError("OpenCV could not decode an RGB frame")
        native_depth = decode_compressed_depth_16uc1(
            depth_message.data, depth_message.format
        )
        if first_bgr is None:
            first_bgr = bgr
            for _ in range(args.warmup_iterations):
                detector.predict(first_bgr, "BGR", args.threshold)
        frame_started = time.perf_counter()
        registration_started = time.perf_counter()
        registration = registrar.register(
            native_depth,
            args.depth_scale,
            minimum_depth_m=float(pose_parameters["minimum_depth_m"]),
            maximum_depth_m=float(pose_parameters["maximum_depth_m"]),
        )
        registration_ms.append((time.perf_counter() - registration_started) * 1000.0)
        pose_started = time.perf_counter()
        result = algorithm.detect_and_estimate(
            image=bgr,
            aligned_depth_m=registration.aligned_depth_m,
            camera=color_camera,
            support_plane=support_plane,
            color_order="BGR",
            frame_key=paired,
            confidence_threshold=args.threshold,
        )
        detection_pose_ms.append((time.perf_counter() - pose_started) * 1000.0)
        pipeline_ms.append((time.perf_counter() - frame_started) * 1000.0)
        registration_valid_ratios.append(float(registration.aligned_valid_ratio))
        diagnostics = dict(postprocessor.last_diagnostics)
        diagnostic_totals.update(
            {key: int(diagnostics.get(key, 0)) for key in DIAGNOSTIC_KEYS}
        )
        valid_count = 0
        for item in result.instances:
            validity_counts[item.validity] += 1
            class_counts[item.class_name] += 1
            if item.position_valid and item.orientation_valid:
                valid_count += 1
                valid_pose_by_class[item.class_name] += 1
            if item.invalid_reason:
                invalid_reasons[item.invalid_reason] += 1
        detections = DetectionBatch(
            image_width=color_camera.width,
            image_height=color_camera.height,
            model_version=result.model_version,
            ontology_version=result.ontology_version,
            instances=result.instances,
        )
        overlay = draw_detections_bgr(bgr, detections)
        polygon = postprocessor.roi_polygon_pixels(
            color_camera.width, color_camera.height
        )
        if polygon is not None:
            cv2.polylines(overlay, [polygon], True, (30, 235, 30), 2, cv2.LINE_AA)
        overlay = draw_pose_axes_bgr(
            overlay, result, color_camera, axis_length_m=args.axis_length_m
        )
        overlay = add_header(
            overlay,
            args.label,
            paired,
            len(result.instances),
            valid_count,
            args.threshold,
        )
        if writer is None:
            writer = cv2.VideoWriter(
                str(overlay_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (overlay.shape[1], overlay.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to create video: {overlay_path}")
        writer.write(overlay)
        record = result_to_dict(result, include_masks=False)
        record.update(
            {
                "color_stamp_ns": stamp_ns(color_message),
                "depth_stamp_ns": stamp_ns(depth_message),
                "pair_delta_ns": abs(stamp_ns(color_message) - stamp_ns(depth_message)),
                "depth_registration": {
                    "aligned_valid_ratio": registration.aligned_valid_ratio,
                    "source_valid_pixels": registration.source_valid_pixels,
                    "projected_points": registration.projected_points,
                    "aligned_valid_pixels": registration.aligned_valid_pixels,
                    "z_buffer_collisions": registration.z_buffer_collisions,
                },
                "postprocessing": diagnostics,
            }
        )
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        paired += 1
        if paired % 25 == 0:
            print(
                f"{args.label}: {paired} pairs, wall "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )

    def try_match(
        stamp: int,
        own: dict[int, CompressedImage],
        other: dict[int, CompressedImage],
        own_is_color: bool,
        stream: object,
    ) -> None:
        if not other:
            return
        nearest = min(other, key=lambda candidate: abs(candidate - stamp))
        delta = abs(nearest - stamp)
        if delta > args.maximum_delta_ns:
            return
        own_message = own.pop(stamp)
        other_message = other.pop(nearest)
        pair_deltas_ns.append(delta)
        if own_is_color:
            process_pair(own_message, other_message, stream)
        else:
            process_pair(other_message, own_message, stream)

    try:
        with jsonl_path.open("w", encoding="utf-8") as stream:
            while reader.has_next():
                if args.max_pairs is not None and paired >= args.max_pairs:
                    break
                topic, serialized, _ = reader.read_next()
                if topic == color_topic:
                    message = deserialize_message(serialized, CompressedImage)
                    stamp = stamp_ns(message)
                    pending_color[stamp] = message
                    try_match(stamp, pending_color, pending_depth, True, stream)
                elif topic == depth_topic:
                    message = deserialize_message(serialized, CompressedImage)
                    stamp = stamp_ns(message)
                    pending_depth[stamp] = message
                    try_match(stamp, pending_depth, pending_color, False, stream)
    finally:
        if writer is not None:
            writer.release()
    if paired == 0:
        raise RuntimeError("no synchronized RGB-depth pairs were processed")

    summary = {
        "schema": "pnu.surgical_tool.mcap_constrained_pose_evaluation.v1",
        "interpretation": (
            "unlabeled external-scene output/validity evaluation; not pose accuracy"
        ),
        "pose_contract": "PLANAR_4DOF_WITH_NORMAL_PRIOR",
        "not_unconstrained_6d": True,
        "bag": str(args.bag),
        "camera": args.camera,
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_size": args.model_size,
        "checkpoint_color_order": detector.checkpoint_color_order,
        "confidence_threshold": args.threshold,
        "roi_profile": str(roi_parameters["workspace_roi_profile"]),
        "support_plane": {
            "normal": support_plane.normal.tolist(),
            "offset_m": support_plane.offset_m,
            "config_version": support_plane.config_version,
            "provisional": "provisional" in support_plane.config_version.lower(),
        },
        "depth": {
            "scale_m_per_unit": args.depth_scale,
            "scale_verified": bool(pose_parameters["depth_scale_verified"]),
            "pair_count": paired,
            "pair_delta_ns_max": max(pair_deltas_ns),
            "unmatched_color_messages": len(pending_color),
            "unmatched_depth_messages": len(pending_depth),
            "aligned_valid_ratio_mean": statistics.fmean(registration_valid_ratios),
        },
        "detections": {
            "instances": sum(class_counts.values()),
            "class_counts": dict(sorted(class_counts.items())),
            "postprocessing_totals": dict(diagnostic_totals),
        },
        "pose": {
            "validity_counts": dict(sorted(validity_counts.items())),
            "position_and_orientation_valid": sum(valid_pose_by_class.values()),
            "position_and_orientation_valid_by_class": dict(
                sorted(valid_pose_by_class.items())
            ),
            "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        },
        "latency_ms": {
            "registration_mean": statistics.fmean(registration_ms),
            "registration_p95": percentile(registration_ms, 95),
            "detection_and_pose_mean": statistics.fmean(detection_pose_ms),
            "detection_and_pose_p95": percentile(detection_pose_ms, 95),
            "pipeline_mean": statistics.fmean(pipeline_ms),
            "pipeline_p95": percentile(pipeline_ms, 95),
        },
        "artifacts": {
            "overlay_video": str(overlay_path),
            "predictions_jsonl": str(jsonl_path),
        },
    }
    summary_path = args.output_dir / f"{args.label}_pose_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
