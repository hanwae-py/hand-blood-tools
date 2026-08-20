#!/usr/bin/env python3
"""Validate the native-depth API against one camera in the reference MCAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import CameraInfo, CompressedImage

from pnu_surgical_tool import (
    CameraCalibration,
    decode_compressed_depth_16uc1,
    DepthToColorRegistrar,
    RigidTransform,
    validate_rgb_depth_timestamps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camera", default="cam_4")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--maximum-delta-ns", type=int, default=1_000_000)
    parser.add_argument("--registration-repeats", type=int, default=5)
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


def nearest_stamp_deltas_ms(color: list[int], depth: list[int]) -> np.ndarray:
    color_array = np.sort(np.asarray(color, dtype=np.int64))
    depth_array = np.sort(np.asarray(depth, dtype=np.int64))
    indices = np.searchsorted(color_array, depth_array)
    before = np.clip(indices - 1, 0, len(color_array) - 1)
    after = np.clip(indices, 0, len(color_array) - 1)
    return np.minimum(
        np.abs(depth_array - color_array[before]),
        np.abs(depth_array - color_array[after]),
    ).astype(np.float64) / 1e6


def main() -> None:
    args = parse_args()
    prefix = f"/synced/{args.camera}"
    color_info_topic = f"{prefix}/color/camera_info"
    depth_info_topic = f"{prefix}/depth/camera_info"
    color_topic = f"{prefix}/color/image_raw/compressed"
    depth_topic = f"{prefix}/depth/image_rect_raw/compressedDepth"
    extrinsics_topic = f"{prefix}/extrinsics/depth_to_color"
    typed_topics = {
        color_info_topic: CameraInfo,
        depth_info_topic: CameraInfo,
        color_topic: CompressedImage,
        depth_topic: CompressedImage,
    }
    first: dict[str, object] = {}
    color_stamps: list[int] = []
    depth_stamps: list[int] = []
    extrinsics_values = None
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id="mcap"),
        ConverterOptions("", ""),
    )
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic in typed_topics:
            message = deserialize_message(serialized, typed_topics[topic])
            first.setdefault(topic, message)
            if topic == color_topic:
                color_stamps.append(stamp_ns(message))
            elif topic == depth_topic:
                depth_stamps.append(stamp_ns(message))
        elif topic == extrinsics_topic and extrinsics_values is None:
            if len(serialized) != 100:
                raise ValueError(
                    f"unexpected RealSense Extrinsics CDR size: {len(serialized)}"
                )
            extrinsics_values = struct.unpack_from("<12d", serialized, 4)
    missing = set(typed_topics) - set(first)
    if missing or extrinsics_values is None:
        raise RuntimeError(f"missing reference inputs: {sorted(missing)}")

    color_message = first[color_topic]
    depth_message = first[depth_topic]
    first_delta_ns = validate_rgb_depth_timestamps(
        stamp_ns(color_message),
        stamp_ns(depth_message),
        args.maximum_delta_ns,
    )
    native_depth = decode_compressed_depth_16uc1(
        depth_message.data, depth_message.format
    )
    depth_camera = calibration(first[depth_info_topic], f"{args.camera}:depth")
    color_camera = calibration(first[color_info_topic], f"{args.camera}:color")
    transform = RigidTransform(
        rotation=np.asarray(extrinsics_values[:9]).reshape(3, 3),
        translation_m=np.asarray(extrinsics_values[9:]),
        source_frame=depth_camera.frame_name,
        target_frame=color_camera.frame_name,
        calibration_version=f"{args.camera}:depth_to_color",
    )
    started = time.perf_counter()
    registrar = DepthToColorRegistrar(depth_camera, color_camera, transform)
    registrar_init_ms = (time.perf_counter() - started) * 1000.0
    registration_times = []
    registration = None
    for _ in range(args.registration_repeats):
        started = time.perf_counter()
        registration = registrar.register(native_depth, args.depth_scale)
        registration_times.append((time.perf_counter() - started) * 1000.0)
    assert registration is not None
    deltas_ms = nearest_stamp_deltas_ms(color_stamps, depth_stamps)
    finite_depth = registration.aligned_depth_m[
        np.isfinite(registration.aligned_depth_m)
    ]
    report = {
        "status": "PASS",
        "bag": str(args.bag),
        "camera": args.camera,
        "input": {
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "color_count": len(color_stamps),
            "depth_count": len(depth_stamps),
            "depth_format": depth_message.format,
            "depth_dtype": str(native_depth.dtype),
            "depth_shape": list(native_depth.shape),
            "depth_scale_m_per_unit": args.depth_scale,
            "depth_scale_source": "caller-supplied; verify against device configuration",
            "color_frame": color_camera.frame_name,
            "depth_frame": depth_camera.frame_name,
            "first_pair_delta_ns": first_delta_ns,
            "nearest_stamp_delta_ms": {
                "median": float(np.median(deltas_ms)),
                "p95": float(np.percentile(deltas_ms, 95)),
                "maximum": float(np.max(deltas_ms)),
                "within_tolerance": int(
                    np.count_nonzero(deltas_ms * 1e6 <= args.maximum_delta_ns)
                ),
            },
        },
        "extrinsics": {
            "translation_m": transform.translation_m.tolist(),
            "translation_norm_m": float(np.linalg.norm(transform.translation_m)),
            "rotation": transform.rotation.tolist(),
        },
        "registration": {
            "registrar_init_ms": registrar_init_ms,
            "per_frame_ms": registration_times,
            "warm_median_ms": float(np.median(registration_times[1:])),
            "source_valid_pixels": registration.source_valid_pixels,
            "projected_points": registration.projected_points,
            "aligned_valid_pixels": registration.aligned_valid_pixels,
            "aligned_valid_ratio": registration.aligned_valid_ratio,
            "z_buffer_collisions": registration.z_buffer_collisions,
            "aligned_depth_m_percentiles": np.percentile(
                finite_depth, (2, 50, 98)
            ).tolist(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
