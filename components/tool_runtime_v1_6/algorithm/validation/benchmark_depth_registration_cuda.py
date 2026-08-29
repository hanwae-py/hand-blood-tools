#!/usr/bin/env python3
"""Shadow-check NumPy/CUDA registration on live RGB-D calibration and depth.

This is an observation-only validator: it subscribes to ingress topics, does
not publish, and never changes the timestamps or state of the running pipeline.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pnu_surgical_tool.depth_registration import (
    decode_compressed_depth_16uc1,
    registrar_from_camera_messages,
)


def _collect_live(camera: str, frame_count: int, timeout_sec: float):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from realsense2_camera_msgs.msg import Extrinsics
    from sensor_msgs.msg import CameraInfo, CompressedImage

    prefix = f"/perception/ingress/{camera}"
    sensor_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    latched_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class Collector(Node):
        def __init__(self):
            super().__init__("depth_registration_cuda_shadow_validator")
            self.color_info = None
            self.depth_info = None
            self.extrinsics = None
            self.depth_messages = []
            self.depth_stamps = set()
            self.create_subscription(
                CameraInfo,
                f"{prefix}/color/camera_info",
                self._color_info,
                sensor_qos,
            )
            self.create_subscription(
                CameraInfo,
                f"{prefix}/depth/camera_info",
                self._depth_info,
                sensor_qos,
            )
            self.create_subscription(
                Extrinsics,
                f"{prefix}/extrinsics/depth_to_color",
                self._extrinsics,
                latched_qos,
            )
            self.create_subscription(
                CompressedImage,
                f"{prefix}/depth/image_rect_raw/compressedDepth",
                self._depth,
                sensor_qos,
            )

        def _color_info(self, message):
            self.color_info = message

        def _depth_info(self, message):
            self.depth_info = message

        def _extrinsics(self, message):
            self.extrinsics = message

        def _depth(self, message):
            stamp = (
                int(message.header.stamp.sec),
                int(message.header.stamp.nanosec),
            )
            if stamp in self.depth_stamps or len(self.depth_messages) >= frame_count:
                return
            self.depth_stamps.add(stamp)
            self.depth_messages.append(message)

        @property
        def complete(self):
            return (
                self.color_info is not None
                and self.depth_info is not None
                and self.extrinsics is not None
                and len(self.depth_messages) >= frame_count
            )

    rclpy.init(args=None)
    node = Collector()
    deadline = time.monotonic() + timeout_sec
    try:
        while not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.complete:
            raise RuntimeError(
                "live capture timed out: "
                f"color_info={node.color_info is not None}, "
                f"depth_info={node.depth_info is not None}, "
                f"extrinsics={node.extrinsics is not None}, "
                f"depth_frames={len(node.depth_messages)}/{frame_count}"
            )
        return (
            node.color_info,
            node.depth_info,
            node.extrinsics,
            list(node.depth_messages),
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _summary(samples_ms):
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
        "p99_ms": round(float(np.percentile(values, 99)), 4),
        "max_ms": round(float(np.max(values)), 4),
        "mean_ms": round(float(np.mean(values)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="cam_3")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--cuda-library", required=True)
    parser.add_argument("--p95-gate-ms", type=float, default=66.7)
    args = parser.parse_args()
    if args.frames <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("frames/iterations must be positive and warmup non-negative")

    color_info, depth_info, extrinsics, depth_messages = _collect_live(
        args.camera, args.frames, args.timeout_sec
    )
    frames = [
        decode_compressed_depth_16uc1(message.data, message.format)
        for message in depth_messages
    ]
    calibration_version = f"{args.camera}:live-shadow"
    reference = registrar_from_camera_messages(
        color_info,
        depth_info,
        extrinsics.rotation,
        extrinsics.translation,
        calibration_version,
        backend="numpy",
    )
    candidate = registrar_from_camera_messages(
        color_info,
        depth_info,
        extrinsics.rotation,
        extrinsics.translation,
        calibration_version,
        backend="cuda",
        allow_sticky_numpy_fallback=False,
        cuda_library_path=args.cuda_library,
    )

    parity_failures = []
    maximum_absolute_error_m = 0.0
    reference_wall_ms = []
    candidate_wall_ms = []
    candidate_gpu_ms = []
    counters = []
    for index, frame in enumerate(frames):
        started = time.perf_counter()
        expected = reference.register(frame, 0.001)
        reference_wall_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        actual = candidate.register(frame, 0.001)
        candidate_wall_ms.append((time.perf_counter() - started) * 1000.0)
        candidate_gpu_ms.append(candidate.last_gpu_ms)
        expected_mask = np.isfinite(expected.aligned_depth_m)
        actual_mask = np.isfinite(actual.aligned_depth_m)
        mask_mismatches = int(np.count_nonzero(expected_mask != actual_mask))
        overlap = expected_mask & actual_mask
        max_error = (
            float(
                np.max(
                    np.abs(
                        expected.aligned_depth_m[overlap]
                        - actual.aligned_depth_m[overlap]
                    )
                )
            )
            if np.any(overlap)
            else 0.0
        )
        maximum_absolute_error_m = max(maximum_absolute_error_m, max_error)
        expected_counts = (
            expected.source_valid_pixels,
            expected.projected_points,
            expected.aligned_valid_pixels,
            expected.z_buffer_collisions,
        )
        actual_counts = (
            actual.source_valid_pixels,
            actual.projected_points,
            actual.aligned_valid_pixels,
            actual.z_buffer_collisions,
        )
        counters.append(actual_counts)
        values_close = bool(
            np.allclose(
                actual.aligned_depth_m[overlap],
                expected.aligned_depth_m[overlap],
                atol=1e-5,
                rtol=2e-6,
            )
        )
        if mask_mismatches or actual_counts != expected_counts or not values_close:
            parity_failures.append(
                {
                    "frame": index,
                    "mask_mismatches": mask_mismatches,
                    "expected_counts": expected_counts,
                    "actual_counts": actual_counts,
                    "max_absolute_error_m": max_error,
                    "values_close": values_close,
                }
            )

    for index in range(args.warmup):
        candidate.register(frames[index % len(frames)], 0.001)
    benchmark_wall_ms = []
    benchmark_gpu_ms = []
    benchmark_started = time.perf_counter()
    for index in range(args.iterations):
        started = time.perf_counter()
        candidate.register(frames[index % len(frames)], 0.001)
        benchmark_wall_ms.append((time.perf_counter() - started) * 1000.0)
        benchmark_gpu_ms.append(candidate.last_gpu_ms)
    benchmark_elapsed = time.perf_counter() - benchmark_started
    sustained_hz = args.iterations / benchmark_elapsed
    wall_summary = _summary(benchmark_wall_ms)
    passed = bool(
        not parity_failures
        and wall_summary["p95_ms"] < args.p95_gate_ms
        and sustained_hz >= 15.0
        and not candidate.fallback_active
    )
    report = {
        "schema": "pnu.depth_registration_cuda_shadow.v1",
        "camera": args.camera,
        "input_shape": list(frames[0].shape),
        "output_shape": [int(color_info.height), int(color_info.width)],
        "frames_compared": len(frames),
        "backend_requested": candidate.requested_backend,
        "backend_active": candidate.backend_name,
        "backend_version": candidate.backend_version,
        "fallback_active": candidate.fallback_active,
        "parity_failures": parity_failures,
        "maximum_absolute_error_m": maximum_absolute_error_m,
        "reference_shadow_wall": _summary(reference_wall_ms),
        "candidate_shadow_wall": _summary(candidate_wall_ms),
        "candidate_shadow_gpu": _summary(candidate_gpu_ms),
        "benchmark_full_call": wall_summary,
        "benchmark_gpu": _summary(benchmark_gpu_ms),
        "benchmark_sustained_hz": round(float(sustained_hz), 3),
        "p95_gate_ms": args.p95_gate_ms,
        "representative_counts": counters[-1],
        "passed": passed,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
