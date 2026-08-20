#!/usr/bin/env python3
"""Validate compressedDepth decoding, synchronization and z-buffer registration."""

from __future__ import annotations

import struct

import cv2
import numpy as np

from pnu_surgical_tool import (
    CameraCalibration,
    decode_compressed_depth_16uc1,
    DepthToColorRegistrar,
    RigidTransform,
    validate_rgb_depth_timestamps,
)


def camera(width: int, height: int, frame: str) -> CameraCalibration:
    return CameraCalibration(
        width=width,
        height=height,
        k=np.array(((100.0, 0.0, 2.0), (0.0, 100.0, 1.5), (0.0, 0.0, 1.0))),
        distortion=np.zeros(5),
        frame_name=frame,
        calibration_version=f"synthetic-{frame}",
    )


def main() -> None:
    raw = np.array(
        ((0, 1000, 1000, 0), (1000, 1000, 500, 1000), (0, 1000, 1000, 0)),
        dtype=np.uint16,
    )
    ok, png = cv2.imencode(".png", raw)
    assert ok
    codec_header = struct.pack("<iff", 0, 0.0, 0.0)
    decoded = decode_compressed_depth_16uc1(
        codec_header + png.tobytes(), "16UC1; compressedDepth png"
    )
    assert np.array_equal(decoded, raw)
    assert validate_rgb_depth_timestamps(10_000_000, 10_062_988, 1_000_000) == 62_988
    try:
        validate_rgb_depth_timestamps(0, 1_000_001, 1_000_000)
    except ValueError:
        pass
    else:
        raise AssertionError("timestamp mismatch was not rejected")

    depth_camera = camera(4, 3, "depth")
    color_camera = camera(4, 3, "color")
    identity = RigidTransform(
        rotation=np.eye(3),
        translation_m=np.zeros(3),
        source_frame="depth",
        target_frame="color",
        calibration_version="identity",
    )
    registered = DepthToColorRegistrar(
        depth_camera, color_camera, identity
    ).register(raw, 0.001)
    expected = raw.astype(np.float32) * 0.001
    expected[raw == 0] = np.nan
    assert np.allclose(registered.aligned_depth_m, expected, equal_nan=True)
    assert registered.source_valid_pixels == int(np.count_nonzero(raw))

    # tx=+0.01 m at z=1 m with fx=100 shifts the projection one pixel right.
    shifted = RigidTransform(
        rotation=np.eye(3),
        translation_m=np.array((0.01, 0.0, 0.0)),
        source_frame="depth",
        target_frame="color",
        calibration_version="shift-one-pixel",
    )
    shift_input = np.zeros((3, 4), dtype=np.uint16)
    shift_input[1, 1] = 1000
    shifted_result = DepthToColorRegistrar(
        depth_camera, color_camera, shifted
    ).register(shift_input, 0.001)
    assert np.isclose(shifted_result.aligned_depth_m[1, 2], 1.0)
    assert np.isnan(shifted_result.aligned_depth_m[1, 1])
    print("native depth registration validation: PASS")


if __name__ == "__main__":
    main()
