#!/usr/bin/env python3
"""Local/recipient validation with an approved image and aligned-point fixture."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from pnu_surgical_tool import (
    CameraCalibration,
    DetectorConfig,
    PlanarPoseEstimator,
    SupportPlane,
    SurgicalToolDetector,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--aligned-points-npy", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--expected-json", type=Path)
    parser.add_argument("--position-tolerance-m", type=float, default=0.010)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)
    points = np.load(args.aligned_points_npy)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("aligned points must be HxWx3")
    geometry = json.loads(args.geometry_json.read_text())
    camera = CameraCalibration(
        width=bgr.shape[1],
        height=bgr.shape[0],
        k=np.asarray(geometry["color_k"]),
        distortion=np.asarray(geometry["color_d"]),
        frame_name=str(geometry["pose_frame"]),
        calibration_version=str(geometry.get("status", "fixture")),
    )
    report = geometry.get("support_plane_report", {})
    plane = SupportPlane(
        normal=np.asarray(geometry["support_plane_normal"]),
        offset_m=float(geometry["support_plane_offset_m"]),
        config_version=str(geometry.get("schema", "fixture")),
        inlier_ratio=report.get("inlier_ratio"),
        residual_p95_m=report.get("residual_p95_m"),
    )
    detector = SurgicalToolDetector(
        DetectorConfig(
            ROOT / "model/cam4_rfdetr_seg_small_v1.pth",
            ROOT / "model/ontology.json",
        )
    )
    detections = detector.predict(bgr, "BGR")
    result = PlanarPoseEstimator().estimate(
        detections,
        points[..., 2].astype(np.float32),
        camera,
        plane,
        frame_key="approved-fixture",
    )
    valid = [item for item in result.instances if item.validity == "VALID"]
    assert all(item.position_m is not None for item in result.instances)
    assert all(
        item.orientation_xyzw is None
        or abs(float(np.linalg.norm(item.orientation_xyzw)) - 1.0) < 1e-5
        for item in result.instances
    )

    if args.expected_json is not None:
        expected = json.loads(args.expected_json.read_text())
        expected_rows = expected["instances"]
        assert len(result.instances) == int(expected["instance_count"])
        assert Counter(item.class_name for item in result.instances) == Counter(
            str(item["class_name"]) for item in expected_rows
        )
        position_errors = []
        unmatched = set(range(len(expected_rows)))
        for actual in result.instances:
            if actual.position_m is None:
                raise AssertionError("missing actual position")
            candidates = [
                (
                    float(
                        np.linalg.norm(
                            np.asarray(actual.position_m)
                            - np.asarray(expected_rows[index]["position_m"])
                        )
                    ),
                    index,
                )
                for index in unmatched
                if expected_rows[index]["class_name"] == actual.class_name
            ]
            if not candidates:
                raise AssertionError(f"no unmatched expected instance for {actual.class_name}")
            error, matched_index = min(candidates)
            unmatched.remove(matched_index)
            position_errors.append(error)
        assert not unmatched
        assert max(position_errors) <= args.position_tolerance_m, max(position_errors)
        print(
            f"PASS: {len(result.instances)} detections, {len(valid)} VALID poses, "
            f"max P_obs error {max(position_errors) * 1000:.3f} mm"
        )
    else:
        print(f"PASS: {len(result.instances)} detections, {len(valid)} VALID poses")


if __name__ == "__main__":
    main()
