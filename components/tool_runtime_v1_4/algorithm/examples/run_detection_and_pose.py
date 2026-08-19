#!/usr/bin/env python3
"""Run RF-DETR and constrained pose with caller-supplied aligned depth/config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pnu_surgical_tool import (
    CameraCalibration,
    DetectorConfig,
    PlanarPoseEstimator,
    SupportPlane,
    SurgicalToolAlgorithm,
    SurgicalToolDetector,
)
from pnu_surgical_tool.rle import encode_uncompressed_coco_rle
from pnu_surgical_tool.types import result_to_dict


ROOT = Path(__file__).resolve().parents[1]


def require_complete_config(payload: dict) -> None:
    required = ["width", "height", "camera_frame_name", "calibration_version", "k", "distortion"]
    if any(payload.get(key) is None for key in required):
        raise ValueError("camera config still contains null template values")
    plane = payload.get("support_plane", {})
    if any(plane.get(key) is None for key in ("normal_camera_frame", "offset_m", "config_version")):
        raise ValueError("support-plane config still contains null template values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--aligned-depth-npy", type=Path, required=True)
    parser.add_argument("--camera-pose-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model/cam4_rfdetr_seg_small_v1.pth")
    parser.add_argument("--ontology", type=Path, default=ROOT / "model/ontology.json")
    parser.add_argument("--color-order", choices=("RGB", "BGR"), required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frame-key")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    supplied = bgr if args.color_order == "BGR" else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(args.aligned_depth_npy)
    config = json.loads(args.camera_pose_config.read_text())
    require_complete_config(config)
    camera = CameraCalibration(
        width=int(config["width"]),
        height=int(config["height"]),
        k=np.asarray(config["k"]),
        distortion=np.asarray(config["distortion"]),
        frame_name=str(config["camera_frame_name"]),
        calibration_version=str(config["calibration_version"]),
    )
    plane_record = config["support_plane"]
    plane = SupportPlane(
        normal=np.asarray(plane_record["normal_camera_frame"]),
        offset_m=float(plane_record["offset_m"]),
        config_version=str(plane_record["config_version"]),
    )
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=args.checkpoint,
            ontology_path=args.ontology,
            confidence_threshold=args.threshold,
            optimize=not args.no_optimize,
        )
    )
    algorithm = SurgicalToolAlgorithm(detector, PlanarPoseEstimator())
    result = algorithm.detect_and_estimate(
        supplied,
        depth,
        camera,
        plane,
        color_order=args.color_order,
        frame_key=args.frame_key,
    )
    payload = result_to_dict(result)
    for output_item, result_item in zip(payload["instances"], result.instances, strict=True):
        output_item["mask_rle"] = encode_uncompressed_coco_rle(result_item.mask)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"instances": len(result.instances), "output": str(args.output_json)}))


if __name__ == "__main__":
    main()

