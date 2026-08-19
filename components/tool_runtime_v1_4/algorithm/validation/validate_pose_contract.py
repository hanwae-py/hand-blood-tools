#!/usr/bin/env python3
"""Clinical-data-free contract test for mask RLE and planar pose."""

from __future__ import annotations

import math

import numpy as np

from pnu_surgical_tool import (
    CameraCalibration,
    DetectionBatch,
    DetectionInstance,
    PlanarPoseEstimator,
    SupportPlane,
)
from pnu_surgical_tool.rle import decode_uncompressed_coco_rle, encode_uncompressed_coco_rle


def main() -> None:
    height, width = 120, 160
    mask = np.zeros((height, width), dtype=bool)
    mask[54:66, 35:125] = True
    depth = np.full((height, width), np.nan, dtype=np.float32)
    depth[mask] = 1.0
    detection = DetectionInstance(
        frame_local_instance_id=0,
        canonical_class_id=7,
        model_class_index=6,
        class_name="Army-Navy Retractor",
        class_confidence=0.95,
        bbox_xyxy_px=(35.0, 54.0, 125.0, 66.0),
        mask=mask,
    )
    batch = DetectionBatch(
        image_width=width,
        image_height=height,
        model_version="synthetic-test",
        ontology_version="pnu.cam4.tool_ontology.v1",
        instances=[detection],
    )
    camera = CameraCalibration(
        width=width,
        height=height,
        k=np.array([[150.0, 0.0, 80.0], [0.0, 150.0, 60.0], [0.0, 0.0, 1.0]]),
        distortion=np.zeros(5),
        frame_name="synthetic_color_optical_frame",
        calibration_version="synthetic-v1",
    )
    plane = SupportPlane(
        normal=np.array([0.0, 0.0, -1.0]),
        offset_m=1.0,
        config_version="synthetic-plane-v1",
    )
    result = PlanarPoseEstimator().estimate(batch, depth, camera, plane, frame_key=123)
    assert result.frame_key == 123 and len(result.instances) == 1
    item = result.instances[0]
    assert item.pose_mode == "PLANAR_4DOF_WITH_NORMAL_PRIOR"
    assert item.position_valid and item.orientation_valid and item.validity == "VALID"
    assert item.symmetry_type == "C2"
    assert item.observation_point_uv_px is not None and item.position_m is not None
    assert item.orientation_xyzw is not None
    u, v = (int(round(value)) for value in item.observation_point_uv_px)
    assert mask[v, u] and math.isclose(item.position_m[2], 1.0, abs_tol=1e-6)
    assert math.isclose(np.linalg.norm(item.orientation_xyzw), 1.0, abs_tol=1e-6)
    rle = encode_uncompressed_coco_rle(mask)
    assert np.array_equal(mask, decode_uncompressed_coco_rle(rle))
    print("PASS: pose contract, P_obs, quaternion, validity, and RLE")


if __name__ == "__main__":
    main()

