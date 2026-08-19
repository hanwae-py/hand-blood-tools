#!/usr/bin/env python3
"""Minimal mapping from package output into a host application's data model."""

from __future__ import annotations

from pnu_surgical_tool.types import ToolFrameResult


def to_host_records(result: ToolFrameResult) -> list[dict]:
    records = []
    for item in result.instances:
        records.append(
            {
                "source_frame_key": result.frame_key,
                "source_camera_frame": result.camera_frame_name,
                "instance_id_scope": "FRAME_LOCAL",
                "instance_id": item.frame_local_instance_id,
                "class_id": item.canonical_class_id,
                "class_name": item.class_name,
                "position_m": item.position_m,
                "quaternion_xyzw": item.orientation_xyzw,
                "pose_mode": item.pose_mode,
                "position_valid": item.position_valid,
                "orientation_valid": item.orientation_valid,
                "validity": item.validity,
                "symmetry_type": item.symmetry_type,
            }
        )
    return records

