#!/usr/bin/env python3
"""Generate the runtime-bundle manifest and checksum file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "build",
    "install",
    "log",
}
EXCLUDED_NAMES = {"MANIFEST.json", "SHA256SUMS"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def payload_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix != ".pyc"
    )


def main() -> None:
    files = payload_files()
    model = ROOT / "algorithm/model/cam4_rfdetr_seg_small_regular_resume_e13_best.pth"
    manifest = {
        "schema": "pnu.surgical_tool.ros2_runtime_manifest.v1",
        "release_id": "PNU_CVLAB_SURGICAL_TOOL_ROS2_RUNTIME_v1_6_COMPAT_20260820_rc1",
        "version": "1.6.0-rc1-compatible",
        "release_date": "2026-08-20",
        "provider": "Pusan National University Computer Vision Laboratory",
        "recipient_role": "surgical-tool algorithm integration owner",
        "target": "ROS 2 Jazzy integration PC",
        "reference_bag_included": False,
        "algorithm_version": "1.6.0-rc1-compatible",
        "message_package": "surgical_perception_msgs/0.2.0",
        "pose_mode": "PLANAR_4DOF_WITH_NORMAL_PRIOR",
        "full_6d_available": False,
        "model_sha256": digest(model),
        "included": [
            "RF-DETR checkpoint, ontology and inference source",
            "native 16UC1 compressedDepth decode and depth-to-color registration",
            "ROS2 RGB/depth pairing, inference, pose and typed output node",
            "surgical_perception_msgs source",
            "reference CAM4 parameter file, build/run scripts and validators",
        ],
        "known_pending": [
            "authoritative device depth scale",
            "production Mayo stand/tray support-plane calibration",
            "RF-DETR overfitting/generalization resolution",
            "live DDS rosbag-play acceptance on recipient PC",
        ],
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    manifest_path = ROOT / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_files = sorted([*files, manifest_path])
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in checksum_files
    ]
    (ROOT / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest for {len(files)} payload files")


if __name__ == "__main__":
    main()
