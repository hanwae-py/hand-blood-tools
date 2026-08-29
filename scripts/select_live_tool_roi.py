#!/usr/bin/env python3
"""Draw and atomically save a Surgical Tool polygon ROI from live ROS RGB."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import cv2
import numpy as np
import yaml


CAMERA_DEFAULTS = {
    "cam_3": {
        "profile": "cam3_live_tray",
        "workspace_zone": "tray",
    },
    "cam_4": {
        "profile": "cam4_live_mayo",
        "workspace_zone": "mayo",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", choices=tuple(CAMERA_DEFAULTS))
    parser.add_argument("--profile")
    parser.add_argument("--output-yaml", type=Path)
    parser.add_argument("--minimum-mask-overlap", type=float, default=0.5)
    parser.add_argument("--first-frame-timeout-sec", type=float, default=8.0)
    args = parser.parse_args()
    defaults = CAMERA_DEFAULTS[args.camera]
    args.profile = args.profile or defaults["profile"]
    args.workspace_zone = defaults["workspace_zone"]
    if args.output_yaml is None:
        args.output_yaml = (
            root
            / "components/tool_runtime_v1_6/ros2_ws/src/"
            "pnu_surgical_perception/config/roi_profiles"
            / f"{args.profile}.yaml"
        )
    return args


def polygon_area(points: list[tuple[int, int]], width: int, height: int) -> float:
    area_twice = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area_twice += x1 * y2 - x2 * y1
    return abs(area_twice) / (2.0 * width * height)


def draw(frame: np.ndarray, points: list[tuple[int, int]], status: str) -> np.ndarray:
    output = frame.copy()
    if points:
        polygon = np.asarray(points, dtype=np.int32)
        cv2.polylines(output, [polygon], len(points) >= 3, (30, 235, 30), 3)
        for index, point in enumerate(points, start=1):
            cv2.circle(output, point, 6, (0, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(
                output, str(index), (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
    cv2.rectangle(output, (0, 0), (output.shape[1], 82), (18, 18, 18), -1)
    lines = (
        "Left click: add | Right click/Backspace: undo | R: reset",
        "Enter/S: save (>=3 points) | Esc/Q: cancel",
        status,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            output, line, (10, 22 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (245, 245, 245), 1, cv2.LINE_AA,
        )
    return output


def receive_frame(camera: str, timeout_sec: float) -> tuple[np.ndarray, str]:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    state: dict[str, object] = {"jpeg": None, "frame_id": ""}
    topic = f"/perception/ingress/{camera}/color/image_raw/compressed"

    class FrameSubscriber(Node):
        def __init__(self) -> None:
            super().__init__(f"{camera}_tool_roi_selector")
            self.create_subscription(
                CompressedImage, topic, self.on_frame, qos_profile_sensor_data
            )

        def on_frame(self, message: CompressedImage) -> None:
            state["jpeg"] = bytes(message.data)
            state["frame_id"] = str(message.header.frame_id)

    rclpy.init()
    node = FrameSubscriber()
    deadline = time.monotonic() + timeout_sec
    try:
        while state["jpeg"] is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if state["jpeg"] is None:
        raise RuntimeError(f"no live frame received from {topic}")
    frame = cv2.imdecode(
        np.frombuffer(state["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise RuntimeError(f"failed to decode live JPEG from {topic}")
    return frame, str(state["frame_id"])


def select_polygon(frame: np.ndarray, camera: str) -> list[tuple[int, int]] | None:
    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []
    window = f"{camera.upper()} Surgical Tool ROI"
    status = "Add polygon vertices around the active workspace."

    def mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        nonlocal status
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            status = f"{len(points)} vertices selected."
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            status = f"{len(points)} vertices selected."

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height)
    cv2.setMouseCallback(window, mouse)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    try:
        while True:
            cv2.imshow(window, draw(frame, points, status))
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                return None
            if key in (8, 127) and points:
                points.pop()
                status = f"{len(points)} vertices selected."
            elif key in (ord("r"), ord("R")):
                points.clear()
                status = "Selection reset."
            elif key in (10, 13, ord("s"), ord("S")):
                if len(points) < 3:
                    status = "At least 3 vertices are required."
                elif polygon_area(points, width, height) < 0.001:
                    status = "ROI polygon is too small."
                else:
                    return list(points)
    finally:
        cv2.destroyAllWindows()


def write_profile(args: argparse.Namespace, frame: np.ndarray,
                  points: list[tuple[int, int]], frame_id: str) -> None:
    if not re.fullmatch(r"cam[34]_[a-z0-9_-]+", args.profile):
        raise ValueError("profile must start with cam3_ or cam4_")
    expected_prefix = args.camera.replace("_", "") + "_"
    if not args.profile.startswith(expected_prefix):
        raise ValueError(f"profile {args.profile!r} does not match {args.camera}")
    if not 0.0 <= args.minimum_mask_overlap <= 1.0:
        raise ValueError("minimum-mask-overlap must lie in [0, 1]")
    height, width = frame.shape[:2]
    normalized: list[float] = []
    for x, y in points:
        normalized.extend((round(x / width, 6), round(y / height, 6)))
    payload = {
        "/**": {
            "ros__parameters": {
                "workspace_zone": args.workspace_zone,
                "workspace_roi_profile": args.profile,
                "workspace_roi_enabled": True,
                "workspace_roi_polygon_norm_xy": normalized,
                "workspace_roi_minimum_mask_overlap": args.minimum_mask_overlap,
                "workspace_roi_require_mask_centroid_inside": True,
            }
        }
    }
    content = (
        f"# Selected from live {args.camera} {width}x{height}; "
        f"frame_id={frame_id or '<empty>'}.\n"
        + yaml.safe_dump(payload, sort_keys=False)
    )
    output = args.output_yaml.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (output.is_symlink() or not output.is_file()):
        raise ValueError("output YAML must be a regular file")
    mode = output.stat().st_mode & 0o777 if output.exists() else 0o644
    lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / (
        f"tool_roi_{args.camera}.lock"
    )
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, output)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    print(json.dumps({
        "status": "saved",
        "camera": args.camera,
        "profile": args.profile,
        "output_yaml": str(output),
        "image_size": {"width": width, "height": height},
        "polygon_norm_xy": normalized,
    }, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    frame, frame_id = receive_frame(args.camera, args.first_frame_timeout_sec)
    points = select_polygon(frame, args.camera)
    if points is None:
        print(json.dumps({"status": "cancelled", "camera": args.camera}))
        return 0
    write_profile(args, frame, points, frame_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
