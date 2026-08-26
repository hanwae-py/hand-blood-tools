#!/usr/bin/env python3
"""Interactively draw a normalized workspace polygon on a video frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-time-sec", type=float, default=0.0)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace-zone", required=True)
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--minimum-mask-overlap", type=float, default=0.5)
    return parser.parse_args()


def draw(frame: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    output = frame.copy()
    if points:
        polygon = np.asarray(points, dtype=np.int32)
        cv2.polylines(output, [polygon], len(points) >= 3, (30, 235, 30), 3)
        for index, point in enumerate(points, start=1):
            cv2.circle(output, point, 6, (0, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(
                output,
                str(index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.rectangle(output, (0, 0), (output.shape[1], 54), (18, 18, 18), -1)
    cv2.putText(
        output,
        "Left click: add | Right click/Backspace: undo | R: reset",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Enter/S: save (>=3 points) | Esc/Q: cancel",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.frame_time_sec < 0.0:
        raise ValueError("frame-time-sec must not be negative")
    if not 0.0 <= args.minimum_mask_overlap <= 1.0:
        raise ValueError("minimum-mask-overlap must lie in [0, 1]")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    capture.set(cv2.CAP_PROP_POS_MSEC, args.frame_time_sec * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(
            f"failed to decode {args.video} at {args.frame_time_sec:.3f}s"
        )
    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []
    window = "Draw workspace ROI"

    def mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height)
    cv2.setMouseCallback(window, mouse)
    saved = False
    while True:
        cv2.imshow(window, draw(frame, points))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (8, 127) and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (10, 13, ord("s")) and len(points) >= 3:
            saved = True
            break
    cv2.destroyAllWindows()
    if not saved:
        print("ROI selection cancelled; no file written")
        return

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
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    preview = args.preview or args.output_yaml.with_suffix(".jpg")
    preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview), draw(frame, points)):
        raise RuntimeError(f"failed to write preview: {preview}")
    print(f"wrote {args.output_yaml}")
    print(f"wrote {preview}")
    print(f"normalized polygon: {normalized}")


if __name__ == "__main__":
    main()
