"""Optional OpenCV overlay for standalone inspection."""

from __future__ import annotations

import cv2
import numpy as np

from .types import DetectionBatch, ToolFrameResult


def draw_detections_bgr(image_bgr: np.ndarray, detections: DetectionBatch) -> np.ndarray:
    output = np.asarray(image_bgr).copy()
    for item in detections.instances:
        color = (40, 210, 250)
        overlay = output.copy()
        overlay[item.mask] = color
        output = cv2.addWeighted(output, 0.72, overlay, 0.28, 0.0)
        x0, y0, x1, y1 = (int(round(value)) for value in item.bbox_xyxy_px)
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            output,
            f"{item.class_name} {item.class_confidence:.2f}",
            (x0, max(18, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def draw_observation_points_bgr(image_bgr: np.ndarray, result: ToolFrameResult) -> np.ndarray:
    output = np.asarray(image_bgr).copy()
    for item in result.instances:
        if item.observation_point_uv_px is None:
            continue
        u, v = (int(round(value)) for value in item.observation_point_uv_px)
        color = (30, 220, 30) if item.validity == "VALID" else (30, 160, 255)
        cv2.circle(output, (u, v), 6, color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{item.class_name}:{item.validity}",
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    return output
