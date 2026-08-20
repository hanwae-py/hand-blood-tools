"""Optional OpenCV overlay for standalone inspection."""

from __future__ import annotations

import cv2
import numpy as np

from .types import CameraCalibration, DetectionBatch, ToolFrameResult


AXIS_COLORS_BGR = {
    "X": (0, 0, 255),
    "Y": (0, 210, 0),
    "Z": (255, 80, 0),
}


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


def quaternion_xyzw_to_rotation_matrix(
    quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Convert a finite xyzw quaternion to a 3x3 rotation matrix."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion_xyzw has zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project_camera_points(
    points_m: np.ndarray, camera: CameraCalibration
) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 1e-6)
    if not np.any(valid):
        return projected
    distortion = np.asarray(camera.distortion, dtype=np.float64).reshape(-1)
    image_points, _ = cv2.projectPoints(
        points[valid],
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(camera.k, dtype=np.float64),
        distortion if distortion.size else None,
    )
    projected[valid] = image_points.reshape(-1, 2)
    return projected


def draw_pose_axes_bgr(
    image_bgr: np.ndarray,
    result: ToolFrameResult,
    camera: CameraCalibration,
    axis_length_m: float = 0.05,
) -> np.ndarray:
    """Draw quaternion X/Y/Z axes on a dedicated pose-only overlay.

    The quaternion rotation-matrix columns are +X, +Y and +Z in the color
    camera frame. Degraded orientations are still rendered so the overlay
    represents the transmitted quaternion, but their label explicitly carries
    the degraded validity. Instances without a metric position or quaternion
    are omitted.
    """
    output = np.asarray(image_bgr).copy()
    if output.dtype != np.uint8 or output.ndim != 3 or output.shape[2] != 3:
        raise ValueError("image_bgr must be uint8 HxWx3")
    if output.shape[:2] != (camera.height, camera.width):
        raise ValueError("image shape does not match camera calibration")
    if not np.isfinite(axis_length_m) or axis_length_m <= 0.0:
        raise ValueError("axis_length_m must be finite and positive")

    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(
        output,
        "POSE AXES: X=red Y=green Z=blue | constrained planar quaternion",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    height, width = output.shape[:2]
    for item in result.instances:
        if item.position_m is None or item.orientation_xyzw is None:
            continue
        origin = np.asarray(item.position_m, dtype=np.float64)
        rotation = quaternion_xyzw_to_rotation_matrix(item.orientation_xyzw)
        points = np.vstack(
            [origin, origin + axis_length_m * rotation[:, 0],
             origin + axis_length_m * rotation[:, 1],
             origin + axis_length_m * rotation[:, 2]]
        )
        pixels = _project_camera_points(points, camera)
        if not np.all(np.isfinite(pixels[0])):
            continue
        origin_px = tuple(int(round(value)) for value in pixels[0])
        if not (0 <= origin_px[0] < width and 0 <= origin_px[1] < height):
            continue
        thickness = 2 if item.orientation_valid else 1
        for axis_index, axis_name in enumerate(("X", "Y", "Z"), start=1):
            if not np.all(np.isfinite(pixels[axis_index])):
                continue
            endpoint = tuple(int(round(value)) for value in pixels[axis_index])
            color = AXIS_COLORS_BGR[axis_name]
            cv2.arrowedLine(
                output,
                origin_px,
                endpoint,
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            if 0 <= endpoint[0] < width and 0 <= endpoint[1] < height:
                cv2.putText(
                    output,
                    axis_name,
                    (endpoint[0] + 3, endpoint[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.circle(output, origin_px, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{item.class_name}:{item.validity}",
            (origin_px[0] + 6, origin_px[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output
