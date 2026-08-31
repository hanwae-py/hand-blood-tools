"""Render current operator overlays as one 2x2 view and four view topics.

Unlike the legacy difference-image compositor, this node never compares two
opaque JPEG video frames.  It draws Tool observations/poses, Hand keypoints,
and Blood masks from their typed result messages on the single latest ingress
base frame.  Every layer is independently freshness-gated, so a slow worker
cannot freeze or ghost the video from another worker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from hand_keypoint_interfaces.msg import (
    HandFacingArray,
    HandGestureArray,
    HandKeypoints,
)
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from surgical_perception_msgs.msg import ToolObservation2DArray, ToolPoseArray

from pnu_surgical_perception.final_overlay_contract import (
    CAMERA_STATUS_KEYS,
    LAYER_NAMES,
    STATUS_SCHEMA,
    Freshness,
    freshness_state,
    layer_is_drawable,
    stamp_dict,
    stamp_ns,
)
from pnu_surgical_perception.cam4_palm_pose_transform import (
    PalmPoseComponents,
    select_camera_palm,
)


CAMERAS = ('cam_3', 'cam_4')
# These names intentionally describe the four *operator-rendered* panels, not
# the worker-stage overlays (``tool/overlay``, ``hand/overlay``, etc.).  A
# panel topic must carry the same pixels an operator sees in the final view.
PANEL_NAMES = ('cam_3', 'cam_4', 'suction', 'right_ee')
TOOL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    'Scalpel': (0, 145, 255),
    'Allis Forceps': (255, 130, 30),
    'Mosquito': (70, 190, 80),
    'Adson Forceps': (205, 70, 195),
    'Bipolar Forceps': (0, 230, 230),
    'Bovie': (100, 60, 255),
    'Army-Navy Retractor': (185, 85, 155),
    'Thyroid Retractor': (210, 180, 30),
}
GESTURE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    'Closed_Fist': (80, 170, 255),
    'Open_Palm': (70, 235, 90),
    'Pointing_Up': (255, 185, 45),
    'Thumb_Down': (70, 90, 245),
    'Thumb_Up': (245, 220, 55),
    'Victory': (230, 105, 230),
    'ILoveYou': (255, 145, 75),
    'None': (185, 185, 185),
}
PALM_FACING_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    'PALM_UP': (70, 235, 90),
    'PALM_DOWN': (80, 170, 255),
    'EDGE': (255, 190, 60),
    'UNKNOWN': (165, 165, 165),
}
HAND_ROI_COLOR_BGR = (0, 190, 255)
TOOL_ROI_COLOR_BGR = (80, 235, 110)
PALM_AXIS_COLORS_BGR = (
    (40, 40, 255),  # +X red
    (40, 230, 40),  # +Y green
    (255, 120, 40),  # +Z blue
)
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def image_reader_qos() -> QoSProfile:
    """Latest-frame image reader QoS for the local ingress fan-out."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def result_qos() -> QoSProfile:
    """Reliable typed-result QoS with no accumulating Debug backlog."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def camera_info_qos() -> QoSProfile:
    """Reliable CameraInfo contract; images use ``image_reader_qos`` instead."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def status_qos() -> QoSProfile:
    """Small retained status document for a Debug UI that joins late."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def decode_jpeg(message: CompressedImage) -> np.ndarray | None:
    """Decode one base frame without letting a malformed packet kill Debug."""
    return cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_binary_mask(message: Image) -> np.ndarray | None:
    """Decode the current Blood ``mono8`` result without changing its stamp."""
    encoding = str(message.encoding).lower()
    if encoding not in {'mono8', '8uc1'}:
        return None
    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or step < width:
        return None
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size < height * step:
        return None
    return data[:height * step].reshape(height, step)[:, :width] > 0


def quaternion_matrix_xyzw(quaternion: Any) -> np.ndarray | None:
    """Return a rotation matrix for a finite, normalized ROS quaternion."""
    try:
        values = np.asarray(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    x, y, z, w = values / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def camera_palm_axis_points(
    palm_pose: Any, axis_length_m: float,
) -> np.ndarray | None:
    """Return camera-frame origin/+XYZ endpoints for a valid palm 6-D pose."""
    try:
        position = np.asarray([
            palm_pose.translation.x,
            palm_pose.translation.y,
            palm_pose.translation.z,
        ], dtype=np.float64)
    except AttributeError:
        return None
    try:
        axis_length = float(axis_length_m)
    except (TypeError, ValueError):
        return None
    rotation = quaternion_matrix_xyzw(getattr(palm_pose, 'orientation', None))
    if (
        rotation is None
        or not math.isfinite(axis_length)
        or axis_length <= 0.0
        or not np.all(np.isfinite(position))
    ):
        return None
    return np.vstack((
        position,
        position + rotation[:, 0] * axis_length,
        position + rotation[:, 1] * axis_length,
        position + rotation[:, 2] * axis_length,
    ))


def matched_humanoid_palm(
    hand_message: HandKeypoints | None,
    humanoid_pose: PoseStamped | None,
) -> tuple[PalmPoseComponents, tuple[float, float, float]] | None:
    """Join an exact source-stamped right palm with its humanoid-frame pose.

    The transformed pose has no ``hand_index``.  Use exactly the same
    single-right-palm selection contract as the transform node, then require
    the output source timestamp and frame ID to agree before placing a label.
    """
    if hand_message is None or humanoid_pose is None:
        return None
    try:
        if stamp_ns(hand_message) != stamp_ns(humanoid_pose):
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    if str(getattr(humanoid_pose.header, 'frame_id', '')).strip() != 'humanoid':
        return None
    selected, _reason = select_camera_palm(hand_message)
    if selected is None:
        return None
    try:
        position = tuple(float(getattr(humanoid_pose.pose.position, key))
                         for key in ('x', 'y', 'z'))
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in position):
        return None
    if quaternion_matrix_xyzw(getattr(humanoid_pose.pose, 'orientation', None)) is None:
        return None
    return selected, position


def draw_outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a high-contrast label that remains legible after panel resizing."""
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (8, 12, 18), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color, thickness, cv2.LINE_AA,
    )


def draw_outlined_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw a stable, contrast-preserving detection box."""
    cv2.rectangle(image, top_left, bottom_right, (8, 12, 18), thickness + 3, cv2.LINE_AA)
    cv2.rectangle(image, top_left, bottom_right, color, thickness, cv2.LINE_AA)


def draw_humanoid_palm_hud(
    image: np.ndarray,
    position_m: tuple[float, float, float],
    *,
    top_y: int = 58,
) -> None:
    """Draw the source-stamp-matched humanoid XYZ without implying control."""
    x0 = 18
    y0 = min(max(58, int(top_y)), max(58, image.shape[0] - 92))
    x1 = min(image.shape[1] - 18, 520)
    y1 = min(image.shape[0] - 18, 132)
    if x1 <= x0 or y1 <= y0:
        return
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 24, 36), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    color = (105, 230, 255)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    draw_outlined_text(
        image, 'HUMANOID PALM  [STATIC ANCHOR]', (x0 + 12, y0 + 29),
        0.55, color, 2,
    )
    draw_outlined_text(
        image,
        f'XYZ  {position_m[0]:+.3f}  {position_m[1]:+.3f}  {position_m[2]:+.3f} m',
        (x0 + 12, y0 + 61), 0.66, (235, 245, 250), 2,
    )


def normalized_roi_pixel_box(
    roi: tuple[float, float, float, float] | list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Map the Hand node's normalized centroid ROI onto its source image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError('ROI image dimensions must be positive')
    if len(roi) != 4:
        raise ValueError('ROI must contain four values')
    values = tuple(float(value) for value in roi)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('ROI values must be finite')
    x_min, x_max, y_min, y_max = values
    if not (0.0 <= x_min < x_max <= 1.0):
        raise ValueError('ROI must satisfy 0 <= x_min < x_max <= 1')
    if not (0.0 <= y_min < y_max <= 1.0):
        raise ValueError('ROI must satisfy 0 <= y_min < y_max <= 1')
    return (
        min(image_width - 1, max(0, int(round(x_min * image_width)))),
        min(image_height - 1, max(0, int(round(y_min * image_height)))),
        min(image_width - 1, max(0, int(round(x_max * image_width)))),
        min(image_height - 1, max(0, int(round(y_max * image_height)))),
    )


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash_px: int = 18,
    gap_px: int = 10,
) -> None:
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta / length
    stride = max(2, int(dash_px) + int(gap_px))
    for offset in range(0, int(math.ceil(length)), stride):
        segment_end = min(length, offset + max(1, int(dash_px)))
        point_a = np.asarray(start, dtype=np.float64) + direction * offset
        point_b = np.asarray(start, dtype=np.float64) + direction * segment_end
        cv2.line(
            image,
            tuple(int(round(value)) for value in point_a),
            tuple(int(round(value)) for value in point_b),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_hand_roi_overlay(
    image: np.ndarray,
    roi: tuple[float, float, float, float] | list[float],
) -> tuple[int, int, int, int]:
    """Draw a persistent dashed CAM4 centroid-filter ROI without tinting it."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = normalized_roi_pixel_box(roi, width, height)
    segments = (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    )
    for start, end in segments:
        _draw_dashed_line(image, start, end, (8, 12, 18), 7)
        _draw_dashed_line(image, start, end, HAND_ROI_COLOR_BGR, 3)

    return x0, y0, x1, y1


def draw_hand_roi_header(
    image: np.ndarray,
    roi: tuple[float, float, float, float] | list[float],
    *,
    minimum_x: int = 0,
) -> tuple[str, int] | None:
    """Identify the ROI in free header space, away from scene annotations."""
    normalized_roi_pixel_box(roi, image.shape[1], image.shape[0])
    x_min, x_max, y_min, y_max = (float(value) for value in roi)
    labels = (
        f'HAND ROI  x:{x_min:.3f}-{x_max:.3f} y:{y_min:.3f}-{y_max:.3f}',
        'HAND ROI',
    )
    font_scale = 0.58
    thickness = 2
    right_margin = 14
    for label in labels:
        (text_width, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = image.shape[1] - right_margin - text_width
        if text_x >= minimum_x:
            draw_outlined_text(
                image, label, (text_x, 32), font_scale,
                HAND_ROI_COLOR_BGR, thickness)
            return label, text_x
    return None


@dataclass(frozen=True)
class ToolRoiOverlayConfig:
    """One camera's effective Tool-recognition acceptance polygon."""

    enabled: bool = False
    profile: str = 'none'
    polygon_norm_xy: tuple[float, ...] = ()


def validate_normalized_polygon(
    polygon_norm_xy: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Validate the same normalized polygon contract used by Tool filtering."""
    polygon = tuple(float(value) for value in polygon_norm_xy)
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError(
            'Tool ROI requires at least three normalized x/y points')
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in polygon
    ):
        raise ValueError('Tool ROI coordinates must be finite and in [0, 1]')
    return polygon


def load_tool_roi_profile(profile_file: str) -> ToolRoiOverlayConfig:
    """Load the exact Tool ROI YAML selected by the corresponding worker."""
    path_text = str(profile_file).strip()
    if not path_text:
        return ToolRoiOverlayConfig()
    path = Path(path_text).expanduser()
    try:
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
        parameters = document['/**']['ros__parameters']
        enabled = parameters['workspace_roi_enabled']
        if not isinstance(enabled, bool):
            raise ValueError('workspace_roi_enabled must be a boolean')
        profile = str(parameters['workspace_roi_profile']).strip()
        polygon = validate_normalized_polygon(
            parameters['workspace_roi_polygon_norm_xy'])
    except (OSError, TypeError, KeyError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f'invalid Tool ROI profile {path}: {exc}') from exc
    if not profile or profile == 'none':
        raise ValueError(f'invalid Tool ROI profile name in {path}')
    return ToolRoiOverlayConfig(
        enabled=enabled,
        profile=profile,
        polygon_norm_xy=polygon,
    )


def normalized_polygon_pixel_points(
    polygon_norm_xy: tuple[float, ...] | list[float],
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, int], ...]:
    """Map normalized Tool ROI vertices onto the original camera image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError('ROI image dimensions must be positive')
    polygon = validate_normalized_polygon(polygon_norm_xy)
    return tuple(
        (
            min(image_width - 1, max(0, int(round(x * (image_width - 1))))),
            min(image_height - 1, max(0, int(round(y * (image_height - 1))))),
        )
        for x, y in zip(polygon[0::2], polygon[1::2])
    )


def draw_tool_roi_overlay(
    image: np.ndarray,
    config: ToolRoiOverlayConfig,
) -> tuple[tuple[int, int], ...]:
    """Draw the enabled Tool-recognition polygon without tinting the scene."""
    if not config.enabled:
        return ()
    height, width = image.shape[:2]
    points = normalized_polygon_pixel_points(
        config.polygon_norm_xy, width, height)
    segments = tuple(zip(points, points[1:] + points[:1]))
    for start, end in segments:
        _draw_dashed_line(
            image, start, end, (8, 12, 18), 7, dash_px=24, gap_px=8)
        _draw_dashed_line(
            image, start, end, TOOL_ROI_COLOR_BGR, 3,
            dash_px=24, gap_px=8)

    anchor_x, anchor_y = min(points, key=lambda point: (point[1], point[0]))
    label_y = min(height - 12, max(68, anchor_y + 28))
    draw_outlined_text(
        image,
        f'TOOL ROI  {config.profile}',
        (max(12, anchor_x), label_y),
        0.58,
        TOOL_ROI_COLOR_BGR,
        2,
    )
    return points


def gesture_display_text(gesture: Any) -> str:
    """Return one compact, explicit MediaPipe gesture label."""
    try:
        hand_index = int(gesture.hand_index)
    except (AttributeError, TypeError, ValueError):
        hand_index = -1
    side = (
        str(gesture.handedness_label).strip()
        if bool(getattr(gesture, 'has_handedness', False))
        else 'Hand'
    )
    if not bool(getattr(gesture, 'has_classification', False)):
        return f'H{hand_index} {side} Unclassified'
    category = str(getattr(gesture, 'category_name', '')).strip() or 'Unclassified'
    try:
        score = float(gesture.score)
    except (AttributeError, TypeError, ValueError):
        score = math.nan
    if math.isfinite(score):
        return f'H{hand_index} {side} {category} {score:.2f}'
    return f'H{hand_index} {side} {category}'


def gesture_color(gesture: Any) -> tuple[int, int, int]:
    """Use a stable color for the winning canned category."""
    if not bool(getattr(gesture, 'has_classification', False)):
        return (165, 165, 165)
    return GESTURE_COLORS_BGR.get(
        str(getattr(gesture, 'category_name', '')).strip(), (235, 235, 235))


def palm_facing_display_text(facing: Any) -> str:
    """Return one explicit depth-backed palm-facing row."""
    try:
        hand_index = int(facing.hand_index)
    except (AttributeError, TypeError, ValueError):
        hand_index = -1
    side = (
        str(facing.handedness_label).strip()
        if bool(getattr(facing, 'has_handedness', False))
        else 'Hand'
    )
    if not bool(getattr(facing, 'has_facing', False)):
        return f'H{hand_index} {side} UNKNOWN'
    label = str(getattr(facing, 'facing_label', '')).strip()
    if label not in {'PALM_UP', 'PALM_DOWN', 'EDGE'}:
        return f'H{hand_index} {side} UNKNOWN'
    try:
        score = float(facing.palm_up_score)
    except (AttributeError, TypeError, ValueError):
        score = math.nan
    if math.isfinite(score):
        return f'H{hand_index} {side} {label} {score:+.2f}'
    return f'H{hand_index} {side} {label}'


def palm_facing_color(facing: Any) -> tuple[int, int, int]:
    if not bool(getattr(facing, 'has_facing', False)):
        return PALM_FACING_COLORS_BGR['UNKNOWN']
    return PALM_FACING_COLORS_BGR.get(
        str(getattr(facing, 'facing_label', '')).strip(),
        PALM_FACING_COLORS_BGR['UNKNOWN'],
    )


def joined_facing_by_hand_index(
    hand_message: HandKeypoints | None,
    facing_message: HandFacingArray | None,
) -> dict[int, Any]:
    """Return unambiguous facing rows for one exact keypoint source frame.

    ``hand_index`` is explicitly frame-local. A bounded source-time delta is
    sufficient for drawing independent layers, but it is not sufficient for
    associating one palm-facing result with one skeleton. Require identical
    source stamps and exactly one row for the index on both sides.
    """
    if hand_message is None or facing_message is None:
        return {}
    try:
        if stamp_ns(hand_message) != stamp_ns(facing_message):
            return {}
    except (AttributeError, TypeError, ValueError):
        return {}

    hand_counts: dict[int, int] = {}
    facing_rows: dict[int, list[Any]] = {}
    for hand in getattr(hand_message, 'hands', ()):
        try:
            hand_index = int(hand.hand_index)
        except (AttributeError, TypeError, ValueError):
            continue
        if hand_index < 0:
            continue
        hand_counts[hand_index] = hand_counts.get(hand_index, 0) + 1
    for facing in getattr(facing_message, 'hands', ()):
        try:
            hand_index = int(facing.hand_index)
        except (AttributeError, TypeError, ValueError):
            continue
        if hand_index < 0:
            continue
        facing_rows.setdefault(hand_index, []).append(facing)
    return {
        hand_index: rows[0]
        for hand_index, rows in facing_rows.items()
        if hand_counts.get(hand_index) == 1 and len(rows) == 1
    }


@dataclass
class LatestBase:
    message: CompressedImage | None = None
    image: np.ndarray | None = None
    source_stamp_ns: int | None = None
    freshness: Freshness = field(default_factory=lambda: Freshness(None))
    received: int = 0
    dropped: int = 0
    # Image subscription callbacks retain only the newest compressed packet.
    # JPEG decode happens on the compositor timer, so a slow render cannot
    # turn a FIFO callback backlog into seconds of visible video latency.
    pending_message: CompressedImage | None = None
    pending_received_monotonic: float | None = None
    pending_sequence: int = 0
    processed_sequence: int = 0


@dataclass
class LatestLayer:
    message: Any | None = None
    payload: Any | None = None
    source_stamp_ns: int | None = None
    freshness: Freshness = field(default_factory=lambda: Freshness(None))
    count: int = 0
    dropped: int = 0
    last_drop_signature: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class LayerDecision:
    state: str
    drawable: bool
    age_sec: float | None

@dataclass
class PanelOutput:
    source_header: Any | None = None
    published_at: float | None = None
    hz: float = 0.0
    bytes: int = 0
    width: int = 0
    height: int = 0


@dataclass
class PanelEncodeSlot:
    """One overwrite-only JPEG job slot for an operator panel."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    scheduled_signature: tuple[Any, ...] | None = None
    pending: tuple[Any, ...] | None = None
    running: bool = False
    coalesced: int = 0


@dataclass
class RightEePalmDisplayFilter:
    """Display-only temporal filter for the right end-effector palm HUD.

    This consumes only the current, source-aligned gesture/keypoint pair used
    by panel 4.  It intentionally does *not* publish or mutate the typed
    ``HandGestureArray``: the result is a visual debounce, never a control
    fact.  A close needs four consecutive, unique source-stamped observations;
    an explicit open or zero-hand observation releases immediately.  A short
    ``None`` grace is retained only after a confirmed display state so one
    dropped geometry frame cannot flicker the HUD.
    """

    closed_entry_frames: int = 4
    hold_sec: float = 0.25
    category: str = ''
    score: float = math.nan
    accepted_monotonic: float | None = None
    _last_source_stamp_ns: int | None = None
    _last_observed_category: str = ''
    _closed_candidate_count: int = 0

    def reset(self) -> tuple[str, float | None, bool]:
        """Clear all display evidence and return the fail-closed state."""
        self.category = ''
        self.score = math.nan
        self.accepted_monotonic = None
        self._last_source_stamp_ns = None
        self._last_observed_category = ''
        self._closed_candidate_count = 0
        return '', None, False

    def _clear_display(self, *, clear_window: bool) -> None:
        self.category = ''
        self.score = math.nan
        self.accepted_monotonic = None
        if clear_window:
            self._closed_candidate_count = 0

    def _none_result(self, now: float) -> tuple[str, float | None, bool]:
        """Return at most one short UI-only grace after confirmed evidence."""
        age_sec = (
            None if self.accepted_monotonic is None
            else max(0.0, now - self.accepted_monotonic)
        )
        if (
            self.category in {'Open_Palm', 'Closed_Fist'}
            and age_sec is not None
            and age_sec <= self.hold_sec
        ):
            return self.category, self.score, True
        # Do not let an old confirmed state seed a later close candidate.
        self._clear_display(clear_window=True)
        return '', None, False

    def _current_result(self, now: float) -> tuple[str, float | None, bool]:
        """Render a duplicate source frame without accidentally adding votes."""
        if self._last_observed_category in {'Open_Palm', 'Closed_Fist'}:
            return (
                self.category,
                self.score if self.category else None,
                False,
            )
        return self._none_result(now)

    def update(
        self,
        *,
        category: str,
        score: float | None,
        hand_present: bool,
        source_stamp_ns: int | None,
        now: float,
    ) -> tuple[str, float | None, bool]:
        """Update once per unique, source-aligned MediaPipe observation.

        ``source_stamp_ns`` is required.  If the gesture and skeleton cannot
        be joined to one source frame, panel 4 fails closed instead of mixing
        semantically unrelated evidence.
        """
        if not hand_present or source_stamp_ns is None:
            return self.reset()

        normalized = category if category in {'Open_Palm', 'Closed_Fist'} else ''
        try:
            finite_score = float(score)
        except (TypeError, ValueError):
            finite_score = math.nan
        if not math.isfinite(finite_score):
            finite_score = math.nan
        if self._last_source_stamp_ns is not None:
            if source_stamp_ns == self._last_source_stamp_ns:
                return self._current_result(now)
            if source_stamp_ns < self._last_source_stamp_ns:
                # A reordered source frame is not temporal evidence. Clearing
                # is safer than allowing a late packet to advance a candidate.
                return self.reset()
        self._last_source_stamp_ns = source_stamp_ns
        self._last_observed_category = normalized

        if normalized == 'Open_Palm':
            # An explicit open is current positive evidence.  It must never
            # be delayed behind a prior close candidate.
            self._closed_candidate_count = 0
            self.category = normalized
            self.score = finite_score
            self.accepted_monotonic = now
            return self.category, self.score, False

        if normalized == 'Closed_Fist':
            self._closed_candidate_count += 1
            if self.category == 'Closed_Fist':
                self.score = finite_score
                self.accepted_monotonic = now
                return self.category, self.score, False

            # Do not leave a stale OPEN label up while a close is being
            # evaluated. Four consecutive distinct source stamps prevent the
            # short half-curled H/I entry sequence from appearing as a fist.
            self._clear_display(clear_window=False)
            if self._closed_candidate_count >= self.closed_entry_frames:
                self.category = normalized
                self.score = finite_score
                self.accepted_monotonic = now
                return self.category, self.score, False
            return '', None, False

        # A ``None`` cannot advance a close candidate. It may only bridge one
        # geometry rejection *after* CLOSED/OPEN has already met its entry
        # rule, for the existing short visual grace.
        self._closed_candidate_count = 0
        return self._none_result(now)


class FinalOverlayCompositor(Node):
    """Publish the operator 2x2 JPEG plus source-aligned per-view JPEGs."""

    def __init__(self, *, context: Context | None = None) -> None:
        super().__init__('final_overlay_compositor', context=context)
        self._declare_parameters()
        self._read_parameters()
        self._base = {camera: LatestBase() for camera in CAMERAS}
        # Suction stays outside the strict CAM3/CAM4 status-v1 camera schema.
        self._suction_base = LatestBase()
        self._suction_overlay = LatestBase()
        self._suction_mask = LatestLayer()
        # Panel 4 is an RGB-only end-effector view.  It deliberately stays
        # outside status-v1, whose strict camera keys are CAM3/CAM4 only.
        self._right_ee_base = LatestBase()
        self._right_ee_hand = LatestLayer()
        self._right_ee_gesture = LatestLayer()
        self._right_ee_palm_filter = RightEePalmDisplayFilter(
            hold_sec=self._right_ee_palm_hold_sec,
        )
        self._layers = {
            camera: {name: LatestLayer() for name in LAYER_NAMES}
            for camera in CAMERAS
        }
        # Gesture and facing are intentionally kept outside the strict
        # final-overlay v1 status-layer schema. They are freshness/source-stamp
        # gated for rendering, but adding public layer keys would break strict
        # consumers of the existing tool/pose/hand/blood contract.
        self._gesture = LatestLayer()
        self._facing = LatestLayer()
        # This is intentionally outside ``LAYER_NAMES`` and status-v1. It is
        # a derived, source-stamped display fact only; the public CAM4 layer
        # contract remains tool/pose/hand/blood.
        self._cam4_palm_pose = LatestLayer()
        self._camera_info: dict[str, CameraInfo | None] = {camera: None for camera in CAMERAS}
        self._last_signature: tuple[Any, ...] | None = None
        self._last_output_at: float | None = None
        self._last_output_source_header: Any | None = None
        self._output_hz = 0.0
        self._last_output_bytes = 0
        self._last_output_width = 0
        self._last_output_height = 0
        # Per-view output is deliberately independent from the legacy 2x2
        # Debug stream.  Each signature is scoped to exactly one rendered
        # panel so a CAM3 callback cannot re-emit an old CAM4/suction/EE JPEG.
        self._last_panel_signatures: dict[str, tuple[Any, ...] | None] = {
            panel: None for panel in PANEL_NAMES
        }
        self._last_panel_source_headers: dict[str, Any | None] = {
            panel: None for panel in PANEL_NAMES
        }
        # The legacy 2x2 stream has its own metrics below.  When it is
        # intentionally disabled, status still needs to identify an actual
        # encoded native panel without fabricating a composite frame.
        self._last_panel_outputs: dict[str, PanelOutput] = {
            panel: PanelOutput() for panel in PANEL_NAMES
        }
        self._panel_encode_slots = {
            panel: PanelEncodeSlot() for panel in PANEL_NAMES
        }
        self._panel_encode_executor = (
            ThreadPoolExecutor(
                max_workers=self._panel_encode_workers,
                thread_name_prefix='final-overlay-jpeg',
            )
            if self._async_panel_encoding and self._enable_per_view_output
            else None
        )

        # The legacy 2x2 topic is optional at runtime.  The operator viewer
        # normally subscribes directly to the four native-resolution outputs
        # below, so it does not need a second lossy image transport.
        self._image_publisher = (
            self.create_publisher(
                CompressedImage, self._output_topic, image_reader_qos())
            if self._enable_composite_output else None
        )
        self._panel_image_publishers = {
            panel: self.create_publisher(
                CompressedImage, topic, image_reader_qos())
            for panel, topic in self._panel_output_topics.items()
        } if self._enable_per_view_output else {}
        self._status_publisher = self.create_publisher(
            String, self._status_topic, status_qos())
        # ``Node`` owns ``_subscriptions`` internally.  Keep application
        # references under a non-rclpy name so Node.destroy_node() can tear
        # down waitables safely.
        self._perception_subscriptions: list[Any] = []
        for camera in CAMERAS:
            prefix = f'/perception/ingress/{camera}'
            self._perception_subscriptions.append(self.create_subscription(
                CompressedImage, self._base_topics[camera],
                lambda message, cam=camera: self._on_base(cam, message), image_reader_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                CameraInfo, self._camera_info_topics[camera],
                lambda message, cam=camera: self._on_camera_info(cam, message), camera_info_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                ToolObservation2DArray, self._tool_topics[camera],
                lambda message, cam=camera: self._on_result(cam, 'tool', message), result_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                ToolPoseArray, self._pose_topics[camera],
                lambda message, cam=camera: self._on_result(cam, 'pose', message), result_qos()))
            if camera == 'cam_4' and self._enable_hand:
                self._perception_subscriptions.append(self.create_subscription(
                    HandKeypoints, self._hand_topic,
                    lambda message: self._on_result('cam_4', 'hand', message), result_qos()))
            if camera == 'cam_4' and self._enable_cam4_palm_pose:
                self._perception_subscriptions.append(self.create_subscription(
                    PoseStamped, self._cam4_palm_pose_topic,
                    self._on_cam4_palm_pose, result_qos()))
            if camera == 'cam_4' and self._enable_gesture:
                self._perception_subscriptions.append(self.create_subscription(
                    HandGestureArray, self._gesture_topic,
                    self._on_gesture, result_qos()))
            if camera == 'cam_4' and self._enable_facing:
                self._perception_subscriptions.append(self.create_subscription(
                    HandFacingArray, self._facing_topic,
                    self._on_facing, result_qos()))
            if camera == 'cam_4' and self._enable_blood:
                self._perception_subscriptions.append(self.create_subscription(
                    Image, self._blood_mask_topic, self._on_blood_mask, result_qos()))
        if self._enable_suction_panel:
            self._perception_subscriptions.append(self.create_subscription(
                CompressedImage, self._suction_color_topic,
                self._on_suction_base, image_reader_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                CompressedImage, self._suction_blood_overlay_topic,
                self._on_suction_overlay, image_reader_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                Image, self._suction_blood_mask_topic,
                self._on_suction_mask, result_qos()))
        if self._enable_right_ee_panel:
            self._perception_subscriptions.append(self.create_subscription(
                CompressedImage, self._right_ee_color_topic,
                self._on_right_ee_base, image_reader_qos()))
            if self._enable_right_ee_hand:
                self._perception_subscriptions.append(self.create_subscription(
                    HandKeypoints, self._right_ee_hand_topic,
                    self._on_right_ee_hand, result_qos()))
            self._perception_subscriptions.append(self.create_subscription(
                HandGestureArray, self._right_ee_gesture_topic,
                self._on_right_ee_gesture, result_qos()))

        self.create_timer(1.0 / self._output_rate_hz, self._publish_if_current)
        self.create_timer(1.0 / self._status_rate_hz, self._publish_status)
        self.get_logger().info(
            f'final overlay reads only local ingress: cam3={self._base_topics["cam_3"]}, '
            f'cam4={self._base_topics["cam_4"]}; legacy_2x2='
            f'{self._output_topic if self._enable_composite_output else "disabled"}; '
            f'per_view_outputs={self._panel_output_topics if self._enable_per_view_output else "disabled"}; '
            f'per_view_native={self._per_view_native_resolution}; '
            f'panel_encoding={"latest-parallel" if self._panel_encode_executor else "inline"}; '
            f'panel_encode_workers={self._panel_encode_workers}; '
            f'opencv_threads={cv2.getNumThreads()}; '
            f'suction={self._suction_color_topic}; '
            f'suction_blood={self._suction_blood_overlay_topic}; '
            f'suction_blood_mask={self._suction_blood_mask_topic}; '
            f'right_ee={self._right_ee_color_topic}; '
            f'right_ee_hand={self._right_ee_hand_topic}; '
            f'right_ee_gesture={self._right_ee_gesture_topic}; '
            f'gesture={self._gesture_topic}; '
            f'facing={self._facing_topic}; '
            f'cam4_palm_humanoid={self._cam4_palm_pose_topic}; '
            f'tool_roi={self._tool_roi_log_summary()}; '
            f'cam4_hand_roi={self._cam4_hand_roi if self._show_cam4_hand_roi else "hidden"}; '
            f'layer_age<={self._max_layer_age_sec:.3f}s; '
            f'source_delta<={self._max_source_delta_ns / 1_000_000:.1f}ms')

    def _tool_roi_log_summary(self) -> str:
        return ','.join(
            f'{camera}:{self._tool_rois[camera].profile}'
            for camera in CAMERAS
        )

    def _declare_parameters(self) -> None:
        for camera in CAMERAS:
            prefix = f'/perception/ingress/{camera}'
            output = f'/perception/{camera}/tool'
            self.declare_parameter(
                f'{camera}_color_topic', f'{prefix}/color/image_raw/compressed')
            self.declare_parameter(
                f'{camera}_camera_info_topic', f'{prefix}/color/camera_info')
            self.declare_parameter(
                f'{camera}_tool_topic', f'{output}/observations')
            self.declare_parameter(
                f'{camera}_pose_topic', f'{output}/poses')
            self.declare_parameter(f'{camera}_tool_roi_profile_file', '')
        self.declare_parameter('hand_topic', '/perception/cam_4/hand/keypoints')
        self.declare_parameter(
            'cam4_palm_pose_topic',
            '/perception/cam_4/hand/palm_pose_humanoid')
        self.declare_parameter('gesture_topic', '/perception/cam_4/hand/gestures')
        self.declare_parameter('facing_topic', '/perception/cam_4/hand/facing')
        self.declare_parameter('blood_mask_topic', '/perception/cam_4/blood/mask')
        self.declare_parameter(
            'suction_color_topic',
            '/perception/ingress/suction/color/image_raw/compressed')
        self.declare_parameter(
            'suction_blood_overlay_topic',
            '/perception/suction/blood/overlay/compressed')
        self.declare_parameter(
            'suction_blood_mask_topic',
            '/perception/suction/blood/mask')
        self.declare_parameter(
            'right_ee_color_topic',
            '/perception/ingress/right_ee/color/image_raw/compressed')
        self.declare_parameter(
            'right_ee_hand_topic', '/perception/right_ee/hand/keypoints')
        self.declare_parameter(
            'right_ee_gesture_topic', '/perception/right_ee/hand/gestures')
        self.declare_parameter('enable_hand', True)
        self.declare_parameter('enable_cam4_palm_pose', True)
        self.declare_parameter('enable_gesture', True)
        self.declare_parameter('enable_facing', True)
        self.declare_parameter('enable_blood', True)
        self.declare_parameter('enable_suction_panel', True)
        self.declare_parameter('enable_right_ee_panel', True)
        self.declare_parameter('enable_right_ee_hand', True)
        self.declare_parameter('show_cam4_hand_roi', True)
        self.declare_parameter('region_x_min', 0.0)
        self.declare_parameter('region_x_max', 1.0)
        self.declare_parameter('region_y_min', 0.0)
        self.declare_parameter('region_y_max', 1.0)
        self.declare_parameter('output_topic', '/perception/debug/final_overlay/compressed')
        self.declare_parameter('status_topic', '/perception/debug/final_overlay/status')
        # Keep this enabled by default for backward compatibility.  The
        # deployed operator view disables it and subscribes directly to the
        # four per-view final overlays, avoiding a screen-only image topic.
        self.declare_parameter('enable_composite_output', True)
        # Root-level ``overlay`` is the public operator image for a view.
        # Keep worker-stage topics beneath ``tool``, ``hand`` and ``blood``
        # separate: they are not pixel-identical to this compositor output.
        self.declare_parameter('enable_per_view_output', True)
        self.declare_parameter(
            'cam_3_overlay_output_topic', '/perception/cam_3/overlay/compressed')
        self.declare_parameter(
            'cam_4_overlay_output_topic', '/perception/cam_4/overlay/compressed')
        self.declare_parameter(
            'suction_overlay_output_topic', '/perception/suction/overlay/compressed')
        self.declare_parameter(
            'right_ee_overlay_output_topic',
            '/perception/right_ee/overlay/compressed')
        # Per-view overlays preserve the ingress image dimensions.  The
        # 2x2 legacy stream, when enabled, may still use ``panel_*`` sizing.
        self.declare_parameter('per_view_native_resolution', True)
        self.declare_parameter('async_panel_encoding', True)
        self.declare_parameter('panel_encode_workers', 4)
        self.declare_parameter('opencv_num_threads', 1)
        self.declare_parameter('output_rate_hz', 10.0)
        self.declare_parameter('status_rate_hz', 2.0)
        self.declare_parameter('max_base_age_sec', 1.0)
        self.declare_parameter('max_layer_age_sec', 1.5)
        # Gesture labels represent a fast human intent cue.  Do not reuse the
        # much wider Tool inference window or a stopped hand will remain on
        # screen as a seemingly current command.
        self.declare_parameter('max_gesture_age_sec', 0.30)
        self.declare_parameter('max_gesture_source_delta_ms', 250.0)
        # The right-EE skeleton comes from the same RGB-only MediaPipe result
        # as its palm label.  Keep its display window equally short and
        # source-stamp aligned so an old hand skeleton cannot trail a newer
        # wrist/camera frame.
        self.declare_parameter('max_right_ee_hand_age_sec', 0.30)
        self.declare_parameter('max_right_ee_hand_source_delta_ms', 250.0)
        # The display-only close entry rule is intentionally fixed at four
        # consecutive source-stamped observations. Keep the short ``None``
        # hold capped: this must never become robot-intent state.
        self.declare_parameter('right_ee_palm_hold_sec', 0.25)
        self.declare_parameter('max_facing_age_sec', 0.35)
        self.declare_parameter('max_facing_source_delta_ms', 450.0)
        # The palm transform is computed directly from the displayed Hand
        # frame. Keep its visual join tight; an old humanoid coordinate must
        # never be rendered on a newer skeleton.
        self.declare_parameter('max_cam4_palm_pose_age_sec', 0.30)
        self.declare_parameter('max_cam4_palm_pose_source_delta_ms', 250.0)
        # The GPU workers publish a source-stamped result roughly once per
        # second.  Comparing that result to a 10-15 Hz *current* base with a
        # 150 ms window made every valid tool result disappear between worker
        # callbacks.  A 1.8 s source window covers the measured inference plus
        # one result period; ``max_layer_age_sec`` remains the independent
        # receiver-time fail-closed gate, so a stopped worker is never held
        # indefinitely.  Rendering still starts from a fresh base copy and
        # therefore cannot accumulate raster ghost trails.
        self.declare_parameter('max_source_delta_ms', 1800.0)
        self.declare_parameter('panel_width', 960)
        self.declare_parameter('panel_height', 540)
        # Four panels form a 1920x1080 operator display at the defaults.
        # Preserve fine instrument edges and small text through the final
        # compressed transport instead of introducing another lossy artifact.
        self.declare_parameter('jpeg_quality', 95)
        # Overlay drawing necessarily creates a new compressed payload. Use
        # the maximum JPEG quality for the native view topics so no extra
        # downsampling or practical display-quality loss is introduced.
        self.declare_parameter('per_view_jpeg_quality', 100)
        self.declare_parameter('pose_axis_length_m', 0.05)
        self.declare_parameter('palm_axis_length_m', 0.08)

    def _read_parameters(self) -> None:
        def text(name: str) -> str:
            value = str(self.get_parameter(name).value).strip()
            if not value.startswith('/'):
                raise ValueError(f'{name} must be an absolute ROS topic')
            return value

        self._base_topics = {camera: text(f'{camera}_color_topic') for camera in CAMERAS}
        self._camera_info_topics = {
            camera: text(f'{camera}_camera_info_topic') for camera in CAMERAS
        }
        self._tool_topics = {camera: text(f'{camera}_tool_topic') for camera in CAMERAS}
        self._pose_topics = {camera: text(f'{camera}_pose_topic') for camera in CAMERAS}
        self._hand_topic = text('hand_topic')
        self._cam4_palm_pose_topic = text('cam4_palm_pose_topic')
        self._gesture_topic = text('gesture_topic')
        self._facing_topic = text('facing_topic')
        self._blood_mask_topic = text('blood_mask_topic')
        self._suction_color_topic = text('suction_color_topic')
        self._suction_blood_overlay_topic = text('suction_blood_overlay_topic')
        self._suction_blood_mask_topic = text('suction_blood_mask_topic')
        self._right_ee_color_topic = text('right_ee_color_topic')
        self._right_ee_hand_topic = text('right_ee_hand_topic')
        self._right_ee_gesture_topic = text('right_ee_gesture_topic')
        self._enable_hand = bool(self.get_parameter('enable_hand').value)
        self._enable_cam4_palm_pose = bool(
            self.get_parameter('enable_cam4_palm_pose').value)
        self._enable_gesture = bool(self.get_parameter('enable_gesture').value)
        self._enable_facing = bool(self.get_parameter('enable_facing').value)
        self._enable_blood = bool(self.get_parameter('enable_blood').value)
        self._enable_suction_panel = bool(
            self.get_parameter('enable_suction_panel').value)
        self._enable_right_ee_panel = bool(
            self.get_parameter('enable_right_ee_panel').value)
        self._enable_right_ee_hand = bool(
            self.get_parameter('enable_right_ee_hand').value)
        self._tool_rois = {
            camera: load_tool_roi_profile(str(
                self.get_parameter(
                    f'{camera}_tool_roi_profile_file').value
            ))
            for camera in CAMERAS
        }
        self._show_cam4_hand_roi = bool(
            self.get_parameter('show_cam4_hand_roi').value)
        self._cam4_hand_roi = tuple(float(
            self.get_parameter(name).value) for name in (
                'region_x_min', 'region_x_max',
                'region_y_min', 'region_y_max'))
        # Validate at startup with the same normalized contract used by Hand.
        normalized_roi_pixel_box(self._cam4_hand_roi, 2, 2)
        self._output_topic = text('output_topic')
        self._status_topic = text('status_topic')
        self._enable_composite_output = bool(
            self.get_parameter('enable_composite_output').value)
        self._enable_per_view_output = bool(
            self.get_parameter('enable_per_view_output').value)
        self._panel_output_topics = {
            panel: text(f'{panel}_overlay_output_topic')
            for panel in PANEL_NAMES
        }
        self._output_rate_hz = max(1.0, min(30.0, float(self.get_parameter('output_rate_hz').value)))
        self._status_rate_hz = max(0.2, min(10.0, float(self.get_parameter('status_rate_hz').value)))
        self._max_base_age_sec = max(0.05, float(self.get_parameter('max_base_age_sec').value))
        self._max_layer_age_sec = max(0.05, float(self.get_parameter('max_layer_age_sec').value))
        self._max_gesture_age_sec = max(
            0.05, float(self.get_parameter('max_gesture_age_sec').value))
        self._max_gesture_source_delta_ns = max(
            0,
            int(float(self.get_parameter('max_gesture_source_delta_ms').value) * 1_000_000),
        )
        self._max_right_ee_hand_age_sec = max(
            0.05, float(self.get_parameter('max_right_ee_hand_age_sec').value))
        self._max_right_ee_hand_source_delta_ns = max(
            0,
            int(float(
                self.get_parameter('max_right_ee_hand_source_delta_ms').value
            ) * 1_000_000),
        )
        self._right_ee_palm_hold_sec = min(
            0.25,
            max(0.0, float(self.get_parameter('right_ee_palm_hold_sec').value)),
        )
        self._max_facing_age_sec = max(
            0.05, float(self.get_parameter('max_facing_age_sec').value))
        self._max_facing_source_delta_ns = max(
            0,
            int(float(self.get_parameter('max_facing_source_delta_ms').value) * 1_000_000),
        )
        self._max_cam4_palm_pose_age_sec = max(
            0.05,
            float(self.get_parameter('max_cam4_palm_pose_age_sec').value),
        )
        self._max_cam4_palm_pose_source_delta_ns = max(
            0,
            int(float(
                self.get_parameter('max_cam4_palm_pose_source_delta_ms').value
            ) * 1_000_000),
        )
        self._max_source_delta_ns = max(
            0, int(float(self.get_parameter('max_source_delta_ms').value) * 1_000_000))
        self._panel_width = max(160, int(self.get_parameter('panel_width').value))
        self._panel_height = max(90, int(self.get_parameter('panel_height').value))
        self._jpeg_quality = max(20, min(100, int(self.get_parameter('jpeg_quality').value)))
        self._per_view_native_resolution = bool(
            self.get_parameter('per_view_native_resolution').value)
        self._async_panel_encoding = bool(
            self.get_parameter('async_panel_encoding').value)
        self._panel_encode_workers = max(
            1, min(4, int(self.get_parameter('panel_encode_workers').value)))
        self._opencv_num_threads = max(
            1, min(8, int(self.get_parameter('opencv_num_threads').value)))
        cv2.setUseOptimized(True)
        cv2.setNumThreads(self._opencv_num_threads)
        self._per_view_jpeg_quality = max(
            20, min(100, int(self.get_parameter('per_view_jpeg_quality').value)))
        self._pose_axis_length_m = max(0.001, float(self.get_parameter('pose_axis_length_m').value))
        self._palm_axis_length_m = max(
            0.01, float(self.get_parameter('palm_axis_length_m').value))

    def _on_base(self, camera: str, message: CompressedImage) -> None:
        self._stage_latest_base(self._base[camera], message)

    def _on_suction_base(self, message: CompressedImage) -> None:
        self._update_external_image(self._suction_base, message)

    def _on_suction_overlay(self, message: CompressedImage) -> None:
        # The final suction view renders the typed mask over current raw RGB;
        # decoding the worker's older full-raster overlay was pure overhead.
        # Retain its source/freshness metadata for diagnostics without ever
        # decoding pixels that are deliberately not displayed.
        state = self._suction_overlay
        state.received += 1
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError, ValueError):
            state.dropped += 1
            return
        state.message = message
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())

    def _on_right_ee_base(self, message: CompressedImage) -> None:
        self._update_external_image(self._right_ee_base, message)

    def _on_right_ee_hand(self, message: HandKeypoints) -> None:
        """Retain the newest 2-D skeleton without ever blocking RGB redraw."""
        state = self._right_ee_hand
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = len(message.hands)

    def _on_right_ee_gesture(self, message: HandGestureArray) -> None:
        state = self._right_ee_gesture
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = len(message.hands)

    def _on_suction_mask(self, message: Image) -> None:
        state = self._suction_mask
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        mask = decode_binary_mask(message)
        if mask is None:
            state.dropped += 1
            return
        state.message = message
        state.payload = mask
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = int(np.count_nonzero(mask))

    @staticmethod
    def _update_external_image(state: LatestBase, message: CompressedImage) -> None:
        """Retain one external JPEG; decode it with the next render tick."""
        FinalOverlayCompositor._stage_latest_base(state, message)

    @staticmethod
    def _stage_latest_base(state: LatestBase, message: CompressedImage) -> None:
        """Keep only the current ingress frame while rendering stays busy.

        This node is spun by one rclpy executor, so replacing the pending
        message here is atomic with respect to its render timer.  The counter
        makes the intentional latest-only coalescing visible through the
        existing base.dropped status field.
        """
        state.received += 1
        state.pending_message = message
        state.pending_received_monotonic = time.monotonic()
        state.pending_sequence += 1

    @staticmethod
    def _drain_latest_base(state: LatestBase) -> bool:
        """Decode at most one newest JPEG and discard superseded arrivals."""
        message = state.pending_message
        sequence = state.pending_sequence
        if message is None or sequence <= state.processed_sequence:
            return False
        received_monotonic = state.pending_received_monotonic
        state.pending_message = None
        state.pending_received_monotonic = None
        state.dropped += max(0, sequence - state.processed_sequence - 1)
        state.processed_sequence = sequence

        image = decode_jpeg(message)
        if image is None:
            state.dropped += 1
            return True
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError, ValueError):
            state.dropped += 1
            return True
        state.message = message
        state.image = image
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(received_monotonic)
        return True

    def _drain_latest_bases(self) -> None:
        """Move all latest image arrivals into renderable native frames."""
        states = [
            *self._base.values(),
            getattr(self, '_suction_base', None),
            getattr(self, '_right_ee_base', None),
        ]
        for state in states:
            if not isinstance(state, LatestBase):
                continue
            self._drain_latest_base(state)

    def _on_camera_info(self, camera: str, message: CameraInfo) -> None:
        self._camera_info[camera] = message

    def _on_result(self, camera: str, layer: str, message: Any) -> None:
        state = self._layers[camera][layer]
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        if layer == 'tool':
            state.count = len(message.instances)
        elif layer == 'pose':
            state.count = len(message.tools)
        elif layer == 'hand':
            state.count = len(message.hands)

    def _on_blood_mask(self, message: Image) -> None:
        state = self._layers['cam_4']['blood']
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        mask = decode_binary_mask(message)
        if mask is None:
            state.dropped += 1
            return
        state.message = message
        state.payload = mask
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = int(np.count_nonzero(mask))

    def _on_gesture(self, message: HandGestureArray) -> None:
        state = self._gesture
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = len(message.hands)

    def _on_facing(self, message: HandFacingArray) -> None:
        state = self._facing
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = len(message.hands)

    def _on_cam4_palm_pose(self, message: PoseStamped) -> None:
        """Retain a derived humanoid pose only as a visual observation layer."""
        state = self._cam4_palm_pose
        try:
            source_stamp = stamp_ns(message)
        except (AttributeError, TypeError):
            state.dropped += 1
            return
        state.message = message
        state.payload = None
        state.source_stamp_ns = source_stamp
        state.freshness = Freshness(time.monotonic())
        state.count = 1

    def _base_state(self, camera: str, now: float) -> tuple[str, float | None]:
        base = self._base[camera]
        age = base.freshness.age(now)
        return (
            freshness_state(
                has_value=base.message is not None and base.image is not None,
                age_sec=age,
                max_age_sec=self._max_base_age_sec,
            ), age,
        )

    def _layer_decision(
        self, camera: str, layer: str, base_state: str, now: float
    ) -> LayerDecision:
        disabled = camera == 'cam_3' and layer in {'hand', 'blood'}
        disabled = disabled or (camera == 'cam_4' and layer == 'hand' and not self._enable_hand)
        disabled = disabled or (camera == 'cam_4' and layer == 'blood' and not self._enable_blood)
        state = self._layers[camera][layer]
        age = state.freshness.age(now)
        public_state = freshness_state(
            has_value=state.message is not None,
            age_sec=age,
            max_age_sec=self._max_layer_age_sec,
            disabled=disabled,
        )
        drawable = layer_is_drawable(
            base_stamp_ns=self._base[camera].source_stamp_ns,
            layer_stamp_ns=state.source_stamp_ns,
            base_state=base_state,
            layer_state=public_state,
            max_source_delta_ns=self._max_source_delta_ns,
        )
        if public_state == 'live' and not drawable:
            signature = (
                self._base[camera].source_stamp_ns, state.source_stamp_ns, base_state,
            )
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            public_state = 'stale'
        return LayerDecision(state=public_state, drawable=drawable, age_sec=age)

    def _camera_context(self, camera: str, now: float) -> dict[str, Any]:
        base_state, base_age = self._base_state(camera, now)
        layers = {
            name: self._layer_decision(camera, name, base_state, now)
            for name in LAYER_NAMES
        }
        gesture = self._gesture_decision(camera, base_state, now)
        facing = self._facing_decision(camera, base_state, now)
        cam4_palm_pose = self._cam4_palm_pose_decision(
            camera, base_state, now)
        return {
            'base_state': base_state,
            'base_age': base_age,
            'layers': layers,
            'gesture': gesture,
            'facing': facing,
            'cam4_palm_pose': cam4_palm_pose,
        }

    def _gesture_decision(
        self, camera: str, base_state: str, now: float,
    ) -> LayerDecision:
        state = self._gesture
        disabled = camera != 'cam_4' or not self._enable_gesture
        age = state.freshness.age(now)
        public_state = freshness_state(
            has_value=state.message is not None,
            age_sec=age,
            max_age_sec=self._max_gesture_age_sec,
            disabled=disabled,
        )
        drawable = layer_is_drawable(
            base_stamp_ns=self._base[camera].source_stamp_ns,
            layer_stamp_ns=state.source_stamp_ns,
            base_state=base_state,
            layer_state=public_state,
            max_source_delta_ns=self._max_gesture_source_delta_ns,
        )
        if public_state == 'live' and not drawable:
            signature = (
                self._base[camera].source_stamp_ns,
                state.source_stamp_ns,
                base_state,
            )
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            public_state = 'stale'
        return LayerDecision(state=public_state, drawable=drawable, age_sec=age)

    def _facing_decision(
        self, camera: str, base_state: str, now: float,
    ) -> LayerDecision:
        state = self._facing
        disabled = camera != 'cam_4' or not self._enable_facing
        age = state.freshness.age(now)
        public_state = freshness_state(
            has_value=state.message is not None,
            age_sec=age,
            max_age_sec=self._max_facing_age_sec,
            disabled=disabled,
        )
        drawable = layer_is_drawable(
            base_stamp_ns=self._base[camera].source_stamp_ns,
            layer_stamp_ns=state.source_stamp_ns,
            base_state=base_state,
            layer_state=public_state,
            max_source_delta_ns=self._max_facing_source_delta_ns,
        )
        if public_state == 'live' and not drawable:
            signature = (
                self._base[camera].source_stamp_ns,
                state.source_stamp_ns,
                base_state,
            )
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            public_state = 'stale'
        return LayerDecision(state=public_state, drawable=drawable, age_sec=age)

    def _cam4_palm_pose_decision(
        self, camera: str, base_state: str, now: float,
    ) -> LayerDecision:
        """Gate the derived humanoid coordinate independently of status-v1."""
        state = getattr(self, '_cam4_palm_pose', LatestLayer())
        disabled = camera != 'cam_4' or not getattr(
            self, '_enable_cam4_palm_pose', False)
        age = state.freshness.age(now)
        public_state = freshness_state(
            has_value=state.message is not None,
            age_sec=age,
            max_age_sec=getattr(self, '_max_cam4_palm_pose_age_sec', 0.30),
            disabled=disabled,
        )
        drawable = layer_is_drawable(
            base_stamp_ns=self._base[camera].source_stamp_ns,
            layer_stamp_ns=state.source_stamp_ns,
            base_state=base_state,
            layer_state=public_state,
            max_source_delta_ns=getattr(
                self, '_max_cam4_palm_pose_source_delta_ns', 250_000_000),
        )
        if public_state == 'live' and not drawable:
            signature = (
                self._base[camera].source_stamp_ns,
                state.source_stamp_ns,
                base_state,
            )
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            public_state = 'stale'
        return LayerDecision(state=public_state, drawable=drawable, age_sec=age)

    def _camera_panel_signature(
        self, camera: str, context: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Drive publication from the newest base frame, not layer churn.

        Tool/Hand results often arrive between two RGB frames. Publishing
        again for each of those callbacks repeats the same large JPEG with an
        older camera timestamp and can exceed the camera rate substantially.
        The next RGB frame is at most one camera period away and renders the
        newest drawable layers, preserving all annotations while keeping the
        operator stream latest-only.
        """
        base_state = context['base_state']
        if base_state != 'live':
            return ('base', base_state)
        return ('live', self._base[camera].source_stamp_ns)

    def _suction_panel_signature(
        self, context: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Publish one newest annotated suction panel per source RGB frame."""
        if context['base_state'] != 'live':
            return ('base', context['base_state'])
        return (
            'live',
            getattr(self, '_suction_base', LatestBase()).source_stamp_ns,
        )

    def _right_ee_panel_signature(
        self, context: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Publish one newest annotated right-EE panel per source RGB frame."""
        if context['base_state'] != 'live':
            return ('base', context['base_state'])
        return (
            'live',
            getattr(self, '_right_ee_base', LatestBase()).source_stamp_ns,
        )

    def _panel_source_header(
        self,
        panel: str,
        contexts: dict[str, dict[str, Any]],
        suction_context: dict[str, Any],
        right_ee_context: dict[str, Any],
    ) -> Any | None:
        """Return a panel's own live ingress header, never a global anchor."""
        state: LatestBase | None
        if panel in CAMERAS:
            state = self._base[panel] if contexts[panel]['base_state'] == 'live' else None
        elif panel == 'suction':
            state = suction_context['selected']
        elif panel == 'right_ee':
            state = right_ee_context['selected']
        else:
            raise ValueError(f'unknown operator panel: {panel}')
        return getattr(getattr(state, 'message', None), 'header', None)

    def _publish_panel_if_changed(
        self,
        panel: str,
        image: np.ndarray,
        signature: tuple[Any, ...],
        source_header: Any | None,
    ) -> CompressedImage | None:
        """Publish one final rendered panel only for a new visible state.

        A stale/missing clearing panel intentionally retains the last header
        from *this same panel*.  Calling a timer timestamp a camera frame here
        would make downstream synchronizers believe a nonexistent RGB sample
        had just arrived.
        """
        publisher = getattr(self, '_panel_image_publishers', {}).get(panel)
        if publisher is None:
            return
        signatures = getattr(self, '_last_panel_signatures', {})
        if signatures.get(panel) == signature:
            return
        headers = getattr(self, '_last_panel_source_headers', {})
        header = source_header if source_header is not None else headers.get(panel)
        # There is no truthful header before the first source-backed panel.
        # Suppress startup placeholders rather than inventing a timer stamp.
        if header is None:
            return
        success, encoded = cv2.imencode(
            '.jpg', image, [
                int(cv2.IMWRITE_JPEG_QUALITY),
                getattr(self, '_per_view_jpeg_quality', self._jpeg_quality),
            ])
        if not success:
            self.get_logger().warning(
                f'could not encode final {panel} overlay JPEG')
            return
        output = CompressedImage()
        output.header = header
        output.format = 'jpeg'
        output.data = encoded.tobytes()
        publisher.publish(output)
        signatures[panel] = signature
        headers[panel] = output.header
        self._last_panel_signatures = signatures
        self._last_panel_source_headers = headers
        return output

    def _panel_target_signature(self, panel: str) -> tuple[Any, ...] | None:
        """Return the newest published-or-scheduled state for one panel."""
        executor = getattr(self, '_panel_encode_executor', None)
        slots = getattr(self, '_panel_encode_slots', {})
        slot = slots.get(panel)
        if executor is None or not isinstance(slot, PanelEncodeSlot):
            return getattr(self, '_last_panel_signatures', {}).get(panel)
        with slot.lock:
            return slot.scheduled_signature

    def _schedule_panel_encode(
        self,
        panel: str,
        image: np.ndarray,
        signature: tuple[Any, ...],
        source_header: Any | None,
    ) -> bool:
        """Schedule only the newest JPEG job; never build an encoder FIFO."""
        executor = getattr(self, '_panel_encode_executor', None)
        slot = getattr(self, '_panel_encode_slots', {}).get(panel)
        if executor is None or not isinstance(slot, PanelEncodeSlot):
            return False
        with slot.lock:
            if slot.scheduled_signature == signature:
                return True
            slot.scheduled_signature = signature
            if slot.pending is not None:
                slot.coalesced += 1
            slot.pending = (image, signature, source_header)
            if slot.running:
                return True
            slot.running = True
        executor.submit(self._panel_encode_loop, panel)
        return True

    def _panel_encode_loop(self, panel: str) -> None:
        """Drain one per-panel latest slot on a dedicated JPEG worker."""
        slot = self._panel_encode_slots[panel]
        while True:
            with slot.lock:
                job = slot.pending
                slot.pending = None
                if job is None:
                    slot.running = False
                    return
            image, signature, source_header = job
            output = self._publish_panel_if_changed(
                panel, image, signature, source_header)
            if output is not None:
                self._record_panel_output(
                    panel, output, image, time.monotonic())
                continue
            # Allow a retry if this exact job failed and no newer source state
            # has replaced it while JPEG encoding was in progress.
            with slot.lock:
                already_published = (
                    getattr(self, '_last_panel_signatures', {}).get(panel)
                    == signature
                )
                if (
                    not already_published
                    and slot.scheduled_signature == signature
                ):
                    slot.scheduled_signature = None

    def _record_panel_output(
        self, panel: str, output: CompressedImage, image: np.ndarray, now: float,
    ) -> None:
        """Retain per-view metrics for status when the 2x2 output is off."""
        entries = getattr(self, '_last_panel_outputs', {})
        metric = entries.get(panel)
        if not isinstance(metric, PanelOutput):
            metric = PanelOutput()
            entries[panel] = metric
        if metric.published_at is not None:
            interval = now - metric.published_at
            if interval > 1e-6:
                metric.hz = 1.0 / interval
        metric.source_header = output.header
        metric.published_at = now
        metric.bytes = len(output.data)
        metric.height, metric.width = image.shape[:2]
        self._last_panel_outputs = entries

    def _status_output(self) -> PanelOutput | None:
        """Return the actual encoded output represented by status-v1."""
        legacy_header = getattr(self, '_last_output_source_header', None)
        legacy_at = getattr(self, '_last_output_at', None)
        if legacy_header is not None and legacy_at is not None:
            return PanelOutput(
                source_header=legacy_header,
                published_at=legacy_at,
                hz=float(getattr(self, '_output_hz', 0.0)),
                bytes=int(getattr(self, '_last_output_bytes', 0)),
                width=int(getattr(self, '_last_output_width', 0)),
                height=int(getattr(self, '_last_output_height', 0)),
            )
        entries = getattr(self, '_last_panel_outputs', {})
        # CAM3/CAM4 are the strict status-v1 cameras.  Prefer whichever of
        # their native final panels was encoded most recently.
        candidates = [
            entries.get(camera) for camera in CAMERAS
            if isinstance(entries.get(camera), PanelOutput)
            and entries[camera].source_header is not None
            and entries[camera].published_at is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.published_at or -1.0)

    def _publish_if_current(self) -> None:
        self._drain_latest_bases()
        now = time.monotonic()
        contexts = {camera: self._camera_context(camera, now) for camera in CAMERAS}
        suction_context = self._suction_context(now)
        right_ee_context = self._right_ee_context(now)
        panel_signatures = {
            'cam_3': self._camera_panel_signature('cam_3', contexts['cam_3']),
            'cam_4': self._camera_panel_signature('cam_4', contexts['cam_4']),
            'suction': self._suction_panel_signature(suction_context),
            'right_ee': self._right_ee_panel_signature(right_ee_context),
        }
        anchors = [
            camera for camera in CAMERAS
            if contexts[camera]['base_state'] == 'live'
        ]
        suction_anchor = suction_context['selected']
        right_ee_anchor = right_ee_context['selected']
        clearing = not anchors and suction_anchor is None and right_ee_anchor is None
        # The composite remains backward-compatible.  Its signature includes
        # every visible per-panel state so status labels clear promptly too.
        signature: tuple[Any, ...] = (
            'clear' if clearing else 'live',
        ) + tuple((panel, panel_signatures[panel]) for panel in PANEL_NAMES)
        publish_composite = (
            bool(getattr(self, '_enable_composite_output', True))
            and getattr(self, '_image_publisher', None) is not None
            and signature != self._last_signature
            and (
                not clearing
                or (
                    self._last_output_at is not None
                    and self._last_output_source_header is not None
                )
            )
        )
        panel_publishers = getattr(self, '_panel_image_publishers', {})
        changed_panels = tuple(
            panel for panel in PANEL_NAMES
            if panel in panel_publishers
            and panel_signatures[panel] != self._panel_target_signature(panel)
        )
        publish_panel = bool(changed_panels)
        if not publish_composite and not publish_panel:
            return

        # Per-view mode is the normal operator path.  Render only panels
        # with a new source/state signature; unrelated views must not spend
        # decode/draw/encode time just because another camera advanced.
        panels_to_render = PANEL_NAMES if publish_composite else changed_panels
        native_panels: dict[str, np.ndarray] = {}
        for panel in panels_to_render:
            if panel in CAMERAS:
                native_panels[panel] = self._render_camera_panel(
                    panel, contexts[panel], native=True)
            elif panel == 'suction':
                native_panels[panel] = self._render_suction_panel(
                    now, suction_context, native=True)
            elif panel == 'right_ee':
                native_panels[panel] = self._render_right_ee_panel(
                    right_ee_context, now=now, native=True)

        per_view_native = getattr(self, '_per_view_native_resolution', True)
        panels = {
            panel: self._letterbox(native_panels[panel])
            for panel in panels_to_render
        } if publish_composite or not per_view_native else {}
        per_view_images = native_panels if per_view_native else panels
        for panel in changed_panels:
            source_header = self._panel_source_header(
                panel, contexts, suction_context, right_ee_context)
            if self._schedule_panel_encode(
                panel,
                per_view_images[panel],
                panel_signatures[panel],
                source_header,
            ):
                continue
            output = self._publish_panel_if_changed(
                panel,
                per_view_images[panel],
                panel_signatures[panel],
                source_header,
            )
            if output is not None:
                self._record_panel_output(
                    panel, output, per_view_images[panel], now)

        if not publish_composite:
            return
        if anchors:
            anchor_camera = max(
                anchors,
                key=lambda camera: (
                    self._base[camera].freshness.received_monotonic or -1.0
                ),
            )
            anchor_state: LatestBase | None = self._base[anchor_camera]
        elif suction_anchor is not None:
            anchor_state = suction_anchor
        else:
            # A right-EE-only bring-up must still show a valid 2x2 frame. Its
            # source header is the only honest anchor when CAM3/CAM4/suction
            # are unavailable.
            anchor_state = right_ee_anchor
        top_row = np.hstack((panels['cam_3'], panels['cam_4']))
        bottom_row = np.hstack((panels['suction'], panels['right_ee']))
        image = np.vstack((top_row, bottom_row))
        success, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not success:
            self.get_logger().warning('could not encode final Debug JPEG')
            return
        source_header = (
            self._last_output_source_header
            if clearing
            else getattr(getattr(anchor_state, 'message', None), 'header', None)
        )
        if source_header is None:
            return
        output = CompressedImage()
        # A live final image keeps its anchor source stamp. A clearing frame
        # has no fresh camera source, so it retains the previous output header;
        # assigning "now" would falsely claim a new camera observation.
        output.header = source_header
        output.format = 'jpeg'
        output.data = encoded.tobytes()
        self._image_publisher.publish(output)
        if self._last_output_at is not None:
            interval = now - self._last_output_at
            if interval > 1e-6:
                self._output_hz = 1.0 / interval
        self._last_output_at = now
        self._last_output_source_header = output.header
        self._last_output_bytes = len(output.data)
        self._last_output_height, self._last_output_width = image.shape[:2]
        self._last_signature = signature

    def _render_camera_panel(
        self, camera: str, context: dict[str, Any], *, native: bool = False,
    ) -> np.ndarray:
        base = self._base[camera]
        if context['base_state'] != 'live' or base.image is None:
            label = 'MISSING' if context['base_state'] == 'missing' else 'STALE'
            return self._placeholder_from_source(
                f'{camera.upper()} BASE {label}', base.image, native=native)
        image = base.image.copy()
        draw_tool_roi_overlay(image, self._tool_rois[camera])
        if camera == 'cam_4' and self._show_cam4_hand_roi:
            draw_hand_roi_overlay(image, self._cam4_hand_roi)
        if context['layers']['tool'].drawable:
            self._draw_tool_observations(image, self._layers[camera]['tool'].message)
        if context['layers']['pose'].drawable:
            self._draw_tool_poses(image, self._layers[camera]['pose'].message, self._camera_info[camera])
        gesture_message = (
            self._gesture.message
            if camera == 'cam_4' and context['gesture'].drawable
            else None
        )
        hand_message = (
            self._layers[camera]['hand'].message
            if context['layers']['hand'].drawable
            else None
        )
        facing_message = (
            self._facing.message
            if camera == 'cam_4' and context['facing'].drawable
            else None
        )
        humanoid_pose = (
            self._cam4_palm_pose.message
            if camera == 'cam_4' and context['cam4_palm_pose'].drawable
            else None
        )
        matched_palm = matched_humanoid_palm(hand_message, humanoid_pose)
        joined_facings = joined_facing_by_hand_index(
            hand_message, facing_message)
        if context['layers']['hand'].drawable:
            self._draw_hands(
                image,
                hand_message,
                # An enabled facing layer with no exact join is explicitly
                # UNKNOWN for each visible hand, never an old semantic label.
                facing_by_hand_index=(
                    joined_facings if camera == 'cam_4' and self._enable_facing
                    else None
                ),
                camera_info=self._camera_info[camera],
                selected_palm_hand_index=(
                    matched_palm[0].hand_index if matched_palm is not None else None
                ),
            )
        if context['layers']['blood'].drawable:
            self._draw_blood(image, self._layers[camera]['blood'])
        gesture_box_bottom = None
        if gesture_message is not None or joined_facings:
            gesture_box_bottom = self._draw_gesture_summary(
                image,
                gesture_message,
                facing_rows=list(joined_facings.values()),
            )
        if matched_palm is not None:
            draw_humanoid_palm_hud(
                image,
                matched_palm[1],
                top_y=(gesture_box_bottom + 12 if gesture_box_bottom else 58),
            )
        status_items = [
            f'{name}:{context["layers"][name].state}' for name in LAYER_NAMES
            if context['layers'][name].state != 'disabled'
        ]
        if context['gesture'].state != 'disabled':
            status_items.append(f'gesture:{context["gesture"].state}')
        if context['facing'].state != 'disabled':
            status_items.append(f'facing:{context["facing"].state}')
        if context['cam4_palm_pose'].state != 'disabled':
            status_items.append(
                f'palm_humanoid:{context["cam4_palm_pose"].state}')
        status = ' '.join(status_items)
        status_text = f'{camera.upper()}  {status}'
        cv2.rectangle(image, (0, 0), (image.shape[1], 46), (12, 24, 36), -1)
        draw_outlined_text(image, status_text, (14, 32),
                          0.68, (220, 240, 250), 2)
        if camera == 'cam_4' and self._show_cam4_hand_roi:
            (status_width, _), _ = cv2.getTextSize(
                status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
            draw_hand_roi_header(
                image,
                self._cam4_hand_roi,
                minimum_x=14 + status_width + 24,
            )
        return image if native else self._letterbox(image)

    def _suction_context(self, now: float) -> dict[str, Any]:
        """Select only source-aligned, receiver-fresh suction content."""
        if not getattr(self, '_enable_suction_panel', False):
            return {
                'base_state': 'disabled',
                'overlay_state': 'disabled',
                'overlay_drawable': False,
                'mask_state': 'disabled',
                'mask_drawable': False,
                'selected': None,
            }
        suction_base = getattr(self, '_suction_base', LatestBase())
        suction_overlay = getattr(self, '_suction_overlay', LatestBase())
        suction_mask = getattr(self, '_suction_mask', LatestLayer())
        base_state = freshness_state(
            has_value=(
                suction_base.message is not None
                and suction_base.image is not None
            ),
            age_sec=suction_base.freshness.age(now),
            max_age_sec=self._max_base_age_sec,
        )
        overlay_state = freshness_state(
            has_value=(
                suction_overlay.message is not None
            ),
            age_sec=suction_overlay.freshness.age(now),
            max_age_sec=self._max_layer_age_sec,
        )
        overlay_drawable = bool(
            base_state == 'live'
            and overlay_state == 'live'
            and suction_base.source_stamp_ns is not None
            and suction_overlay.source_stamp_ns is not None
            and abs(
                suction_base.source_stamp_ns
                - suction_overlay.source_stamp_ns
            ) <= self._max_source_delta_ns
        )
        if overlay_state == 'live' and not overlay_drawable:
            overlay_state = 'stale'
        mask_state = freshness_state(
            has_value=(
                suction_mask.message is not None
                and isinstance(suction_mask.payload, np.ndarray)
            ),
            age_sec=suction_mask.freshness.age(now),
            max_age_sec=self._max_layer_age_sec,
        )
        mask_drawable = layer_is_drawable(
            base_stamp_ns=suction_base.source_stamp_ns,
            layer_stamp_ns=suction_mask.source_stamp_ns,
            base_state=base_state,
            layer_state=mask_state,
            max_source_delta_ns=self._max_source_delta_ns,
        )
        if mask_state == 'live' and not mask_drawable:
            mask_state = 'stale'
        # Keep the third panel live at the camera cadence.  A complete Blood
        # overlay JPEG contains an older RGB raster and would otherwise freeze
        # the panel to inference cadence; only the source-stamped mask is
        # composited over the current raw suction frame.
        selected = suction_base if base_state == 'live' else None
        return {
            'base_state': base_state,
            'overlay_state': overlay_state,
            'overlay_drawable': overlay_drawable,
            'mask_state': mask_state,
            'mask_drawable': mask_drawable,
            'selected': selected,
        }

    def _render_suction_panel(
        self, now: float, context: dict[str, Any] | None = None,
        *, native: bool = False,
    ) -> np.ndarray:
        """Show live Blood overlay, with current RGB as a visual fallback."""
        if not getattr(self, '_enable_suction_panel', False):
            return self._placeholder_panel('SUCTION PANEL DISABLED')
        context = self._suction_context(now) if context is None else context
        selected = context['selected']
        if selected is None or selected.image is None:
            label = (
                'MISSING' if context['base_state'] == 'missing' else 'STALE'
            )
            return self._placeholder_from_source(
                f'SUCTION BASE {label}',
                getattr(getattr(self, '_suction_base', None), 'image', None),
                native=native,
            )
        image = selected.image.copy()
        if context['mask_drawable']:
            self._draw_blood(image, self._suction_mask, label='BLEEDING')
        blood_label = (
            'LIVE' if context['mask_drawable']
            else str(context['mask_state']).upper()
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 46), (12, 24, 36), -1)
        draw_outlined_text(
            image,
            f'SUCTION  BLEEDING:{blood_label}',
            (14, 32),
            0.68,
            (100, 245, 150)
            if context['mask_drawable'] else (65, 180, 255),
            2,
        )
        return image if native else self._letterbox(image)

    def _right_ee_context(self, now: float) -> dict[str, Any]:
        """Return raw EE RGB plus source-aligned skeleton and palm evidence."""
        if not getattr(self, '_enable_right_ee_panel', False):
            return {
                'base_state': 'disabled',
                'hand_state': 'disabled',
                'hand_drawable': False,
                'gesture_state': 'disabled',
                'gesture_drawable': False,
                'selected': None,
            }
        base = getattr(self, '_right_ee_base', LatestBase())
        hand = getattr(self, '_right_ee_hand', LatestLayer())
        gesture = getattr(self, '_right_ee_gesture', LatestLayer())
        hand_max_age_sec = getattr(
            self, '_max_right_ee_hand_age_sec', self._max_gesture_age_sec)
        hand_max_source_delta_ns = getattr(
            self, '_max_right_ee_hand_source_delta_ns',
            self._max_gesture_source_delta_ns,
        )
        base_state = freshness_state(
            has_value=base.message is not None and base.image is not None,
            age_sec=base.freshness.age(now),
            max_age_sec=self._max_base_age_sec,
        )
        hand_state = freshness_state(
            has_value=hand.message is not None,
            age_sec=hand.freshness.age(now),
            max_age_sec=hand_max_age_sec,
            disabled=not getattr(self, '_enable_right_ee_hand', False),
        )
        hand_drawable = layer_is_drawable(
            base_stamp_ns=base.source_stamp_ns,
            layer_stamp_ns=hand.source_stamp_ns,
            base_state=base_state,
            layer_state=hand_state,
            max_source_delta_ns=hand_max_source_delta_ns,
        )
        if hand_state == 'live' and not hand_drawable:
            signature = (
                base.source_stamp_ns,
                hand.source_stamp_ns,
                base_state,
            )
            if hand.last_drop_signature != signature:
                hand.dropped += 1
                hand.last_drop_signature = signature
            hand_state = 'stale'
        gesture_state = freshness_state(
            has_value=gesture.message is not None,
            age_sec=gesture.freshness.age(now),
            max_age_sec=self._max_gesture_age_sec,
        )
        gesture_drawable = layer_is_drawable(
            base_stamp_ns=base.source_stamp_ns,
            layer_stamp_ns=gesture.source_stamp_ns,
            base_state=base_state,
            layer_state=gesture_state,
            max_source_delta_ns=self._max_gesture_source_delta_ns,
        )
        if gesture_state == 'live' and not gesture_drawable:
            gesture_state = 'stale'
        return {
            'base_state': base_state,
            'hand_state': hand_state,
            'hand_drawable': hand_drawable,
            'gesture_state': gesture_state,
            'gesture_drawable': gesture_drawable,
            'selected': base if base_state == 'live' else None,
        }

    def _render_right_ee_panel(
        self, context: dict[str, Any], *, now: float | None = None,
        native: bool = False,
    ) -> np.ndarray:
        """Panel 4: current EE RGB, aligned skeleton, and palm evidence."""
        if not getattr(self, '_enable_right_ee_panel', False):
            return self._placeholder_panel('RIGHT EE PANEL DISABLED')
        selected = context['selected']
        if selected is None or selected.image is None:
            label = 'MISSING' if context['base_state'] == 'missing' else 'STALE'
            return self._placeholder_from_source(
                f'RIGHT EE BASE {label}',
                getattr(getattr(self, '_right_ee_base', None), 'image', None),
                native=native,
            )
        image = selected.image.copy()
        right_ee_hand = getattr(self, '_right_ee_hand', LatestLayer())
        right_ee_gesture = getattr(self, '_right_ee_gesture', LatestLayer())
        hand_message = (
            right_ee_hand.message if context.get('hand_drawable', False) else None
        )
        gesture_message = (
            right_ee_gesture.message if context['gesture_drawable'] else None
        )
        if hand_message is not None:
            self._draw_hands(image, hand_message)
        palm_label, palm_color = self._draw_right_ee_palm_state(
            image,
            gesture_message,
            hand_message=hand_message,
            source_stamp_ns=(
                right_ee_gesture.source_stamp_ns
                if (
                    gesture_message is not None
                    and hand_message is not None
                    and right_ee_gesture.source_stamp_ns is not None
                    and right_ee_gesture.source_stamp_ns
                    == right_ee_hand.source_stamp_ns
                )
                else None
            ),
            now=time.monotonic() if now is None else now,
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 46), (12, 24, 36), -1)
        draw_outlined_text(
            image,
            f'RIGHT EE  MEDIAPIPE PALM:{palm_label} '
            f'HAND:{str(context.get("hand_state", "disabled")).upper()}',
            (14, 32), 0.58, palm_color, 2,
        )
        return image if native else self._letterbox(image)

    def _right_ee_palm_filter_state(self) -> RightEePalmDisplayFilter:
        """Lazily allocate UI-only state for direct renderer unit tests."""
        filter_ = getattr(self, '_right_ee_palm_filter', None)
        if not isinstance(filter_, RightEePalmDisplayFilter):
            filter_ = RightEePalmDisplayFilter(
                hold_sec=float(getattr(self, '_right_ee_palm_hold_sec', 0.25)),
            )
            self._right_ee_palm_filter = filter_
        return filter_

    def _draw_right_ee_palm_state(
        self,
        image: np.ndarray,
        message: HandGestureArray | None,
        *,
        hand_message: HandKeypoints | None,
        source_stamp_ns: int | None,
        now: float,
    ) -> tuple[str, tuple[int, int, int]]:
        """Draw source-aligned palm evidence with a display-only filter.

        The gesture source stamp is passed only when it exactly matches the
        displayed keypoint source stamp.  A current zero-hand frame clears the
        HUD immediately.  This leaves the typed gesture stream untouched and
        never promotes stale or absent evidence into a robot-control signal.
        """
        rows = list(getattr(message, 'hands', [])) if message is not None else []
        labels: list[tuple[str, float]] = []
        for hand in rows:
            if not bool(getattr(hand, 'has_classification', False)):
                continue
            category = str(getattr(hand, 'category_name', '')).strip()
            if category not in {'Open_Palm', 'Closed_Fist'}:
                continue
            try:
                score = float(getattr(hand, 'score', float('nan')))
            except (TypeError, ValueError):
                score = float('nan')
            labels.append((category, score))
        hands = list(getattr(hand_message, 'hands', [])) if hand_message is not None else []
        category, score = (
            max(labels, key=lambda item: item[1] if math.isfinite(item[1]) else -1.0)
            if labels else ('', math.nan)
        )
        accepted, accepted_score, held = self._right_ee_palm_filter_state().update(
            category=category,
            score=score,
            hand_present=bool(hands),
            source_stamp_ns=source_stamp_ns,
            now=now,
        )
        if accepted == 'Open_Palm':
            label, color = 'OPEN', (70, 235, 90)
        elif accepted == 'Closed_Fist':
            label, color = 'CLOSED', (80, 170, 255)
        else:
            label, color = 'UNKNOWN', (65, 180, 255)
        if accepted and accepted_score is not None and math.isfinite(accepted_score):
            label += f' {accepted_score:.2f}'
        if held:
            label += ' / HOLD'
        overlay = image.copy()
        x0, y0 = 18, 62
        x1, y1 = min(image.shape[1] - 18, 520), min(image.shape[0] - 18, 130)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 24, 36), -1)
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        draw_outlined_text(image, f'PALM {label}', (x0 + 14, y0 + 45), 0.92, color, 2)
        return label, color

    def _placeholder_panel(self, label: str) -> np.ndarray:
        panel = np.zeros(
            (self._panel_height, self._panel_width, 3), dtype=np.uint8)
        draw_outlined_text(
            panel, label, (28, 58), 0.95, (135, 155, 175), 2)
        return panel

    def _placeholder_from_source(
        self, label: str, source_image: np.ndarray | None, *, native: bool,
    ) -> np.ndarray:
        """Keep a stale native output at its last truthful source resolution."""
        if native and isinstance(source_image, np.ndarray) and source_image.ndim == 3:
            panel = np.zeros_like(source_image)
            draw_outlined_text(
                panel, label, (28, 58), 0.95, (135, 155, 175), 2)
            return panel
        return self._placeholder_panel(label)

    def _render_reserved_panel(self) -> np.ndarray:
        return self._placeholder_panel('PANEL 4  RESERVED')

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(self._panel_width / width, self._panel_height / height)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        panel = np.zeros((self._panel_height, self._panel_width, 3), dtype=np.uint8)
        y = (self._panel_height - resized.shape[0]) // 2
        x = (self._panel_width - resized.shape[1]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return panel

    def _draw_tool_observations(self, image: np.ndarray, message: ToolObservation2DArray) -> None:
        for item in message.instances:
            color = TOOL_COLORS_BGR.get(str(item.class_name), (235, 235, 235))
            x0, y0, x1, y1 = (int(round(value)) for value in item.bbox_xyxy_px)
            draw_outlined_rectangle(image, (x0, y0), (x1, y1), color)
            label = f'{item.class_name} {float(item.class_confidence):.2f}'
            draw_outlined_text(image, label, (x0, max(26, y0 - 8)), 0.78, color, 2)
            if bool(item.observation_point_valid):
                u, v = (int(round(value)) for value in item.observation_point_uv_px)
                cv2.circle(image, (u, v), 5, color, -1, cv2.LINE_AA)
                cv2.circle(image, (u, v), 9, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_tool_poses(
        self, image: np.ndarray, message: ToolPoseArray, camera_info: CameraInfo | None
    ) -> None:
        if camera_info is None or len(camera_info.k) != 9:
            return
        K = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        D = np.asarray(camera_info.d, dtype=np.float64).reshape(-1, 1)
        for item in message.tools:
            if not bool(item.position_valid):
                continue
            position = np.asarray([
                item.pose.position.x, item.pose.position.y, item.pose.position.z
            ], dtype=np.float64)
            rotation = quaternion_matrix_xyzw(item.pose.orientation)
            if not np.all(np.isfinite(position)) or rotation is None:
                continue
            points = np.vstack((
                position,
                position + rotation[:, 0] * self._pose_axis_length_m,
                position + rotation[:, 1] * self._pose_axis_length_m,
                position + rotation[:, 2] * self._pose_axis_length_m,
            ))
            if np.any(points[:, 2] <= 0.0):
                continue
            try:
                projected, _ = cv2.projectPoints(
                    points, np.zeros(3), np.zeros(3), K, D)
            except cv2.error:
                continue
            uv = projected.reshape(-1, 2)
            if not np.all(np.isfinite(uv)):
                continue
            origin = tuple(np.rint(uv[0]).astype(int))
            for endpoint, axis_color in zip(uv[1:], ((40, 40, 255), (40, 230, 40), (255, 120, 40))):
                cv2.line(image, origin, tuple(np.rint(endpoint).astype(int)), axis_color, 3, cv2.LINE_AA)

    def _draw_camera_palm_axes(
        self, image: np.ndarray, hand: Any, camera_info: CameraInfo | None,
    ) -> None:
        """Project the selected metric ``T_cam4_palm`` onto the CAM4 image."""
        if camera_info is None or len(camera_info.k) != 9:
            return
        points = camera_palm_axis_points(
            getattr(hand, 'palm_6d', None),
            getattr(self, '_palm_axis_length_m', 0.08),
        )
        if points is None or np.any(points[:, 2] <= 0.0):
            return
        K = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        D = np.asarray(camera_info.d, dtype=np.float64).reshape(-1, 1)
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
            return
        try:
            projected, _ = cv2.projectPoints(
                points, np.zeros(3), np.zeros(3), K, D)
        except cv2.error:
            return
        uv = projected.reshape(-1, 2)
        if not np.all(np.isfinite(uv)):
            return
        origin = tuple(np.rint(uv[0]).astype(int))
        cv2.circle(image, origin, 7, (250, 250, 250), -1, cv2.LINE_AA)
        for label, endpoint, axis_color in zip(
            ('X', 'Y', 'Z'), uv[1:], PALM_AXIS_COLORS_BGR,
        ):
            endpoint_px = tuple(np.rint(endpoint).astype(int))
            cv2.line(image, origin, endpoint_px, axis_color, 4, cv2.LINE_AA)
            draw_outlined_text(
                image, label, (endpoint_px[0] + 4, endpoint_px[1] - 4),
                0.54, axis_color, 2,
            )
        draw_outlined_text(
            image, 'T_CAM4_PALM', (origin[0] + 9, origin[1] + 25),
            0.56, (235, 245, 250), 2,
        )

    def _draw_hands(
        self,
        image: np.ndarray,
        message: HandKeypoints,
        *,
        facing_by_hand_index: dict[int, Any] | None = None,
        camera_info: CameraInfo | None = None,
        selected_palm_hand_index: int | None = None,
    ) -> None:
        show_facing = facing_by_hand_index is not None
        for hand in message.hands:
            color = (50, 220, 50) if str(hand.handedness_label).lower() == 'right' else (255, 180, 35)
            points = [
                (int(round(joint.u)), int(round(joint.v)))
                for joint in hand.joints_2d
            ]
            # A malformed partial message is not valid skeleton evidence.  Do
            # not let it terminate the compositor timer or leave a fake hand
            # on a newer EE camera frame.
            if len(points) != 21:
                continue
            for start, end in HAND_EDGES:
                cv2.line(image, points[start], points[end], color, 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, point, 4, (250, 250, 250), -1, cv2.LINE_AA)
            if points:
                label = str(hand.handedness_label) if bool(hand.has_handedness) else 'Hand'
                facing = None
                if show_facing:
                    try:
                        facing = facing_by_hand_index.get(int(hand.hand_index))
                    except (AttributeError, TypeError, ValueError):
                        pass
                if show_facing:
                    if facing is None:
                        label += ' | UNKNOWN'
                    else:
                        facing_label = str(
                            getattr(facing, 'facing_label', '')).strip()
                        if (
                            not bool(getattr(facing, 'has_facing', False))
                            or facing_label not in {'PALM_UP', 'PALM_DOWN', 'EDGE'}
                        ):
                            label += ' | UNKNOWN'
                        else:
                            try:
                                score = float(facing.palm_up_score)
                            except (AttributeError, TypeError, ValueError):
                                score = math.nan
                            label += f' | {facing_label}'
                            if math.isfinite(score):
                                label += f' {score:+.2f}'
                draw_outlined_text(image, label, (points[0][0] + 8, points[0][1] - 8),
                                  0.68, color, 2)
            try:
                hand_index = int(hand.hand_index)
            except (AttributeError, TypeError, ValueError):
                hand_index = -1
            if hand_index == selected_palm_hand_index:
                self._draw_camera_palm_axes(image, hand, camera_info)

    def _draw_gesture_summary(
        self,
        image: np.ndarray,
        message: HandGestureArray | None,
        *,
        facing_rows: list[Any] | tuple[Any, ...] = (),
    ) -> int | None:
        gestures = list(message.hands)[:4] if message is not None else []
        facings = list(facing_rows)[:4]
        if not gestures and not facings:
            return None
        rows = [
            (gesture_display_text(gesture), gesture_color(gesture))
            for gesture in gestures
        ]
        rows.extend(
            (palm_facing_display_text(hand), palm_facing_color(hand))
            for hand in facings
        )
        box_width = min(680, max(400, image.shape[1] - 20))
        row_height = 31
        box_height = 40 + row_height * len(rows)
        # Keep the hand HUD on the CAM4 upper-left as requested.  The palm
        # coordinate HUD, if source-matched, is stacked underneath it.
        x0 = 18
        y0 = 56
        x1 = min(image.shape[1] - 1, x0 + box_width)
        y1 = min(image.shape[0] - 1, y0 + box_height)
        if y1 <= y0:
            return None
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 24, 36), -1)
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
        cv2.rectangle(image, (x0, y0), (x1, y1), (180, 205, 220), 2, cv2.LINE_AA)
        draw_outlined_text(image, 'TOP-VIEW HAND STATE', (x0 + 12, y0 + 27),
                          0.62, (225, 240, 250), 2)
        for row_index, (row, color) in enumerate(rows):
            baseline = y0 + 58 + row_index * row_height
            if baseline >= y1:
                break
            draw_outlined_text(image, row, (x0 + 12, baseline),
                              0.64, color, 2)
        return y1

    def _draw_blood(
        self,
        image: np.ndarray,
        state: LatestLayer,
        *,
        label: str | None = None,
    ) -> None:
        mask = state.payload
        if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
            signature = (state.source_stamp_ns, image.shape[:2], getattr(mask, 'shape', None))
            if state.last_drop_signature != signature:
                state.dropped += 1
                state.last_drop_signature = signature
            return
        tinted = image.copy()
        tinted[mask] = (40, 40, 235)
        blended = cv2.addWeighted(image, 0.65, tinted, 0.35, 0.0)
        image[mask] = blended[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (35, 35, 255), 3, cv2.LINE_AA)
        if label and bool(np.any(mask)):
            ys, xs = np.nonzero(mask)
            draw_outlined_text(
                image,
                str(label),
                (max(8, int(xs.min()) + 6), max(28, int(ys.min()) - 8)),
                0.78,
                (35, 35, 255),
                2,
            )

    def _publish_status(self) -> None:
        now = time.monotonic()
        # The Debug UI's status transport deliberately has a strict base/output
        # schema.  Until each camera has contributed one decoded base and one
        # actual encoded final view (legacy composite or native per-view), its source
        # stamp or receiver age would be unknowable.  Suppress the document rather than forge a
        # zero timestamp or make the strict consumer parse a partial object.
        output = self._status_output()
        if output is None or any(
            self._base[camera].message is None or self._base[camera].image is None
            for camera in CAMERAS
        ):
            return
        contexts = {camera: self._camera_context(camera, now) for camera in CAMERAS}
        wall_ns = time.time_ns()
        payload: dict[str, Any] = {
            'schema': STATUS_SCHEMA,
            'published_at': {'sec': wall_ns // 1_000_000_000, 'nanosec': wall_ns % 1_000_000_000},
            'output': {
                # Do not recalculate this from a newer callback.  A base can
                # arrive between image timer and status timer; the status must
                # identify the already published JPEG, not that later frame.
                'source_stamp': stamp_dict(output.source_header),
                'hz': round(float(output.hz), 3),
                'bytes': int(output.bytes),
                'width': int(output.width),
                'height': int(output.height),
            },
            'cameras': {},
        }
        for camera in CAMERAS:
            base = self._base[camera]
            base_state = contexts[camera]['base_state']
            layers: dict[str, Any] = {}
            for layer in LAYER_NAMES:
                current = self._layers[camera][layer]
                decision = contexts[camera]['layers'][layer]
                layers[layer] = {
                    'state': decision.state,
                    'source_stamp': stamp_dict(current.message),
                    'age_sec': None if decision.age_sec is None else round(decision.age_sec, 3),
                    'count': int(current.count),
                    'dropped': int(current.dropped),
                }
            payload['cameras'][CAMERA_STATUS_KEYS[camera]] = {
                'state': base_state,
                'base': {
                    'source_stamp': stamp_dict(base.message),
                    'age_sec': None if contexts[camera]['base_age'] is None else round(contexts[camera]['base_age'], 3),
                    'received': int(base.received),
                    'dropped': int(base.dropped),
                },
                'layers': layers,
            }
        self._status_publisher.publish(String(data=json.dumps(payload, separators=(',', ':'))))

    def destroy_node(self) -> bool:
        executor = getattr(self, '_panel_encode_executor', None)
        self._panel_encode_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FinalOverlayCompositor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
