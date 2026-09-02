"""Estimate constrained tool poses from ROS RGB, sampling depth when present."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import cv2

import numpy as np

from pnu_surgical_perception.depth_to_color_extrinsics import (
    DepthToColorExtrinsics,
    validate_depth_to_color_extrinsics,
)
from pnu_surgical_perception.native_depth_sync import (
    ApproximateRgbDepthPairer,
    RgbDepthPair,
)
from pnu_surgical_perception.pose_message_mapping import (
    to_observation_array_from_detections,
    to_pose_array_from_result,
)
from pnu_surgical_perception.tool_pose_tf import (
    CONSTRAINED_SE3_PROVENANCE,
    selector_horizontal_u_px,
    source_age_seconds,
    source_stamp_nanoseconds,
    spatial_tool_child_frames,
    ToolSpatialTfSelector,
)

import rclpy
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)

from realsense2_camera_msgs.msg import Extrinsics
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, String
from surgical_perception_msgs.msg import (
    ToolObservation2DArray,
    ToolPose,
    ToolPoseArray,
)
from tf2_ros import TransformBroadcaster


DEFAULT_ALGORITHM_PATH = os.environ.get(
    'PNU_SURGICAL_TOOL_ALGORITHM_PATH', ''
)
DEFAULT_CHECKPOINT = os.environ.get('PNU_RFDETR_CHECKPOINT', '')
DEFAULT_ONTOLOGY = os.environ.get('PNU_SURGICAL_TOOL_ONTOLOGY', '')
DEFAULT_DEPTH_REGISTRATION_CUDA_LIBRARY = os.environ.get(
    'PNU_DEPTH_REGISTRATION_CUDA_LIBRARY', ''
)


# Fixed canonical colors keep the same instrument recognizable across CAM3,
# CAM4, and process restarts. BGR is required by OpenCV drawing functions.
TOOL_OVERLAY_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    'Scalpel': (0, 145, 255),
    'Allis Forceps': (255, 130, 30),
    'Mosquito': (70, 190, 80),
    'Adson Forceps': (205, 70, 195),
    'Bipolar Forceps': (0, 230, 230),
    'Bovie': (100, 60, 255),
    'Army-Navy Retractor': (185, 85, 155),
    'Thyroid Retractor': (210, 180, 30),
}

DEFAULT_CLASS_MASK_NAMES = tuple(TOOL_OVERLAY_COLORS_BGR)


def tool_overlay_color_bgr(class_name: str) -> tuple[int, int, int]:
    """Return the canonical visual color for a recognized surgical tool."""
    return TOOL_OVERLAY_COLORS_BGR.get(class_name, (235, 235, 235))


def class_mask_slug(class_name: str) -> str:
    """Return a deterministic ROS-topic suffix for one ontology class."""
    slug = re.sub(r'[^a-z0-9]+', '_', str(class_name).strip().lower()).strip('_')
    if not slug:
        raise ValueError('class mask name must contain an ASCII letter or digit')
    return slug


def union_class_masks(
    detections: Any,
    class_names: tuple[str, ...] | list[str],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Build one uint8 mono mask per class, including empty classes."""
    height, width = (int(value) for value in image_shape)
    if height <= 0 or width <= 0:
        raise ValueError('image_shape must contain positive height and width')
    masks = {
        str(name): np.zeros((height, width), dtype=np.uint8)
        for name in class_names
    }
    for instance in detections.instances:
        class_name = str(instance.class_name)
        if class_name not in masks:
            continue
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f'{class_name} mask shape {mask.shape} != RGB shape '
                f'{(height, width)}'
            )
        masks[class_name][mask] = 255
    return masks


def class_mask_messages(
    detections: Any,
    class_names: tuple[str, ...] | list[str],
    image_shape: tuple[int, int],
    header: Any,
) -> tuple[tuple[str, Image], ...]:
    """Build the complete source-stamped mask set for one output bundle."""
    messages = []
    for class_name, mask in union_class_masks(
        detections, class_names, image_shape
    ).items():
        message = Image()
        message.header = header
        message.height = int(mask.shape[0])
        message.width = int(mask.shape[1])
        message.encoding = 'mono8'
        message.is_bigendian = False
        message.step = int(mask.shape[1])
        message.data = mask.tobytes(order='C')
        messages.append((class_name, message))
    return tuple(messages)


def aligned_depth_to_meters(
    native_depth: np.ndarray,
    image_shape: tuple[int, int],
    depth_scale_m_per_unit: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> np.ndarray:
    """Validate color-aligned uint16 depth and convert it to metres."""
    source = np.asarray(native_depth)
    expected = tuple(int(value) for value in image_shape)
    if source.ndim != 2 or source.shape != expected:
        raise ValueError(
            f'aligned depth shape {source.shape} != RGB shape {expected}'
        )
    if source.dtype != np.uint16:
        raise ValueError(f'aligned depth must be uint16, got {source.dtype}')
    scale = float(depth_scale_m_per_unit)
    minimum = float(minimum_depth_m)
    maximum = float(maximum_depth_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('depth scale must be finite and positive')
    if not 0.0 <= minimum < maximum or not np.isfinite(maximum):
        raise ValueError('depth limits must satisfy 0 <= minimum < maximum')
    depth_m = source.astype(np.float32) * scale
    valid = (
        (source != 0)
        & np.isfinite(depth_m)
        & (depth_m >= minimum)
        & (depth_m <= maximum)
    )
    depth_m[~valid] = np.nan
    return depth_m


def reliable_qos(depth: int = 5) -> QoSProfile:
    """Return reliable volatile QoS for compact pose results."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def image_reader_qos() -> QoSProfile:
    """Latest-frame image reader, compatible with ingress BEST_EFFORT output."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def camera_info_qos() -> QoSProfile:
    """Reliable CameraInfo calibration reader kept separate from image QoS."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def depth_to_color_extrinsics_qos() -> QoSProfile:
    """Match the latched VIPLab RealSense depth-to-color calibration QoS."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def decode_rgb(message: CompressedImage) -> np.ndarray:
    """Decode a compressed RGB message to an OpenCV BGR image."""
    frame = cv2.imdecode(
        np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise ValueError('OpenCV could not decode the RGB image')
    return frame


def finite_vector(name: str, values: Any, length: int) -> np.ndarray:
    """Validate and return a finite fixed-length floating-point vector."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain {length} finite values')
    return vector


def camera_info_signature(message: CameraInfo) -> tuple[Any, ...]:
    """Return the fields that require rebuilding the registration cache."""
    return (
        int(message.width),
        int(message.height),
        str(message.header.frame_id),
        *tuple(float(value) for value in message.k),
        *tuple(float(value) for value in message.d),
    )


@dataclass(frozen=True)
class PendingPoseFrame:
    """One RGB frame, with optional paired depth for pose and UV sampling."""

    rgb: CompressedImage
    depth: CompressedImage | None
    color_info: CameraInfo | None
    depth_info: CameraInfo | None
    depth_to_color_extrinsics: DepthToColorExtrinsics | None
    extrinsics_revision: int
    rgb_depth_delta_ns: int | None
    received_monotonic: float


@dataclass(frozen=True)
class PendingOverlayFrame:
    """Latest-only visualization job, kept off the pose/mask critical path."""

    rgb: np.ndarray
    detections: Any
    observation_array: ToolObservation2DArray
    selector_labels: tuple[str, ...]
    header: Any
    pose_result: Any | None
    pose_camera: Any | None


@dataclass(frozen=True)
class PendingOutputBundle:
    """One exact-stamp pose/observation/mask result published atomically."""

    pose_array: ToolPoseArray | None
    observation_array: ToolObservation2DArray
    class_mask_messages: tuple[tuple[str, Image], ...]
    selector_u_by_instance_id: dict[int, float]
    diagnostics: dict[str, Any]
    process_started: float
    queued_monotonic: float
    source_stamp_ns: int
    instance_count: int
    valid_pose_count: int


class LatestOnlyOutputSlot:
    """Thread-safe singleton that overwrites stale, not-yet-published output."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: PendingOutputBundle | None = None
        self._stopping = False
        self.overwritten_total = 0

    def put(self, bundle: PendingOutputBundle) -> None:
        with self._condition:
            if self._stopping:
                return
            if self._pending is not None:
                self.overwritten_total += 1
            self._pending = bundle
            self._condition.notify()

    def take(self) -> PendingOutputBundle | None:
        with self._condition:
            while self._pending is None and not self._stopping:
                self._condition.wait()
            if self._stopping:
                return None
            pending = self._pending
            self._pending = None
            return pending

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

    @property
    def has_pending(self) -> bool:
        with self._condition:
            return self._pending is not None


class NativeDepthPoseNode(Node):
    """Latest-frame ROS adapter for native depth registration and tool pose."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Initialize configured subscriptions, publishers, and worker."""
        super().__init__('native_depth_tool_pose', context=context)
        self._declare_parameters()
        self._read_parameters()

        self._pose_publisher = self.create_publisher(
            ToolPoseArray, self._pose_topic, reliable_qos()
        )
        self._observation_publisher = self.create_publisher(
            ToolObservation2DArray, self._observation_topic, reliable_qos()
        )
        self._class_mask_publishers = {
            class_name: self.create_publisher(
                Image,
                self._class_mask_topics[class_name],
                qos_profile_sensor_data,
            )
            for class_name in self._class_mask_names
            if self._publish_class_masks
        }
        self._overlay_publisher = self.create_publisher(
            CompressedImage, self._overlay_topic, qos_profile_sensor_data
        )
        self._pose_overlay_publisher = self.create_publisher(
            CompressedImage, self._pose_overlay_topic, qos_profile_sensor_data
        )
        self._diagnostics_publisher = self.create_publisher(
            String, self._diagnostics_topic, reliable_qos(10)
        )
        self._health_publisher = self.create_publisher(
            String, self._health_topic, reliable_qos(1)
        )
        self._tf_broadcaster = (
            TransformBroadcaster(self) if self._publish_tool_tf else None
        )

        self._condition = threading.Condition()
        self._pairer = ApproximateRgbDepthPairer(
            self._maximum_stamp_delta_ns, self._sync_queue_size
        )
        self._color_info: CameraInfo | None = None
        self._depth_info: CameraInfo | None = None
        self._depth_to_color_extrinsics: DepthToColorExtrinsics | None = None
        self._extrinsics_revision = 0
        self._received_extrinsics = 0
        self._rejected_extrinsics = 0
        self._last_extrinsics_error = ''
        self._pending: PendingPoseFrame | None = None
        self._overlay_pending: PendingOverlayFrame | None = None
        self._output_slot = LatestOnlyOutputSlot()
        self._stopping = False
        self._model_ready = False
        self._registrar_ready = False
        self._aligned_depth_ready = False
        self._last_success_monotonic: float | None = None
        self._last_pair_monotonic: float | None = None
        self._last_error_code = ''
        self._last_error_message = ''
        self._received_rgb = 0
        self._received_depth = 0
        self._paired_frames = 0
        self._processed_frames = 0
        self._class_mask_frames_published = 0
        self._dropped_pending_frames = 0
        self._overlay_processed_frames = 0
        self._overlay_dropped_pending_frames = 0
        self._last_overlay_latency_ms = 0.0
        self._last_overlay_error = ''
        self._output_published_bundles = 0
        self._last_output_bundle_queue_wait_ms = 0.0
        self._last_output_bundle_publish_ms = 0.0
        self._last_output_bundle_latency_ms = 0.0
        self._last_output_bundle_source_age_ms = 0.0
        self._last_output_error = ''
        self._last_publish_log_monotonic = 0.0
        self._sequence = 0
        self._registrar = None
        self._registrar_key: tuple[Any, ...] | None = None
        self._algorithm = None
        self._detector = None
        self._detection_postprocessor = None
        self._support_plane = None
        self._algorithm_symbols: dict[str, Any] = {}
        self._tf_broadcast_total = 0
        self._tf_skipped_total = 0
        self._tf_last_input_source_stamp_ns: int | None = None
        self._tf_last_input_source_age_sec: float | None = None
        self._tf_last_output_source_stamp_ns: int | None = None
        self._tf_last_parent_frame = ''
        self._tf_last_child_frames: tuple[str, ...] = ()
        self._tf_last_skip_reason = ''
        self._tf_last_error = ''
        self._last_tf_log_monotonic = 0.0

        image_qos = image_reader_qos()
        info_qos = camera_info_qos()
        # ``Node`` owns ``_subscriptions`` for lifecycle bookkeeping.  Keep
        # references that this adapter owns separately so ``destroy_node`` can
        # remove each ROS subscription exactly once.
        self._perception_subscriptions = [
            self.create_subscription(
                CameraInfo,
                self._color_info_topic,
                self._receive_color_info,
                info_qos,
            ),
            self.create_subscription(
                CameraInfo,
                self._depth_info_topic,
                self._receive_depth_info,
                info_qos,
            ),
            self.create_subscription(
                CompressedImage,
                self._rgb_topic,
                self._receive_rgb,
                image_qos,
            ),
            self.create_subscription(
                CompressedImage,
                self._depth_topic,
                self._receive_depth,
                image_qos,
            ),
        ]
        if self._extrinsics_topic:
            self._perception_subscriptions.append(
                self.create_subscription(
                    Extrinsics,
                    self._extrinsics_topic,
                    self._receive_depth_to_color_extrinsics,
                    depth_to_color_extrinsics_qos(),
                )
            )
        if self._processing_gate_topic:
            self._perception_subscriptions.append(
                self.create_subscription(
                    Bool,
                    self._processing_gate_topic,
                    self._receive_processing_gate,
                    reliable_qos(1),
                )
            )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name='native-depth-tool-pose',
            daemon=True,
        )
        self._overlay_worker = threading.Thread(
            target=self._overlay_worker_loop,
            name='native-depth-tool-overlay',
            daemon=True,
        )
        self._output_worker = threading.Thread(
            target=self._output_worker_loop,
            name='native-depth-tool-output',
            daemon=True,
        )
        self._worker.start()
        self._overlay_worker.start()
        self._output_worker.start()
        self._health_timer = self.create_timer(1.0, self._publish_health)
        self.get_logger().info(
            f'native-depth pose node started: rgb={self._rgb_topic}, '
            f'depth={self._depth_topic}, require_depth={self._require_depth}, '
            f'depth_alignment={self._depth_alignment_mode}, '
            f'extrinsics={self._extrinsics_topic}, '
            f'workspace_zone={self._workspace_zone}, '
            f'workspace_roi_profile={self._workspace_roi_profile}, '
            f'max_delta_ns={self._maximum_stamp_delta_ns}, '
            f'publish_tool_tf={self._publish_tool_tf}, '
            f'class_mask_topics={len(self._class_mask_topics)}'
        )

    def _declare_parameters(self) -> None:
        """Declare transport, model, and geometry parameters."""
        self.declare_parameter('camera', 'cam_4')
        camera = str(self.get_parameter('camera').value).strip() or 'cam_4'
        default_workspace_zone = {
            'cam_3': 'tray',
            'cam_4': 'mayo',
        }.get(camera, camera)
        self.declare_parameter('workspace_zone', default_workspace_zone)
        self.declare_parameter('workspace_roi_profile', 'none')
        prefix = f'/synced/{camera}'
        out = f'/perception/{camera}/tool'
        self.declare_parameter(
            'rgb_topic', f'{prefix}/color/image_raw/compressed'
        )
        self.declare_parameter(
            'color_camera_info_topic', f'{prefix}/color/camera_info'
        )
        self.declare_parameter(
            'depth_topic', f'{prefix}/depth/image_rect_raw/compressedDepth'
        )
        self.declare_parameter(
            'depth_camera_info_topic', f'{prefix}/depth/camera_info'
        )
        self.declare_parameter(
            'extrinsics_topic', f'{prefix}/extrinsics/depth_to_color'
        )
        self.declare_parameter('require_extrinsics_topic', True)
        self.declare_parameter('depth_aligned_to_color', False)
        self.declare_parameter('pose_topic', f'{out}/poses')
        self.declare_parameter('observation_topic', f'{out}/observations')
        self.declare_parameter('publish_class_masks', True)
        self.declare_parameter('class_mask_topic_prefix', f'{out}/masks')
        self.declare_parameter(
            'class_mask_names', list(DEFAULT_CLASS_MASK_NAMES)
        )
        self.declare_parameter('overlay_topic', f'{out}/overlay/compressed')
        self.declare_parameter(
            'pose_overlay_topic', f'{out}/pose_overlay/compressed'
        )
        self.declare_parameter('diagnostics_topic', f'{out}/diagnostics')
        self.declare_parameter('health_topic', f'{out}/health')
        self.declare_parameter('view', camera)
        self.declare_parameter('maximum_stamp_delta_ns', 1_000_000)
        self.declare_parameter('sync_queue_size', 8)
        self.declare_parameter('latest_frame_only', True)
        self.declare_parameter('opencv_num_threads', 2)
        self.declare_parameter('input_freshness_sec', 2.0)
        self.declare_parameter('processing_enabled', True)
        self.declare_parameter('processing_gate_topic', '')
        self.declare_parameter('require_depth', True)

        self.declare_parameter(
            'algorithm_python_path', DEFAULT_ALGORITHM_PATH
        )
        self.declare_parameter('checkpoint', DEFAULT_CHECKPOINT)
        self.declare_parameter('ontology', DEFAULT_ONTOLOGY)
        self.declare_parameter('model_size', 'xlarge')
        self.declare_parameter('checkpoint_color_order', 'RGB')
        self.declare_parameter('model_version', '')
        self.declare_parameter('trt_server_socket', '')
        self.declare_parameter('trt_request_timeout_sec', 10.0)
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('adson_forceps_confidence_threshold', -1.0)
        self.declare_parameter('bovie_confidence_threshold', -1.0)
        self.declare_parameter('enable_class_agnostic_nms', False)
        self.declare_parameter('class_agnostic_nms_iou', 0.8)
        self.declare_parameter('mask_component_cleanup_enabled', True)
        self.declare_parameter('mask_component_minimum_area_px', 16)
        self.declare_parameter('mask_component_minimum_area_ratio', 0.005)
        self.declare_parameter('workspace_roi_enabled', False)
        self.declare_parameter(
            'workspace_roi_polygon_norm_xy',
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        )
        self.declare_parameter('workspace_roi_minimum_mask_overlap', 0.5)
        self.declare_parameter(
            'workspace_roi_require_mask_centroid_inside', True
        )
        self.declare_parameter('temporal_class_smoothing_enabled', False)
        self.declare_parameter('temporal_class_history_size', 7)
        self.declare_parameter('temporal_class_minimum_switch_frames', 3)
        self.declare_parameter('temporal_class_switch_score_margin', 0.2)
        self.declare_parameter('temporal_association_minimum_mask_iou', 0.10)
        self.declare_parameter('temporal_association_minimum_bbox_iou', 0.20)
        self.declare_parameter(
            'temporal_association_maximum_centroid_distance_norm', 0.06
        )
        self.declare_parameter(
            'temporal_association_maximum_mask_area_ratio', 3.0
        )
        self.declare_parameter('temporal_track_max_missed_frames', 3)
        self.declare_parameter('positive_y_image_direction', 'class_based')
        self.declare_parameter('adson_face_on_width_enabled', False)
        self.declare_parameter('pose_axis_length_m', 0.05)
        self.declare_parameter('optimize_for_inference', True)
        self.declare_parameter('jit_compile', True)
        self.declare_parameter('fp16', True)
        self.declare_parameter('jpeg_quality', 90)

        self.declare_parameter('depth_scale_m_per_unit', 0.0)
        self.declare_parameter('depth_scale_verified', False)
        self.declare_parameter('depth_registration_backend', 'numpy')
        self.declare_parameter(
            'depth_registration_allow_numpy_fallback', False
        )
        self.declare_parameter(
            'depth_registration_cuda_library',
            DEFAULT_DEPTH_REGISTRATION_CUDA_LIBRARY,
        )
        self.declare_parameter('minimum_depth_m', 0.05)
        self.declare_parameter('maximum_depth_m', 10.0)
        self.declare_parameter('depth_to_color_rotation', [float('nan')] * 9)
        self.declare_parameter(
            'depth_to_color_rotation_convention', 'column_major'
        )
        self.declare_parameter(
            'depth_to_color_translation_m', [float('nan')] * 3
        )
        self.declare_parameter('minimum_depth_to_color_baseline_m', 0.02)
        self.declare_parameter('maximum_depth_to_color_baseline_m', 0.12)
        self.declare_parameter(
            'expected_depth_to_color_translation_direction',
            [-1.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'minimum_depth_to_color_direction_cosine', 0.95
        )
        self.declare_parameter(
            'depth_to_color_rotation_orthonormal_tolerance', 1e-4
        )
        self.declare_parameter(
            'depth_to_color_rotation_determinant_tolerance', 1e-4
        )
        self.declare_parameter(
            'expected_color_frame', f'{camera}_color_optical_frame'
        )
        self.declare_parameter(
            'expected_depth_frame', f'{camera}_depth_optical_frame'
        )
        self.declare_parameter('calibration_version', '')

        self.declare_parameter('support_plane_normal', [float('nan')] * 3)
        self.declare_parameter('support_plane_offset_m', float('nan'))
        self.declare_parameter('support_plane_config_version', '')
        self.declare_parameter('support_plane_inlier_ratio', 0.0)
        self.declare_parameter('support_plane_residual_p95_m', 0.0)
        self.declare_parameter('publish_tool_tf', True)
        self.declare_parameter('tf_stale_after_sec', 2.0)
        self.declare_parameter('tf_max_future_sec', 0.5)
        self.declare_parameter('tf_track_max_displacement_m', 0.10)
        self.declare_parameter('tf_track_ttl_sec', 5.0)
        self.declare_parameter('tf_track_max_active_per_class', 8)
        self.declare_parameter('tf_track_reset_stamp_jump_sec', 5.0)
        self.declare_parameter('tf_position_stabilization_enabled', False)
        self.declare_parameter('tf_position_deadband_m', 0.0)
        self.declare_parameter('tf_position_smoothing_alpha', 0.20)
        self.declare_parameter('tf_position_max_jump_m', 0.04)
        self.declare_parameter(
            'tf_position_relocation_confirmation_frames', 2
        )
        self.declare_parameter(
            'tf_position_relocation_consistency_m', 0.015
        )
        self.declare_parameter('tf_position_max_missed_frames', 3)
        self.declare_parameter('tf_axis_stabilization_enabled', False)
        self.declare_parameter('tf_axis_flip_confirmation_frames', 3)
        self.declare_parameter('tf_axis_flip_dot_threshold', 0.0)
        self.declare_parameter('tf_axis_pending_consistency_dot', 0.85)
        self.declare_parameter('tf_axis_max_missed_frames', 3)

    def _read_parameters(self) -> None:
        """Read and fail closed on incomplete metric geometry configuration."""
        def value(name: str) -> Any:
            return self.get_parameter(name).value

        self._rgb_topic = str(value('rgb_topic'))
        self._color_info_topic = str(value('color_camera_info_topic'))
        self._depth_topic = str(value('depth_topic'))
        self._depth_info_topic = str(value('depth_camera_info_topic'))
        self._extrinsics_topic = str(value('extrinsics_topic')).strip()
        self._require_extrinsics_topic = bool(
            value('require_extrinsics_topic')
        )
        self._depth_aligned_to_color = bool(value('depth_aligned_to_color'))
        self._depth_alignment_mode = (
            'color_aligned' if self._depth_aligned_to_color else 'native'
        )
        if self._depth_aligned_to_color and self._require_extrinsics_topic:
            raise ValueError(
                'color-aligned depth must not require an extrinsics topic'
            )
        if self._require_extrinsics_topic and not self._extrinsics_topic:
            raise ValueError(
                'extrinsics_topic is required for native-depth 3D pose'
            )
        self._pose_topic = str(value('pose_topic'))
        self._observation_topic = str(value('observation_topic'))
        self._publish_class_masks = bool(value('publish_class_masks'))
        self._class_mask_topic_prefix = str(
            value('class_mask_topic_prefix')
        ).strip().rstrip('/')
        if (
            self._publish_class_masks
            and not self._class_mask_topic_prefix.startswith('/')
        ):
            raise ValueError('class_mask_topic_prefix must be an absolute topic')
        self._class_mask_names = tuple(
            str(item).strip() for item in value('class_mask_names')
        )
        if self._publish_class_masks and not self._class_mask_names:
            raise ValueError('class_mask_names must not be empty')
        if any(not item for item in self._class_mask_names):
            raise ValueError('class_mask_names must not contain empty names')
        class_mask_slugs = tuple(
            class_mask_slug(item) for item in self._class_mask_names
        )
        if len(set(self._class_mask_names)) != len(self._class_mask_names):
            raise ValueError('class_mask_names must be unique')
        if len(set(class_mask_slugs)) != len(class_mask_slugs):
            raise ValueError('class_mask_names produce duplicate topic suffixes')
        self._class_mask_topics = (
            {
                name: f'{self._class_mask_topic_prefix}/{slug}'
                for name, slug in zip(self._class_mask_names, class_mask_slugs)
            }
            if self._publish_class_masks
            else {}
        )
        self._overlay_topic = str(value('overlay_topic'))
        self._pose_overlay_topic = str(value('pose_overlay_topic'))
        self._diagnostics_topic = str(value('diagnostics_topic'))
        self._health_topic = str(value('health_topic'))
        self._view = str(value('view'))
        self._workspace_zone = str(value('workspace_zone')).strip()
        if not self._workspace_zone:
            raise ValueError('workspace_zone must not be empty')
        self._workspace_roi_profile = str(
            value('workspace_roi_profile')
        ).strip()
        if not self._workspace_roi_profile:
            raise ValueError('workspace_roi_profile must not be empty')
        self._maximum_stamp_delta_ns = int(value('maximum_stamp_delta_ns'))
        self._sync_queue_size = int(value('sync_queue_size'))
        self._latest_frame_only = bool(value('latest_frame_only'))
        self._opencv_num_threads = int(value('opencv_num_threads'))
        if not 1 <= self._opencv_num_threads <= 8:
            raise ValueError('opencv_num_threads must be in [1, 8]')
        cv2.setUseOptimized(True)
        cv2.setNumThreads(self._opencv_num_threads)
        self._input_freshness_sec = float(value('input_freshness_sec'))
        self._processing_enabled = bool(value('processing_enabled'))
        self._processing_gate_topic = str(value('processing_gate_topic')).strip()
        self._require_depth = bool(value('require_depth'))
        if self._maximum_stamp_delta_ns < 0 or self._sync_queue_size < 1:
            raise ValueError(
                'timestamp tolerance and sync queue size are invalid'
            )

        algorithm_path = str(value('algorithm_python_path')).strip()
        checkpoint = str(value('checkpoint')).strip()
        ontology = str(value('ontology')).strip()
        self._algorithm_python_path = (
            Path(algorithm_path).expanduser() if algorithm_path else None
        )
        if not checkpoint or not ontology:
            raise ValueError(
                'checkpoint and ontology paths must be configured'
            )
        self._checkpoint = Path(checkpoint).expanduser()
        self._ontology = Path(ontology).expanduser()
        self._model_size = str(value('model_size')).strip().lower()
        if self._model_size not in ('small', 'medium', 'large', 'xlarge'):
            raise ValueError(
                'model_size must be one of small, medium, large, xlarge'
            )
        self._checkpoint_color_order = str(
            value('checkpoint_color_order')
        ).strip().upper()
        if self._checkpoint_color_order not in ('RGB', 'BGR'):
            raise ValueError(
                'checkpoint_color_order must be RGB or BGR'
            )
        self._model_version = str(value('model_version')).strip() or None
        self._trt_server_socket = str(value('trt_server_socket')).strip()
        self._trt_request_timeout_sec = float(
            value('trt_request_timeout_sec')
        )
        if (
            not np.isfinite(self._trt_request_timeout_sec)
            or self._trt_request_timeout_sec <= 0.0
        ):
            raise ValueError('trt_request_timeout_sec must be positive')
        self._confidence_threshold = float(value('confidence_threshold'))
        self._adson_forceps_confidence_threshold = float(
            value('adson_forceps_confidence_threshold')
        )
        if self._adson_forceps_confidence_threshold != -1.0 and not (
            0.0 <= self._adson_forceps_confidence_threshold <= 1.0
        ):
            raise ValueError(
                'adson_forceps_confidence_threshold must be -1 or in [0, 1]'
            )
        self._bovie_confidence_threshold = float(
            value('bovie_confidence_threshold')
        )
        if self._bovie_confidence_threshold != -1.0 and not (
            0.0 <= self._bovie_confidence_threshold <= 1.0
        ):
            raise ValueError(
                'bovie_confidence_threshold must be -1 or in [0, 1]'
            )
        enabled_class_thresholds = tuple(
            threshold
            for threshold in (
                self._adson_forceps_confidence_threshold,
                self._bovie_confidence_threshold,
            )
            if threshold >= 0.0
        )
        self._inference_confidence_threshold = min(
            (self._confidence_threshold, *enabled_class_thresholds)
        )
        self._enable_class_agnostic_nms = bool(
            value('enable_class_agnostic_nms')
        )
        self._class_agnostic_nms_iou = float(
            value('class_agnostic_nms_iou')
        )
        self._mask_component_cleanup_enabled = bool(
            value('mask_component_cleanup_enabled')
        )
        self._mask_component_minimum_area_px = int(
            value('mask_component_minimum_area_px')
        )
        self._mask_component_minimum_area_ratio = float(
            value('mask_component_minimum_area_ratio')
        )
        if self._mask_component_minimum_area_px < 1:
            raise ValueError('mask_component_minimum_area_px must be positive')
        if not 0.0 <= self._mask_component_minimum_area_ratio <= 1.0:
            raise ValueError(
                'mask_component_minimum_area_ratio must be in [0, 1]'
            )
        self._workspace_roi_enabled = bool(value('workspace_roi_enabled'))
        self._workspace_roi_polygon_norm_xy = tuple(
            float(item) for item in value('workspace_roi_polygon_norm_xy')
        )
        self._workspace_roi_minimum_mask_overlap = float(
            value('workspace_roi_minimum_mask_overlap')
        )
        self._workspace_roi_require_mask_centroid_inside = bool(
            value('workspace_roi_require_mask_centroid_inside')
        )
        self._temporal_class_smoothing_enabled = bool(
            value('temporal_class_smoothing_enabled')
        )
        self._temporal_class_history_size = int(
            value('temporal_class_history_size')
        )
        self._temporal_class_minimum_switch_frames = int(
            value('temporal_class_minimum_switch_frames')
        )
        self._temporal_class_switch_score_margin = float(
            value('temporal_class_switch_score_margin')
        )
        self._temporal_association_minimum_mask_iou = float(
            value('temporal_association_minimum_mask_iou')
        )
        self._temporal_association_minimum_bbox_iou = float(
            value('temporal_association_minimum_bbox_iou')
        )
        self._temporal_association_maximum_centroid_distance_norm = float(
            value('temporal_association_maximum_centroid_distance_norm')
        )
        self._temporal_association_maximum_mask_area_ratio = float(
            value('temporal_association_maximum_mask_area_ratio')
        )
        self._temporal_track_max_missed_frames = int(
            value('temporal_track_max_missed_frames')
        )
        self._positive_y_image_direction = str(
            value('positive_y_image_direction')
        ).strip().lower()
        self._adson_face_on_width_enabled = bool(
            value('adson_face_on_width_enabled')
        )
        if self._positive_y_image_direction not in (
            'class_based',
            'down',
            'right',
        ):
            raise ValueError(
                'positive_y_image_direction must be class_based, down, or right'
            )
        self._pose_axis_length_m = float(value('pose_axis_length_m'))
        if not 0.0 <= self._class_agnostic_nms_iou <= 1.0:
            raise ValueError('class_agnostic_nms_iou must be in [0, 1]')
        if not np.isfinite(self._pose_axis_length_m) or (
            self._pose_axis_length_m <= 0.0
        ):
            raise ValueError('pose_axis_length_m must be positive')
        self._optimize = bool(value('optimize_for_inference'))
        self._jit_compile = bool(value('jit_compile'))
        self._fp16 = bool(value('fp16'))
        self._jpeg_quality = int(value('jpeg_quality'))

        self._depth_scale = float(value('depth_scale_m_per_unit'))
        self._depth_scale_verified = bool(value('depth_scale_verified'))
        self._depth_registration_backend = str(
            value('depth_registration_backend')
        ).strip().lower()
        self._depth_registration_allow_numpy_fallback = bool(
            value('depth_registration_allow_numpy_fallback')
        )
        self._depth_registration_cuda_library = str(
            value('depth_registration_cuda_library')
        ).strip()
        if self._depth_registration_backend not in ('numpy', 'cuda'):
            raise ValueError(
                'depth_registration_backend must be numpy or cuda'
            )
        if (
            self._depth_registration_backend == 'cuda'
            and not self._depth_registration_cuda_library
        ):
            raise ValueError(
                'depth_registration_cuda_library is required for CUDA depth '
                'registration'
            )
        self._minimum_depth_m = float(value('minimum_depth_m'))
        self._maximum_depth_m = float(value('maximum_depth_m'))
        if not np.isfinite(self._depth_scale) or self._depth_scale <= 0.0:
            raise ValueError(
                'depth_scale_m_per_unit must be configured and positive'
            )
        if not 0.0 <= self._minimum_depth_m < self._maximum_depth_m:
            raise ValueError('invalid minimum/maximum depth range')
        self._minimum_extrinsics_baseline_m = float(
            value('minimum_depth_to_color_baseline_m')
        )
        self._maximum_extrinsics_baseline_m = float(
            value('maximum_depth_to_color_baseline_m')
        )
        self._expected_extrinsics_direction = value(
            'expected_depth_to_color_translation_direction'
        )
        self._minimum_extrinsics_direction_cosine = float(
            value('minimum_depth_to_color_direction_cosine')
        )
        self._extrinsics_orthonormal_tolerance = float(
            value('depth_to_color_rotation_orthonormal_tolerance')
        )
        self._extrinsics_determinant_tolerance = float(
            value('depth_to_color_rotation_determinant_tolerance')
        )
        self._depth_to_color_rotation_convention = str(
            value('depth_to_color_rotation_convention')
        ).strip().lower()
        if self._depth_to_color_rotation_convention != 'column_major':
            raise ValueError(
                'depth_to_color_rotation_convention must be column_major'
            )
        self._expected_color_frame = str(value('expected_color_frame'))
        self._expected_depth_frame = str(value('expected_depth_frame'))
        self._calibration_version = str(value('calibration_version'))
        if not self._calibration_version:
            raise ValueError('calibration_version is required')

        # These values document a camera-specific reference transform only.
        # Runtime 3D pose never falls back to it: a valid latched topic is
        # mandatory unless an explicitly legacy profile opts out.
        reference_rotation = np.asarray(
            value('depth_to_color_rotation'), dtype=np.float64
        ).reshape(-1)
        reference_translation = np.asarray(
            value('depth_to_color_translation_m'), dtype=np.float64
        ).reshape(-1)
        has_reference = bool(
            np.any(np.isfinite(reference_rotation))
            or np.any(np.isfinite(reference_translation))
        )
        self._reference_extrinsics: DepthToColorExtrinsics | None = None
        if has_reference:
            try:
                self._reference_extrinsics = (
                    validate_depth_to_color_extrinsics(
                        reference_rotation,
                        reference_translation,
                        minimum_baseline_m=self._minimum_extrinsics_baseline_m,
                        maximum_baseline_m=self._maximum_extrinsics_baseline_m,
                        expected_translation_direction=(
                            self._expected_extrinsics_direction
                        ),
                        minimum_direction_cosine=(
                            self._minimum_extrinsics_direction_cosine
                        ),
                        orthonormal_tolerance=(
                            self._extrinsics_orthonormal_tolerance
                        ),
                        determinant_tolerance=(
                            self._extrinsics_determinant_tolerance
                        ),
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f'configured depth-to-color reference is invalid: {error}'
                ) from error
        if (
            not self._require_extrinsics_topic
            and self._reference_extrinsics is None
            and not self._depth_aligned_to_color
        ):
            raise ValueError(
                'a legacy no-topic profile requires a complete reference '
                'depth-to-color transform'
            )

        self._plane_normal = finite_vector(
            'support_plane_normal', value('support_plane_normal'), 3
        )
        self._plane_offset = float(value('support_plane_offset_m'))
        self._plane_version = str(value('support_plane_config_version'))
        self._plane_inlier_ratio = float(value('support_plane_inlier_ratio'))
        self._plane_residual_p95_m = float(
            value('support_plane_residual_p95_m')
        )
        if not np.isfinite(self._plane_offset) or not self._plane_version:
            raise ValueError(
                'support-plane offset and config version are required'
            )

        self._publish_tool_tf = bool(value('publish_tool_tf'))
        self._tf_stale_after_sec = float(value('tf_stale_after_sec'))
        self._tf_max_future_sec = float(value('tf_max_future_sec'))
        if (
            not np.isfinite(self._tf_stale_after_sec)
            or self._tf_stale_after_sec <= 0.0
        ):
            raise ValueError('tf_stale_after_sec must be positive')
        if (
            not np.isfinite(self._tf_max_future_sec)
            or self._tf_max_future_sec < 0.0
        ):
            raise ValueError('tf_max_future_sec must be non-negative')
        self._tf_track_max_displacement_m = float(
            value('tf_track_max_displacement_m')
        )
        self._tf_track_ttl_sec = float(value('tf_track_ttl_sec'))
        self._tf_track_max_active_per_class = int(
            value('tf_track_max_active_per_class')
        )
        self._tf_track_reset_stamp_jump_sec = float(
            value('tf_track_reset_stamp_jump_sec')
        )
        self._tf_position_stabilization_enabled = bool(
            value('tf_position_stabilization_enabled')
        )
        self._tf_position_deadband_m = float(
            value('tf_position_deadband_m')
        )
        self._tf_position_smoothing_alpha = float(
            value('tf_position_smoothing_alpha')
        )
        self._tf_position_max_jump_m = float(
            value('tf_position_max_jump_m')
        )
        self._tf_position_relocation_confirmation_frames = int(
            value('tf_position_relocation_confirmation_frames')
        )
        self._tf_position_relocation_consistency_m = float(
            value('tf_position_relocation_consistency_m')
        )
        self._tf_position_max_missed_frames = int(
            value('tf_position_max_missed_frames')
        )
        self._tf_axis_stabilization_enabled = bool(
            value('tf_axis_stabilization_enabled')
        )
        self._tf_axis_flip_confirmation_frames = int(
            value('tf_axis_flip_confirmation_frames')
        )
        self._tf_axis_flip_dot_threshold = float(
            value('tf_axis_flip_dot_threshold')
        )
        self._tf_axis_pending_consistency_dot = float(
            value('tf_axis_pending_consistency_dot')
        )
        self._tf_axis_max_missed_frames = int(
            value('tf_axis_max_missed_frames')
        )
        self._tf_tracker = ToolSpatialTfSelector(
            max_tools_per_class=self._tf_track_max_active_per_class,
            reset_stamp_jump_sec=self._tf_track_reset_stamp_jump_sec,
            position_stabilization_enabled=(
                self._tf_position_stabilization_enabled
            ),
            position_deadband_m=self._tf_position_deadband_m,
            position_smoothing_alpha=self._tf_position_smoothing_alpha,
            position_max_jump_m=self._tf_position_max_jump_m,
            position_relocation_confirmation_frames=(
                self._tf_position_relocation_confirmation_frames
            ),
            position_relocation_consistency_m=(
                self._tf_position_relocation_consistency_m
            ),
            position_max_missed_frames=(
                self._tf_position_max_missed_frames
            ),
            axis_stabilization_enabled=(
                self._tf_axis_stabilization_enabled
            ),
            axis_flip_confirmation_frames=(
                self._tf_axis_flip_confirmation_frames
            ),
            axis_flip_dot_threshold=self._tf_axis_flip_dot_threshold,
            axis_pending_consistency_dot=(
                self._tf_axis_pending_consistency_dot
            ),
            axis_max_missed_frames=self._tf_axis_max_missed_frames,
        )

        self._additional_status_flags: list[str] = []
        if not self._depth_scale_verified:
            self._additional_status_flags.append('DEPTH_SCALE_UNVERIFIED')
        if 'provisional' in self._plane_version.lower():
            self._additional_status_flags.append('SUPPORT_PLANE_PROVISIONAL')
        if 'provisional' in self._calibration_version.lower():
            self._additional_status_flags.append('CALIBRATION_PROVISIONAL')

    def _receive_color_info(self, message: CameraInfo) -> None:
        """Cache the latest color-camera calibration."""
        with self._condition:
            self._color_info = message

    def _receive_depth_info(self, message: CameraInfo) -> None:
        """Cache the latest depth-camera calibration."""
        with self._condition:
            self._depth_info = message

    def _receive_depth_to_color_extrinsics(
        self, message: Extrinsics
    ) -> None:
        """Accept only a physically valid latched depth-to-color transform."""
        try:
            extrinsics = validate_depth_to_color_extrinsics(
                message.rotation,
                message.translation,
                minimum_baseline_m=self._minimum_extrinsics_baseline_m,
                maximum_baseline_m=self._maximum_extrinsics_baseline_m,
                expected_translation_direction=(
                    self._expected_extrinsics_direction
                ),
                minimum_direction_cosine=(
                    self._minimum_extrinsics_direction_cosine
                ),
                orthonormal_tolerance=self._extrinsics_orthonormal_tolerance,
                determinant_tolerance=self._extrinsics_determinant_tolerance,
            )
        except ValueError as error:
            reason = str(error)
            with self._condition:
                self._received_extrinsics += 1
                self._rejected_extrinsics += 1
                self._depth_to_color_extrinsics = None
                self._extrinsics_revision += 1
                self._registrar = None
                self._registrar_key = None
                self._registrar_ready = False
                self._last_extrinsics_error = reason
                self._last_error_code = 'DEPTH_TO_COLOR_EXTRINSICS_INVALID'
                self._last_error_message = reason
            self.get_logger().error(
                f'rejected depth-to-color extrinsics: {reason}'
            )
            return

        with self._condition:
            self._received_extrinsics += 1
            self._depth_to_color_extrinsics = extrinsics
            self._extrinsics_revision += 1
            self._registrar = None
            self._registrar_key = None
            self._registrar_ready = False
            self._last_extrinsics_error = ''
            if self._last_error_code == 'DEPTH_TO_COLOR_EXTRINSICS_INVALID':
                self._last_error_code = ''
                self._last_error_message = ''
        self.get_logger().info(
            'accepted depth-to-color extrinsics '
            f'(baseline={extrinsics.baseline_m:.6f} m)'
        )

    def _active_extrinsics_locked(
        self,
    ) -> DepthToColorExtrinsics | None:
        """Return a topic transform, or an explicit legacy reference only."""
        if self._depth_to_color_extrinsics is not None:
            return self._depth_to_color_extrinsics
        if not self._require_extrinsics_topic:
            return self._reference_extrinsics
        return None

    def _receive_processing_gate(self, message: Bool) -> None:
        """Enable or pause inference while keeping the model preloaded."""
        enabled = bool(message.data)
        with self._condition:
            changed = self._processing_enabled != enabled
            self._processing_enabled = enabled
            if not enabled:
                self._pending = None
                if changed:
                    self._tf_tracker.reset()
                    self._tf_last_output_source_stamp_ns = None
                    self._tf_last_parent_frame = ''
                    self._tf_last_child_frames = ()
                    self._tf_last_skip_reason = 'SELECTOR_RESET_GATE_INACTIVE'
            self._condition.notify_all()
        if changed:
            self.get_logger().info(
                f'processing gate: {"ACTIVE" if enabled else "INACTIVE"}'
            )

    def _receive_rgb(self, message: CompressedImage) -> None:
        """Run recognition on RGB. Attach depth only when a fresh pair exists."""
        with self._condition:
            self._received_rgb += 1
            pair = self._pairer.add_rgb(message)
            if pair is not None:
                self._queue_pair(pair)
                return
            if self._require_depth:
                return
            self._queue_rgb_only(message)

    def _receive_depth(self, message: CompressedImage) -> None:
        """Cache native-depth for pairing. RGB-only recognition does not wait."""
        with self._condition:
            self._received_depth += 1
            pair = self._pairer.add_depth(message)
            if pair is not None:
                self._queue_pair(pair)

    def _queue_pair(self, pair: RgbDepthPair) -> None:
        """Queue a complete RGB-D pair for the latest-frame worker."""
        self._paired_frames += 1
        self._last_pair_monotonic = time.monotonic()
        if not self._processing_enabled:
            return
        pending = PendingPoseFrame(
            rgb=pair.rgb,
            depth=pair.depth,
            color_info=self._color_info,
            depth_info=self._depth_info,
            depth_to_color_extrinsics=self._active_extrinsics_locked(),
            extrinsics_revision=self._extrinsics_revision,
            rgb_depth_delta_ns=pair.delta_ns,
            received_monotonic=time.monotonic(),
        )
        self._set_pending(pending)

    def _queue_rgb_only(self, message: CompressedImage) -> None:
        """Queue RGB recognition when depth is absent or unmatched."""
        self._last_pair_monotonic = time.monotonic()
        if not self._processing_enabled:
            return
        pending = PendingPoseFrame(
            rgb=message,
            depth=None,
            color_info=self._color_info,
            depth_info=self._depth_info,
            depth_to_color_extrinsics=self._active_extrinsics_locked(),
            extrinsics_revision=self._extrinsics_revision,
            rgb_depth_delta_ns=None,
            received_monotonic=time.monotonic(),
        )
        self._set_pending(pending)

    def _set_pending(self, pending: PendingPoseFrame) -> None:
        """Keep only the newest pending frame when latest-frame mode is on."""
        if self._pending is not None:
            if not self._latest_frame_only:
                self._dropped_pending_frames += 1
                return
            self._dropped_pending_frames += 1
        self._pending = pending
        self._condition.notify()

    def _load_algorithm(self) -> None:
        """Load the non-ROS algorithm and the configured support plane."""
        if self._algorithm_python_path is not None:
            algorithm_path = str(self._algorithm_python_path.resolve())
            if not self._algorithm_python_path.is_dir():
                raise FileNotFoundError(
                    f'algorithm_python_path not found: {algorithm_path}'
                )
            if algorithm_path not in sys.path:
                sys.path.insert(0, algorithm_path)
        if not self._checkpoint.is_file():
            raise FileNotFoundError(
                f'checkpoint not found: {self._checkpoint}'
            )
        if not self._ontology.is_file():
            raise FileNotFoundError(f'ontology not found: {self._ontology}')
        from pnu_surgical_tool import (  # Imported after path configuration.
            CameraCalibration,
            decode_compressed_depth_16uc1,
            DepthToColorRegistrar,
            DetectionPostprocessor,
            DetectionPostprocessorConfig,
            DetectorConfig,
            PlanarPoseConfig,
            PlanarPoseEstimator,
            RigidTransform,
            SmallComponentCleanupConfig,
            SupportPlane,
            SurgicalToolAlgorithm,
            SurgicalToolDetector,
            TemporalClassConfig,
            WorkspaceRoiConfig,
            draw_pose_axes_bgr,
        )

        detector = SurgicalToolDetector(
            DetectorConfig(
                checkpoint_path=self._checkpoint,
                ontology_path=self._ontology,
                model_size=self._model_size,
                confidence_threshold=self._confidence_threshold,
                enable_class_agnostic_nms=self._enable_class_agnostic_nms,
                class_agnostic_nms_iou=self._class_agnostic_nms_iou,
                optimize=self._optimize,
                jit_compile=self._jit_compile,
                fp16=self._fp16,
                model_version=self._model_version,
                checkpoint_color_order=self._checkpoint_color_order,
                trt_server_socket=self._trt_server_socket or None,
                trt_camera_key=self._view,
                trt_request_timeout_sec=self._trt_request_timeout_sec,
            )
        )
        detector.load()
        self._detector = detector
        class_confidence_thresholds = []
        if self._adson_forceps_confidence_threshold >= 0.0:
            class_confidence_thresholds.append((
                'Adson Forceps',
                self._adson_forceps_confidence_threshold,
            ))
        if self._bovie_confidence_threshold >= 0.0:
            class_confidence_thresholds.append((
                'Bovie',
                self._bovie_confidence_threshold,
            ))
        self._detection_postprocessor = DetectionPostprocessor(
            DetectionPostprocessorConfig(
                default_class_confidence_threshold=(
                    self._confidence_threshold
                ),
                class_confidence_thresholds=tuple(
                    class_confidence_thresholds
                ),
                small_component_cleanup=SmallComponentCleanupConfig(
                    enabled=self._mask_component_cleanup_enabled,
                    minimum_area_px=self._mask_component_minimum_area_px,
                    minimum_area_ratio=(
                        self._mask_component_minimum_area_ratio
                    ),
                ),
                workspace_roi=WorkspaceRoiConfig(
                    enabled=self._workspace_roi_enabled,
                    polygon_norm_xy=self._workspace_roi_polygon_norm_xy,
                    minimum_mask_overlap=(
                        self._workspace_roi_minimum_mask_overlap
                    ),
                    require_mask_centroid_inside=(
                        self._workspace_roi_require_mask_centroid_inside
                    ),
                ),
                temporal_class=TemporalClassConfig(
                    enabled=self._temporal_class_smoothing_enabled,
                    history_size=self._temporal_class_history_size,
                    minimum_switch_frames=(
                        self._temporal_class_minimum_switch_frames
                    ),
                    switch_score_margin=(
                        self._temporal_class_switch_score_margin
                    ),
                    minimum_mask_iou=(
                        self._temporal_association_minimum_mask_iou
                    ),
                    minimum_bbox_iou=(
                        self._temporal_association_minimum_bbox_iou
                    ),
                    maximum_centroid_distance_norm=(
                        self._temporal_association_maximum_centroid_distance_norm
                    ),
                    maximum_mask_area_ratio=(
                        self._temporal_association_maximum_mask_area_ratio
                    ),
                    max_missed_frames=self._temporal_track_max_missed_frames,
                ),
            )
        )
        self._algorithm = SurgicalToolAlgorithm(
            detector,
            PlanarPoseEstimator(
                PlanarPoseConfig(
                    positive_y_image_direction=(
                        self._positive_y_image_direction
                    ),
                    adson_face_on_width_enabled=(
                        self._adson_face_on_width_enabled
                    ),
                )
            ),
            postprocessor=self._detection_postprocessor,
        )
        self._support_plane = SupportPlane(
            normal=self._plane_normal,
            offset_m=self._plane_offset,
            config_version=self._plane_version,
            inlier_ratio=self._plane_inlier_ratio,
            residual_p95_m=self._plane_residual_p95_m,
        )
        self._algorithm_symbols = {
            'CameraCalibration': CameraCalibration,
            'decode_depth': decode_compressed_depth_16uc1,
            'DepthToColorRegistrar': DepthToColorRegistrar,
            'RigidTransform': RigidTransform,
            'draw_pose_axes_bgr': draw_pose_axes_bgr,
        }

    def _worker_loop(self) -> None:
        """Process the newest RGB frame, sampling depth when a pair exists."""
        try:
            self._load_algorithm()
        except Exception as error:
            self._set_error('MODEL_OR_CONFIG_LOAD_FAILED', str(error))
            self.get_logger().error(f'pose algorithm load failed: {error}')
            return
        with self._condition:
            self._model_ready = True
            self._last_error_code = ''
            self._last_error_message = ''
        self.get_logger().info('RF-DETR and planar-pose algorithm loaded')

        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                pending = self._pending
                self._pending = None
            if pending is not None:
                self._process_frame(pending)

    def _queue_overlays(self, pending: PendingOverlayFrame) -> None:
        """Replace any stale visualization job without delaying ROS results."""
        with self._condition:
            if self._overlay_pending is not None:
                self._overlay_dropped_pending_frames += 1
            self._overlay_pending = pending
            self._condition.notify_all()

    def _overlay_worker_loop(self) -> None:
        """Render only the most recent completed perception frame."""
        while True:
            with self._condition:
                while self._overlay_pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                pending = self._overlay_pending
                self._overlay_pending = None
            if pending is None:
                continue
            started = time.perf_counter()
            error_message = ''
            try:
                if (
                    pending.pose_result is not None
                    and pending.pose_camera is not None
                ):
                    self._publish_pose_overlay(
                        pending.rgb,
                        pending.pose_result,
                        pending.pose_camera,
                        pending.header,
                    )
                self._publish_detection_overlay(
                    pending.rgb,
                    pending.detections,
                    pending.observation_array,
                    list(pending.selector_labels),
                    pending.header,
                )
            except Exception as error:
                error_message = f'{type(error).__name__}: {error}'
                self.get_logger().warning(
                    f'latest-only tool overlay failed: {error_message}'
                )
            finally:
                latency_ms = (time.perf_counter() - started) * 1000.0
                with self._condition:
                    self._overlay_processed_frames += 1
                    self._last_overlay_latency_ms = latency_ms
                    self._last_overlay_error = error_message

    def _empty_tf_report(self) -> dict[str, Any]:
        return {
            'enabled': self._publish_tool_tf,
            'published_count': 0,
            'skipped_count': 0,
            'source_age_sec': None,
            'skip_reason': 'POSE_RESULT_UNAVAILABLE',
            'active_track_count': self._tf_tracker.active_track_count,
            'track_created_total': self._tf_tracker.created_total,
            'track_expired_total': self._tf_tracker.expired_total,
            'track_rejected_total': self._tf_tracker.rejected_total,
            'track_reset_total': self._tf_tracker.reset_total,
            'position_filter_active_count': (
                self._tf_tracker.position_filter_active_count
            ),
            'position_filter_held_total': (
                self._tf_tracker.position_filter_held_total
            ),
            'position_filter_smoothed_total': (
                self._tf_tracker.position_filter_smoothed_total
            ),
            'position_filter_outlier_held_total': (
                self._tf_tracker.position_filter_outlier_held_total
            ),
            'position_filter_relocation_total': (
                self._tf_tracker.position_filter_relocation_total
            ),
            'position_filter_association_reset_total': (
                self._tf_tracker.position_filter_association_reset_total
            ),
        }

    def _output_worker_loop(self) -> None:
        """Publish only complete, newest pose/observation/mask bundles."""
        while True:
            bundle = self._output_slot.take()
            if bundle is None or self._stopping:
                return
            output_started = time.perf_counter()
            output_cpu_started = time.thread_time()
            queue_wait_ms = (
                output_started - bundle.queued_monotonic
            ) * 1000.0
            try:
                pose_publish_ms = 0.0
                pose_publish_cpu_ms = 0.0
                pose_publish_completed = None
                tf_publish_ms = 0.0
                tf_publish_cpu_ms = 0.0
                if bundle.pose_array is not None:
                    stage_started = time.perf_counter()
                    stage_cpu_started = time.thread_time()
                    self._pose_publisher.publish(bundle.pose_array)
                    pose_publish_ms = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    pose_publish_cpu_ms = (
                        time.thread_time() - stage_cpu_started
                    ) * 1000.0
                    pose_publish_completed = time.perf_counter()
                    stage_started = time.perf_counter()
                    stage_cpu_started = time.thread_time()
                    tf_report = self._publish_constrained_tool_tf(
                        bundle.pose_array,
                        selector_u_by_instance_id=(
                            bundle.selector_u_by_instance_id
                        ),
                    )
                    tf_publish_ms = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                    tf_publish_cpu_ms = (
                        time.thread_time() - stage_cpu_started
                    ) * 1000.0
                else:
                    tf_report = self._empty_tf_report()

                stage_started = time.perf_counter()
                stage_cpu_started = time.thread_time()
                self._observation_publisher.publish(
                    bundle.observation_array
                )
                observation_publish_ms = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                observation_publish_cpu_ms = (
                    time.thread_time() - stage_cpu_started
                ) * 1000.0
                observation_publish_completed = time.perf_counter()
                mask_publish_started = time.perf_counter()
                mask_publish_cpu_started = time.thread_time()
                mask_publish_ms_by_class: dict[str, float] = {}
                for class_name, message in bundle.class_mask_messages:
                    stage_started = time.perf_counter()
                    self._class_mask_publishers[class_name].publish(message)
                    mask_publish_ms_by_class[class_name] = (
                        time.perf_counter() - stage_started
                    ) * 1000.0
                class_masks_publish_ms = (
                    time.perf_counter() - mask_publish_started
                ) * 1000.0
                class_masks_publish_cpu_ms = (
                    time.thread_time() - mask_publish_cpu_started
                ) * 1000.0
                masks_publish_completed = time.perf_counter()

                publish_ms = (
                    time.perf_counter() - output_started
                ) * 1000.0
                publish_thread_cpu_ms = (
                    time.thread_time() - output_cpu_started
                ) * 1000.0
                latency_ms = (
                    time.perf_counter() - bundle.process_started
                ) * 1000.0
                source_age_ms = max(
                    0.0,
                    (
                        self.get_clock().now().nanoseconds
                        - bundle.source_stamp_ns
                    ) / 1_000_000.0,
                )
                with self._condition:
                    if bundle.class_mask_messages:
                        self._class_mask_frames_published += 1
                    self._output_published_bundles += 1
                    published_total = self._output_published_bundles
                    self._last_output_bundle_queue_wait_ms = queue_wait_ms
                    self._last_output_bundle_publish_ms = publish_ms
                    self._last_output_bundle_latency_ms = latency_ms
                    self._last_output_bundle_source_age_ms = source_age_ms
                    self._last_output_error = ''

                diagnostics = dict(bundle.diagnostics)
                diagnostics.update({
                    'total_latency_ms': latency_ms,
                    'output_execution': 'async_latest_only_bundle',
                    'output_bundle_queue_wait_ms': queue_wait_ms,
                    'output_bundle_publish_ms': publish_ms,
                    'output_bundle_publish_thread_cpu_ms': (
                        publish_thread_cpu_ms
                    ),
                    'output_pose_publish_ms': pose_publish_ms,
                    'output_pose_publish_thread_cpu_ms': pose_publish_cpu_ms,
                    'output_pose_end_to_end_ms': (
                        (pose_publish_completed - bundle.process_started)
                        * 1000.0
                        if pose_publish_completed is not None
                        else None
                    ),
                    'output_tf_publish_ms': tf_publish_ms,
                    'output_tf_publish_thread_cpu_ms': tf_publish_cpu_ms,
                    'output_observation_publish_ms': observation_publish_ms,
                    'output_observation_publish_thread_cpu_ms': (
                        observation_publish_cpu_ms
                    ),
                    'output_observation_end_to_end_ms': (
                        observation_publish_completed
                        - bundle.process_started
                    ) * 1000.0,
                    'output_class_masks_publish_ms': class_masks_publish_ms,
                    'output_class_masks_publish_thread_cpu_ms': (
                        class_masks_publish_cpu_ms
                    ),
                    'output_class_mask_publish_ms_by_class': (
                        mask_publish_ms_by_class
                    ),
                    'output_class_masks_end_to_end_ms': (
                        masks_publish_completed - bundle.process_started
                    ) * 1000.0,
                    'output_class_mask_payload_bytes': sum(
                        len(message.data)
                        for _class_name, message in bundle.class_mask_messages
                    ),
                    'output_bundle_latency_ms': latency_ms,
                    'output_bundle_source_age_ms': source_age_ms,
                    'output_bundle_mask_count': len(
                        bundle.class_mask_messages
                    ),
                    'output_bundle_overwritten_total': (
                        self._output_slot.overwritten_total
                    ),
                    'output_bundle_published_total': published_total,
                    'class_mask_frames_published': (
                        self._class_mask_frames_published
                    ),
                    'tf_published_count': tf_report['published_count'],
                    'tf_skipped_count': tf_report['skipped_count'],
                    'tf_source_age_sec': tf_report['source_age_sec'],
                    'tf_skip_reason': tf_report['skip_reason'],
                    'tf_active_track_count': tf_report['active_track_count'],
                    'tf_track_created_total': tf_report[
                        'track_created_total'
                    ],
                    'tf_track_expired_total': tf_report[
                        'track_expired_total'
                    ],
                    'tf_track_rejected_total': tf_report[
                        'track_rejected_total'
                    ],
                    'tf_track_reset_total': tf_report['track_reset_total'],
                    'tf_position_filter_active_count': tf_report[
                        'position_filter_active_count'
                    ],
                    'tf_position_filter_held_total': tf_report[
                        'position_filter_held_total'
                    ],
                    'tf_position_filter_smoothed_total': tf_report[
                        'position_filter_smoothed_total'
                    ],
                    'tf_position_filter_outlier_held_total': tf_report[
                        'position_filter_outlier_held_total'
                    ],
                    'tf_position_filter_relocation_total': tf_report[
                        'position_filter_relocation_total'
                    ],
                    'tf_position_filter_association_reset_total': tf_report[
                        'position_filter_association_reset_total'
                    ],
                })
                self._diagnostics_publisher.publish(String(
                    data=json.dumps(diagnostics, separators=(',', ':'))
                ))
                now_monotonic = time.monotonic()
                if (
                    now_monotonic - self._last_publish_log_monotonic
                    >= 1.0
                ):
                    self.get_logger().info(
                        f'published v1.6 tools: {bundle.instance_count} '
                        f'instances; valid_poses={bundle.valid_pose_count}; '
                        f'output_bundle_ms={latency_ms:.1f}; '
                        f'overwritten={self._output_slot.overwritten_total}'
                    )
                    self._last_publish_log_monotonic = now_monotonic
                with self._condition:
                    self._processed_frames += 1
                    self._last_success_monotonic = time.monotonic()
                    self._last_error_code = ''
                    self._last_error_message = ''
            except Exception as error:
                if self._stopping or not self.context.ok():
                    return
                message = f'{type(error).__name__}: {error}'
                with self._condition:
                    self._last_output_error = message
                self._set_error('OUTPUT_BUNDLE_PUBLISH_FAILED', message)
                self.get_logger().error(
                    f'latest-only output bundle failed: {message}'
                )

    def _camera_calibration(self, info: CameraInfo, suffix: str) -> Any:
        """Convert a ROS CameraInfo to the core calibration type."""
        calibration_type = self._algorithm_symbols['CameraCalibration']
        return calibration_type(
            width=int(info.width),
            height=int(info.height),
            k=np.asarray(info.k, dtype=np.float64).reshape(3, 3),
            distortion=np.asarray(info.d, dtype=np.float64),
            frame_name=str(info.header.frame_id),
            calibration_version=f'{self._calibration_version}:{suffix}',
        )

    def _get_aligned_color_camera(
        self,
        color_info: CameraInfo,
        depth_info: CameraInfo,
        rgb_shape: tuple[int, int],
        depth_shape: tuple[int, int],
    ) -> Any:
        """Validate aligned-depth geometry and return the color calibration."""
        color_frame = str(color_info.header.frame_id)
        depth_frame = str(depth_info.header.frame_id)
        if self._expected_color_frame and color_frame != self._expected_color_frame:
            raise ValueError(
                f'color frame {color_frame!r} != expected '
                f'{self._expected_color_frame!r}'
            )
        if self._expected_depth_frame and depth_frame != self._expected_depth_frame:
            raise ValueError(
                f'depth frame {depth_frame!r} != expected '
                f'{self._expected_depth_frame!r}'
            )
        if color_frame != depth_frame:
            raise ValueError(
                'color-aligned depth must use the color optical frame: '
                f'color={color_frame!r}, depth={depth_frame!r}'
            )
        expected_rgb_shape = (int(color_info.height), int(color_info.width))
        expected_depth_shape = (int(depth_info.height), int(depth_info.width))
        if tuple(rgb_shape) != expected_rgb_shape:
            raise ValueError(
                f'RGB shape {tuple(rgb_shape)} != Color CameraInfo '
                f'{expected_rgb_shape}'
            )
        if tuple(depth_shape) != expected_depth_shape:
            raise ValueError(
                f'aligned depth shape {tuple(depth_shape)} != Depth CameraInfo '
                f'{expected_depth_shape}'
            )
        if expected_rgb_shape != expected_depth_shape:
            raise ValueError(
                'color-aligned depth and RGB CameraInfo dimensions differ'
            )
        camera = self._camera_calibration(color_info, 'aligned_color')
        with self._condition:
            self._aligned_depth_ready = True
        return camera

    def _get_registrar(
        self,
        color_info: CameraInfo,
        depth_info: CameraInfo,
        extrinsics: DepthToColorExtrinsics,
    ) -> Any:
        """Return a cached registrar, rebuilding it on calibration changes."""
        color_frame = str(color_info.header.frame_id)
        depth_frame = str(depth_info.header.frame_id)
        if (
            self._expected_color_frame
            and color_frame != self._expected_color_frame
        ):
            raise ValueError(
                f'color frame {color_frame!r} != expected '
                f'{self._expected_color_frame!r}'
            )
        if (
            self._expected_depth_frame
            and depth_frame != self._expected_depth_frame
        ):
            raise ValueError(
                f'depth frame {depth_frame!r} != expected '
                f'{self._expected_depth_frame!r}'
            )
        key = (
            camera_info_signature(color_info),
            camera_info_signature(depth_info),
            *tuple(extrinsics.rotation.ravel()),
            *tuple(extrinsics.translation_m),
        )
        if key == self._registrar_key and self._registrar is not None:
            return self._registrar

        color_camera = self._camera_calibration(color_info, 'color')
        depth_camera = self._camera_calibration(depth_info, 'depth')
        transform_type = self._algorithm_symbols['RigidTransform']
        transform = transform_type(
            rotation=extrinsics.rotation,
            translation_m=extrinsics.translation_m,
            source_frame=depth_frame,
            target_frame=color_frame,
            calibration_version=(
                f'{self._calibration_version}:live_depth_to_color'
            ),
        )
        registrar_type = self._algorithm_symbols['DepthToColorRegistrar']
        self._registrar = registrar_type(
            depth_camera,
            color_camera,
            transform,
            backend=self._depth_registration_backend,
            allow_sticky_numpy_fallback=(
                self._depth_registration_allow_numpy_fallback
            ),
            cuda_library_path=(
                self._depth_registration_cuda_library or None
            ),
        )
        self._registrar_key = key
        with self._condition:
            self._registrar_ready = True
        return self._registrar

    def _process_frame(self, pending: PendingPoseFrame) -> None:
        """Detect on RGB always; sample depth and estimate pose only if present."""
        started = time.perf_counter()
        process_cpu_started = time.thread_time()
        try:
            decode_started = time.perf_counter()
            decode_cpu_started = time.thread_time()
            rgb = decode_rgb(pending.rgb)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            decode_cpu_ms = (
                time.thread_time() - decode_cpu_started
            ) * 1000.0
            with self._condition:
                self._sequence += 1
                sequence = self._sequence
            detect_started = time.perf_counter()
            detect_cpu_started = time.thread_time()
            detections = self._algorithm.detect(
                rgb, 'BGR', self._inference_confidence_threshold
            )
            detect_ms = (time.perf_counter() - detect_started) * 1000.0
            detect_cpu_ms = (
                time.thread_time() - detect_cpu_started
            ) * 1000.0

            aligned_depth_m = None
            registration_ms = 0.0
            pose_result = None
            pose_camera = None
            inference_pose_ms = 0.0
            inference_pose_cpu_ms = 0.0
            with self._condition:
                extrinsics_ready = bool(
                    not self._depth_aligned_to_color
                    and pending.depth_to_color_extrinsics is not None
                    and pending.extrinsics_revision == self._extrinsics_revision
                    and self._active_extrinsics_locked()
                    is pending.depth_to_color_extrinsics
                )
            if (
                pending.depth is not None
                and pending.color_info is not None
                and pending.depth_info is not None
            ):
                native_depth = self._algorithm_symbols['decode_depth'](
                    pending.depth.data, pending.depth.format
                )
                registrar_started = time.perf_counter()
                if self._depth_aligned_to_color:
                    aligned_depth_m = aligned_depth_to_meters(
                        native_depth,
                        rgb.shape[:2],
                        self._depth_scale,
                        self._minimum_depth_m,
                        self._maximum_depth_m,
                    )
                    pose_camera = self._get_aligned_color_camera(
                        pending.color_info,
                        pending.depth_info,
                        rgb.shape[:2],
                        native_depth.shape,
                    )
                elif extrinsics_ready:
                    registrar = self._get_registrar(
                        pending.color_info,
                        pending.depth_info,
                        pending.depth_to_color_extrinsics,
                    )
                    registration = registrar.register(
                        native_depth,
                        self._depth_scale,
                        minimum_depth_m=self._minimum_depth_m,
                        maximum_depth_m=self._maximum_depth_m,
                    )
                    if rgb.shape[:2] == registration.aligned_depth_m.shape:
                        aligned_depth_m = registration.aligned_depth_m
                        pose_camera = registrar.color_camera
                registration_ms = (
                    time.perf_counter() - registrar_started
                ) * 1000.0
                if aligned_depth_m is not None and pose_camera is not None:
                    pose_started = time.perf_counter()
                    pose_cpu_started = time.thread_time()
                    pose_result = self._algorithm.pose_estimator.estimate(
                        detections,
                        aligned_depth_m,
                        pose_camera,
                        self._support_plane,
                        frame_key=f'{self._view}:{sequence}',
                        image_bgr=rgb,
                    )
                    inference_pose_ms = (
                        time.perf_counter() - pose_started
                    ) * 1000.0
                    inference_pose_cpu_ms = (
                        time.thread_time() - pose_cpu_started
                    ) * 1000.0

            observation_build_started = time.perf_counter()
            observation_array = to_observation_array_from_detections(
                detections=detections,
                header=pending.rgb.header,
                sequence=sequence,
                view=self._view,
                aligned_depth_m=aligned_depth_m,
            )
            observation_build_ms = (
                time.perf_counter() - observation_build_started
            ) * 1000.0
            mask_message_build_started = time.perf_counter()
            mask_messages = (
                class_mask_messages(
                    detections,
                    self._class_mask_names,
                    rgb.shape[:2],
                    pending.rgb.header,
                )
                if self._publish_class_masks
                else ()
            )
            mask_message_build_ms = (
                time.perf_counter() - mask_message_build_started
            ) * 1000.0
            selector_labels = spatial_tool_child_frames(
                self._workspace_zone, list(observation_array.instances)
            )
            selector_u_by_instance_id = {
                int(item.frame_local_instance_id): selector_horizontal_u_px(item)
                for item in observation_array.instances
            }

            valid_count = 0
            pose_array = None
            pose_message_build_ms = 0.0
            if pose_result is not None:
                pose_message_build_started = time.perf_counter()
                pose_array = to_pose_array_from_result(
                    result=pose_result,
                    header=pending.rgb.header,
                    sequence=sequence,
                    view=self._view,
                    support_plane=self._support_plane,
                    additional_status_flags=tuple(self._additional_status_flags),
                    degrade_for_additional_flags=True,
                )
                pose_message_build_ms = (
                    time.perf_counter() - pose_message_build_started
                ) * 1000.0
                valid_count = sum(
                    item.validity == ToolPose.VALIDITY_VALID
                    for item in pose_array.tools
                )

            self._queue_overlays(
                PendingOverlayFrame(
                    rgb=rgb,
                    detections=detections,
                    observation_array=observation_array,
                    selector_labels=tuple(selector_labels),
                    header=pending.rgb.header,
                    pose_result=pose_result,
                    pose_camera=pose_camera,
                )
            )

            total_ms = (time.perf_counter() - started) * 1000.0
            process_thread_cpu_ms = (
                time.thread_time() - process_cpu_started
            ) * 1000.0
            diagnostics = {
                'schema': 'pnu.native_depth_tool_pose_diagnostics.v1',
                'view': self._view,
                'sequence': sequence,
                'source_stamp_sec': int(pending.rgb.header.stamp.sec),
                'source_stamp_nanosec': int(pending.rgb.header.stamp.nanosec),
                'frame_id': pending.rgb.header.frame_id,
                'rgb_depth_delta_ns': pending.rgb_depth_delta_ns,
                'depth_sampled': aligned_depth_m is not None,
                'depth_alignment_mode': self._depth_alignment_mode,
                'aligned_depth_ready': self._aligned_depth_ready,
                'depth_to_color_extrinsics_ready': extrinsics_ready,
                'depth_to_color_extrinsics_revision': (
                    pending.extrinsics_revision
                ),
                'decode_latency_ms': decode_ms,
                'decode_thread_cpu_ms': decode_cpu_ms,
                'detect_latency_ms': detect_ms,
                'detect_thread_cpu_ms': detect_cpu_ms,
                'detector_backend': self._detector.runtime_backend,
                'detector_runtime': dict(
                    self._detector.last_runtime_diagnostics
                ),
                'depth_registration_latency_ms': registration_ms,
                'depth_registration_backend': (
                    self._registrar.backend_name
                    if self._registrar is not None
                    else (
                        'color_aligned'
                        if self._depth_aligned_to_color
                        else 'uninitialized'
                    )
                ),
                'depth_registration_gpu_ms': (
                    self._registrar.last_gpu_ms
                    if self._registrar is not None
                    else 0.0
                ),
                'inference_pose_latency_ms': inference_pose_ms,
                'inference_pose_thread_cpu_ms': inference_pose_cpu_ms,
                'pose_estimator_runtime': dict(
                    self._algorithm.pose_estimator.last_runtime_diagnostics
                    if pose_result is not None
                    else {}
                ),
                'observation_message_build_ms': observation_build_ms,
                'class_mask_message_build_ms': mask_message_build_ms,
                'pose_message_build_ms': pose_message_build_ms,
                'process_thread_cpu_ms': process_thread_cpu_ms,
                'total_latency_ms': total_ms,
                'overlay_execution': 'async_latest_only',
                'opencv_num_threads': cv2.getNumThreads(),
                'instance_count': len(detections.instances),
                'valid_pose_count': valid_count,
                'class_mask_topics': list(self._class_mask_topics.values()),
                'class_mask_frames_published': (
                    self._class_mask_frames_published
                ),
                'confidence_threshold': self._confidence_threshold,
                'inference_confidence_threshold': (
                    self._inference_confidence_threshold
                ),
                'adson_forceps_confidence_threshold': (
                    self._adson_forceps_confidence_threshold
                ),
                'bovie_confidence_threshold': (
                    self._bovie_confidence_threshold
                ),
                'enable_class_agnostic_nms': self._enable_class_agnostic_nms,
                'class_agnostic_nms_iou': self._class_agnostic_nms_iou,
                'workspace_roi_enabled': self._workspace_roi_enabled,
                'workspace_roi_name': self._workspace_zone,
                'workspace_roi_profile': self._workspace_roi_profile,
                'temporal_class_smoothing_enabled': (
                    self._temporal_class_smoothing_enabled
                ),
                'detection_postprocessing': dict(
                    self._detection_postprocessor.last_diagnostics
                ),
                'pose_mode': 'PLANAR_4DOF_WITH_NORMAL_PRIOR',
                'tf_topic': '/tf',
                'tf_selector_semantics': (
                    'workspace_zone_class_current_left_to_right'
                ),
                'tf_workspace_zone': self._workspace_zone,
                'tf_selector_order_axis': 'camera_image_u_ascending',
                'tf_orientation_provenance': CONSTRAINED_SE3_PROVENANCE,
                'tf_position_stabilization_enabled': (
                    self._tf_position_stabilization_enabled
                ),
                'tf_axis_stabilization_enabled': (
                    self._tf_axis_stabilization_enabled
                ),
                'tf_axis_flip_held_total': (
                    self._tf_tracker.axis_filter_flip_held_total
                ),
                'tf_axis_flip_confirmed_total': (
                    self._tf_tracker.axis_filter_flip_confirmed_total
                ),
                'error_code': '',
            }
            source_stamp_ns = (
                int(pending.rgb.header.stamp.sec) * 1_000_000_000
                + int(pending.rgb.header.stamp.nanosec)
            )
            self._output_slot.put(PendingOutputBundle(
                pose_array=pose_array,
                observation_array=observation_array,
                class_mask_messages=mask_messages,
                selector_u_by_instance_id=selector_u_by_instance_id,
                diagnostics=diagnostics,
                process_started=started,
                queued_monotonic=time.perf_counter(),
                source_stamp_ns=source_stamp_ns,
                instance_count=len(detections.instances),
                valid_pose_count=valid_count,
            ))
        except Exception as error:
            self._set_error('FRAME_PROCESSING_FAILED', str(error))
            self.get_logger().error(f'tool recognition frame failed: {error}')
            diagnostics = {
                'schema': 'pnu.native_depth_tool_pose_diagnostics.v1',
                'source_stamp_sec': int(pending.rgb.header.stamp.sec),
                'source_stamp_nanosec': int(pending.rgb.header.stamp.nanosec),
                'rgb_depth_delta_ns': pending.rgb_depth_delta_ns,
                'error_code': 'FRAME_PROCESSING_FAILED',
                'error_message': str(error),
            }
            self._diagnostics_publisher.publish(
                String(data=json.dumps(diagnostics, separators=(',', ':')))
            )

    def _publish_constrained_tool_tf(
        self,
        pose_array: ToolPoseArray,
        *,
        selector_u_by_instance_id: dict[int, float],
    ) -> dict[str, Any]:
        """Emit current measured constrained tool transforms on dynamic ``/tf``.

        The input pose contract remains planar 4-DoF: position and yaw are
        measured while roll/pitch are completed by the configured support-plane
        normal. Quality flags and source age remain available in
        ``ToolPoseArray``/diagnostics but do not suppress a finite measured pose
        representation. Only a missing source stamp or structurally invalid
        numeric pose is rejected before broadcast.
        """
        report: dict[str, Any] = {
            'enabled': self._publish_tool_tf,
            'published_count': 0,
            'skipped_count': 0,
            'source_age_sec': None,
            'skip_reason': '',
            'active_track_count': 0,
            'track_created_total': 0,
            'track_expired_total': 0,
            'track_rejected_total': 0,
            'track_reset_total': 0,
            'position_filter_active_count': 0,
            'position_filter_held_total': 0,
            'position_filter_smoothed_total': 0,
            'position_filter_outlier_held_total': 0,
            'position_filter_relocation_total': 0,
            'position_filter_association_reset_total': 0,
        }
        if not self._publish_tool_tf or self._tf_broadcaster is None:
            report['skip_reason'] = 'TF_DISABLED'
            return report

        source_stamp_ns = source_stamp_nanoseconds(pose_array.header)
        source_age_sec = source_age_seconds(
            pose_array.header, self.get_clock().now().nanoseconds
        )
        report['source_age_sec'] = source_age_sec
        with self._condition:
            self._tf_last_input_source_stamp_ns = source_stamp_ns
            self._tf_last_input_source_age_sec = source_age_sec

        if source_age_sec is None:
            return self._record_tf_skip(report, 'SOURCE_STAMP_MISSING')

        with self._condition:
            decisions = self._tf_tracker.assign(
                pose_array.header,
                list(pose_array.tools),
                self._workspace_zone,
                horizontal_u_by_instance_id=selector_u_by_instance_id,
            )
            report['active_track_count'] = self._tf_tracker.active_track_count
            report['track_created_total'] = self._tf_tracker.created_total
            report['track_expired_total'] = self._tf_tracker.expired_total
            report['track_rejected_total'] = self._tf_tracker.rejected_total
            report['track_reset_total'] = self._tf_tracker.reset_total
            report['position_filter_active_count'] = (
                self._tf_tracker.position_filter_active_count
            )
            report['position_filter_held_total'] = (
                self._tf_tracker.position_filter_held_total
            )
            report['position_filter_smoothed_total'] = (
                self._tf_tracker.position_filter_smoothed_total
            )
            report['position_filter_outlier_held_total'] = (
                self._tf_tracker.position_filter_outlier_held_total
            )
            report['position_filter_relocation_total'] = (
                self._tf_tracker.position_filter_relocation_total
            )
            report['position_filter_association_reset_total'] = (
                self._tf_tracker.position_filter_association_reset_total
            )
        transforms = [
            decision.transform
            for decision in decisions
            if decision.transform is not None
        ]
        skip_reasons = [
            decision.reason
            for decision in decisions
            if decision.transform is None
        ]
        report['skipped_count'] = len(skip_reasons)
        if skip_reasons:
            report['skip_reason'] = skip_reasons[0]

        if not transforms:
            return self._record_tf_skip(
                report, report['skip_reason'] or 'NO_VALID_TOOL_TRANSFORMS'
            )

        try:
            self._tf_broadcaster.sendTransform(transforms)
        except Exception as error:
            message = str(error)
            with self._condition:
                self._tf_last_error = message
                self._tf_skipped_total += len(transforms)
                self._tf_last_skip_reason = 'TF_BROADCAST_FAILED'
            self.get_logger().error(f'constrained tool TF publish failed: {message}')
            report['skipped_count'] += len(transforms)
            report['skip_reason'] = 'TF_BROADCAST_FAILED'
            return report

        child_frames = tuple(
            transform.child_frame_id for transform in transforms
        )
        with self._condition:
            self._tf_broadcast_total += len(transforms)
            self._tf_last_output_source_stamp_ns = source_stamp_ns
            self._tf_last_parent_frame = transforms[0].header.frame_id
            self._tf_last_child_frames = child_frames
            self._tf_last_error = ''
            self._tf_last_skip_reason = report['skip_reason']
            self._tf_skipped_total += report['skipped_count']
        report['published_count'] = len(transforms)
        return report

    def _record_tf_skip(
        self, report: dict[str, Any], reason: str
    ) -> dict[str, Any]:
        """Record an intentional no-broadcast decision without faulting RGB-D."""
        report['skip_reason'] = reason
        report['skipped_count'] = max(1, int(report['skipped_count']))
        with self._condition:
            self._tf_skipped_total += int(report['skipped_count'])
            self._tf_last_skip_reason = reason
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_tf_log_monotonic >= 1.0:
            self.get_logger().warning(
                f'constrained tool TF skipped: {reason}'
            )
            self._last_tf_log_monotonic = now_monotonic
        return report

    def _publish_detection_overlay(
        self,
        rgb: np.ndarray,
        detections: Any,
        observation_array: ToolObservation2DArray,
        selector_labels: list[str],
        header: Any,
    ) -> None:
        """Publish masks with current workspace/class/spatial selectors."""
        output = rgb.copy()
        if self._detection_postprocessor is not None:
            polygon = self._detection_postprocessor.roi_polygon_pixels(
                output.shape[1], output.shape[0]
            )
            if polygon is not None:
                cv2.polylines(
                    output,
                    [polygon],
                    isClosed=True,
                    color=(30, 230, 30),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                anchor = tuple(int(value) for value in polygon[0])
                cv2.putText(
                    output,
                    f'ROI:{self._workspace_roi_profile}',
                    (anchor[0], max(18, anchor[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (30, 230, 30),
                    2,
                    cv2.LINE_AA,
                )
        for item, observation, selector_label in zip(
            detections.instances,
            observation_array.instances,
            selector_labels,
            strict=False,
        ):
            color = tool_overlay_color_bgr(item.class_name)
            layer = output.copy()
            layer[item.mask] = color
            output = cv2.addWeighted(output, 0.72, layer, 0.28, 0.0)
            x0, y0, x1, y1 = (
                int(round(value)) for value in item.bbox_xyxy_px
            )
            cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
            label = f'{selector_label} {item.class_confidence:.2f}'
            cv2.putText(
                output,
                label,
                (x0, max(18, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                2,
            )
            if observation.observation_point_valid:
                u = int(round(observation.observation_point_uv_px[0]))
                v = int(round(observation.observation_point_uv_px[1]))
                cv2.circle(output, (u, v), 6, color, 2, cv2.LINE_AA)
        success, encoded = cv2.imencode(
            '.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success:
            raise RuntimeError('OpenCV could not encode pose overlay')
        message = CompressedImage()
        message.header = header
        message.format = 'bgr8; jpeg compressed bgr8'
        message.data = encoded.tobytes()
        self._overlay_publisher.publish(message)

    def _publish_pose_overlay(
        self, rgb: np.ndarray, result: Any, camera: Any, header: Any
    ) -> None:
        """Publish the v1.6 quaternion-axis debug overlay."""
        output = self._algorithm_symbols['draw_pose_axes_bgr'](
            rgb,
            result,
            camera,
            axis_length_m=self._pose_axis_length_m,
        )
        success, encoded = cv2.imencode(
            '.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success:
            raise RuntimeError('OpenCV could not encode pose-axis overlay')
        message = CompressedImage()
        message.header = header
        message.format = 'bgr8; jpeg compressed bgr8'
        message.data = encoded.tobytes()
        self._pose_overlay_publisher.publish(message)

    def _publish_overlay(
        self, rgb: np.ndarray, result: Any, header: Any
    ) -> None:
        """Publish a human-readable mask and observation-point overlay."""
        output = rgb.copy()
        for item in result.instances:
            color = tool_overlay_color_bgr(item.class_name)
            layer = output.copy()
            layer[item.mask] = color
            output = cv2.addWeighted(output, 0.72, layer, 0.28, 0.0)
            x0, y0, x1, y1 = (
                int(round(value)) for value in item.bbox_xyxy_px
            )
            cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
            label = (
                f'{item.class_name} {item.class_confidence:.2f} '
                f'{item.validity}'
            )
            cv2.putText(
                output,
                label,
                (x0, max(18, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                2,
            )
            if item.observation_point_uv_px is not None:
                u, v = (
                    int(round(value)) for value in item.observation_point_uv_px
                )
                cv2.circle(output, (u, v), 6, color, 2, cv2.LINE_AA)
        success, encoded = cv2.imencode(
            '.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success:
            raise RuntimeError('OpenCV could not encode pose overlay')
        message = CompressedImage()
        message.header = header
        message.format = 'bgr8; jpeg compressed bgr8'
        message.data = encoded.tobytes()
        self._overlay_publisher.publish(message)

    def _set_error(self, code: str, message: str) -> None:
        """Store the most recent runtime error."""
        with self._condition:
            self._last_error_code = code
            self._last_error_message = message

    def _publish_health(self) -> None:
        """Publish readiness without claiming unconstrained full-6D support."""
        now = time.monotonic()
        ros_now_ns = self.get_clock().now().nanoseconds
        with self._condition:
            input_fresh = (
                self._last_pair_monotonic is not None
                and now - self._last_pair_monotonic
                <= self._input_freshness_sec
            )
            active_extrinsics = self._active_extrinsics_locked()
            extrinsics_ready = active_extrinsics is not None
            if not self._require_depth:
                depth_ok = True
            elif self._depth_aligned_to_color:
                depth_ok = bool(self._aligned_depth_ready)
            else:
                depth_ok = bool(self._registrar_ready) and extrinsics_ready
            tf_input_age_sec = (
                (ros_now_ns - self._tf_last_input_source_stamp_ns)
                / 1_000_000_000.0
                if self._tf_last_input_source_stamp_ns is not None
                else None
            )
            tf_output_age_sec = (
                (ros_now_ns - self._tf_last_output_source_stamp_ns)
                / 1_000_000_000.0
                if self._tf_last_output_source_stamp_ns is not None
                else None
            )
            tf_input_fresh = bool(
                tf_input_age_sec is not None
                and -self._tf_max_future_sec
                <= tf_input_age_sec
                <= self._tf_stale_after_sec
            )
            tf_output_active = bool(
                tf_output_age_sec is not None
                and -self._tf_max_future_sec
                <= tf_output_age_sec
                <= self._tf_stale_after_sec
            )
            payload = {
                'schema': 'pnu.native_depth_tool_pose_health.v1',
                'ready': bool(
                    self._model_ready
                    and depth_ok
                    and input_fresh
                    and self._last_success_monotonic is not None
                    and (not self._require_depth or not self._additional_status_flags)
                    and not self._last_error_code
                ),
                'model_ready': self._model_ready,
                'detector_backend': (
                    self._detector.runtime_backend
                    if self._detector is not None
                    else 'unloaded'
                ),
                'detector_runtime': (
                    dict(self._detector.last_runtime_diagnostics)
                    if self._detector is not None
                    else {}
                ),
                'color_camera_info_ready': self._color_info is not None,
                'depth_camera_info_ready': self._depth_info is not None,
                'depth_alignment_mode': self._depth_alignment_mode,
                'depth_aligned_to_color': self._depth_aligned_to_color,
                'aligned_depth_ready': self._aligned_depth_ready,
                'depth_geometry_ready': depth_ok,
                'depth_registrar_ready': self._registrar_ready,
                'depth_registration_backend_requested': (
                    self._depth_registration_backend
                ),
                'depth_registration_backend_active': (
                    self._registrar.backend_name
                    if self._registrar is not None
                    else (
                        'color_aligned'
                        if self._depth_aligned_to_color
                        else 'uninitialized'
                    )
                ),
                'depth_registration_fallback_active': bool(
                    self._registrar is not None
                    and self._registrar.fallback_active
                ),
                'depth_to_color_extrinsics_topic': self._extrinsics_topic,
                'depth_to_color_extrinsics_required': (
                    self._require_extrinsics_topic
                ),
                'depth_to_color_extrinsics_ready': extrinsics_ready,
                'depth_to_color_extrinsics_baseline_m': (
                    active_extrinsics.baseline_m
                    if active_extrinsics is not None
                    else None
                ),
                'depth_to_color_extrinsics_received': self._received_extrinsics,
                'depth_to_color_extrinsics_rejected': self._rejected_extrinsics,
                'depth_to_color_extrinsics_revision': self._extrinsics_revision,
                'last_depth_to_color_extrinsics_error': (
                    self._last_extrinsics_error
                ),
                'require_depth': self._require_depth,
                'input_fresh': input_fresh,
                'processing_enabled': self._processing_enabled,
                'depth_scale_verified': self._depth_scale_verified,
                'metric_calibration_verified': (
                    not self._additional_status_flags
                ),
                'workspace_roi_enabled': self._workspace_roi_enabled,
                'workspace_roi_profile': self._workspace_roi_profile,
                'class_masks_enabled': self._publish_class_masks,
                'class_mask_topics': list(self._class_mask_topics.values()),
                'class_mask_frames_published': (
                    self._class_mask_frames_published
                ),
                'output_execution': 'async_latest_only_bundle',
                'output_bundle_pending': self._output_slot.has_pending,
                'output_bundle_published_total': (
                    self._output_published_bundles
                ),
                'output_bundle_overwritten_total': (
                    self._output_slot.overwritten_total
                ),
                'last_output_bundle_queue_wait_ms': (
                    self._last_output_bundle_queue_wait_ms
                ),
                'last_output_bundle_publish_ms': (
                    self._last_output_bundle_publish_ms
                ),
                'last_output_bundle_latency_ms': (
                    self._last_output_bundle_latency_ms
                ),
                'last_output_bundle_source_age_ms': (
                    self._last_output_bundle_source_age_ms
                ),
                'last_output_error': self._last_output_error,
                'available_pose_mode': 'PLANAR_4DOF_WITH_NORMAL_PRIOR',
                'full_6d_available': False,
                'tf_topic': '/tf',
                'tf_enabled': self._publish_tool_tf,
                'tf_selector_semantics': (
                    'workspace_zone_class_current_left_to_right'
                ),
                'tf_workspace_zone': self._workspace_zone,
                'tf_selector_order_axis': 'camera_image_u_ascending',
                'tf_orientation_provenance': CONSTRAINED_SE3_PROVENANCE,
                'tf_input_fresh': tf_input_fresh,
                'tf_output_active': tf_output_active,
                'tf_stale_after_sec': self._tf_stale_after_sec,
                'tf_max_future_sec': self._tf_max_future_sec,
                'tf_last_input_age_sec': tf_input_age_sec,
                'tf_last_output_age_sec': tf_output_age_sec,
                'tf_last_parent_frame': self._tf_last_parent_frame,
                'tf_last_child_frames': list(self._tf_last_child_frames),
                'tf_broadcast_total': self._tf_broadcast_total,
                'tf_skipped_total': self._tf_skipped_total,
                'tf_last_skip_reason': self._tf_last_skip_reason,
                'tf_last_error': self._tf_last_error,
                'tf_track_max_displacement_m': (
                    self._tf_track_max_displacement_m
                ),
                'tf_track_ttl_sec': self._tf_track_ttl_sec,
                'tf_track_geometry_gate_applied': False,
                'tf_track_max_active_per_class': (
                    self._tf_track_max_active_per_class
                ),
                'tf_active_track_count': self._tf_tracker.active_track_count,
                'tf_track_created_total': self._tf_tracker.created_total,
                'tf_track_expired_total': self._tf_tracker.expired_total,
                'tf_track_rejected_total': self._tf_tracker.rejected_total,
                'tf_track_reset_total': self._tf_tracker.reset_total,
                'received_rgb': self._received_rgb,
                'received_depth': self._received_depth,
                'paired_frames': self._paired_frames,
                'processed_frames': self._processed_frames,
                'dropped_pending_frames': self._dropped_pending_frames,
                'overlay_processed_frames': self._overlay_processed_frames,
                'overlay_dropped_pending_frames': (
                    self._overlay_dropped_pending_frames
                ),
                'last_overlay_latency_ms': self._last_overlay_latency_ms,
                'last_overlay_error': self._last_overlay_error,
                'dropped_unmatched_frames': self._pairer.dropped_unmatched,
                'last_error_code': self._last_error_code,
                'last_error_message': self._last_error_message,
            }
        self._health_publisher.publish(
            String(data=json.dumps(payload, separators=(',', ':')))
        )

    def stop(self) -> None:
        """Stop and join the processing worker."""
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._output_slot.stop()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        if self._overlay_worker.is_alive():
            self._overlay_worker.join(timeout=5.0)
        if self._output_worker.is_alive():
            self._output_worker.join(timeout=5.0)
        if self._detector is not None:
            self._detector.close()


def main(args=None) -> None:
    """Run the native-depth pose node."""
    rclpy.init(args=args)
    node = NativeDepthPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
