"""Estimate constrained tool poses from ROS RGB, sampling depth when present."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
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
    to_pose_and_observation_arrays,
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
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from surgical_perception_msgs.msg import (
    ToolObservation2DArray,
    ToolPose,
    ToolPoseArray,
)


DEFAULT_ALGORITHM_PATH = os.environ.get(
    'PNU_SURGICAL_TOOL_ALGORITHM_PATH', ''
)
DEFAULT_CHECKPOINT = os.environ.get('PNU_RFDETR_CHECKPOINT', '')
DEFAULT_ONTOLOGY = os.environ.get('PNU_SURGICAL_TOOL_ONTOLOGY', '')


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


def tool_overlay_color_bgr(class_name: str) -> tuple[int, int, int]:
    """Return the canonical visual color for a recognized surgical tool."""
    return TOOL_OVERLAY_COLORS_BGR.get(class_name, (235, 235, 235))


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
        self._stopping = False
        self._model_ready = False
        self._registrar_ready = False
        self._last_success_monotonic: float | None = None
        self._last_pair_monotonic: float | None = None
        self._last_error_code = ''
        self._last_error_message = ''
        self._received_rgb = 0
        self._received_depth = 0
        self._paired_frames = 0
        self._processed_frames = 0
        self._dropped_pending_frames = 0
        self._last_publish_log_monotonic = 0.0
        self._sequence = 0
        self._registrar = None
        self._registrar_key: tuple[Any, ...] | None = None
        self._algorithm = None
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
        self._worker.start()
        self._health_timer = self.create_timer(1.0, self._publish_health)
        self.get_logger().info(
            f'native-depth pose node started: rgb={self._rgb_topic}, '
            f'depth={self._depth_topic}, require_depth={self._require_depth}, '
            f'extrinsics={self._extrinsics_topic}, '
            f'workspace_zone={self._workspace_zone}, '
            f'workspace_roi_profile={self._workspace_roi_profile}, '
            f'max_delta_ns={self._maximum_stamp_delta_ns}, '
            f'publish_tool_tf={self._publish_tool_tf}'
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
        self.declare_parameter('pose_topic', f'{out}/poses')
        self.declare_parameter('observation_topic', f'{out}/observations')
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
        self.declare_parameter('input_freshness_sec', 2.0)
        self.declare_parameter('processing_enabled', True)
        self.declare_parameter('processing_gate_topic', '')
        self.declare_parameter('require_depth', True)

        self.declare_parameter(
            'algorithm_python_path', DEFAULT_ALGORITHM_PATH
        )
        self.declare_parameter('checkpoint', DEFAULT_CHECKPOINT)
        self.declare_parameter('ontology', DEFAULT_ONTOLOGY)
        self.declare_parameter('model_size', 'small')
        self.declare_parameter('checkpoint_color_order', 'BGR')
        self.declare_parameter('model_version', '')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('enable_class_agnostic_nms', True)
        self.declare_parameter('class_agnostic_nms_iou', 0.8)
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
        self.declare_parameter('pose_axis_length_m', 0.05)
        self.declare_parameter('optimize_for_inference', True)
        self.declare_parameter('jit_compile', True)
        self.declare_parameter('fp16', True)
        self.declare_parameter('jpeg_quality', 90)

        self.declare_parameter('depth_scale_m_per_unit', 0.0)
        self.declare_parameter('depth_scale_verified', False)
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
        if self._require_extrinsics_topic and not self._extrinsics_topic:
            raise ValueError(
                'extrinsics_topic is required for native-depth 3D pose'
            )
        self._pose_topic = str(value('pose_topic'))
        self._observation_topic = str(value('observation_topic'))
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
        self._confidence_threshold = float(value('confidence_threshold'))
        self._enable_class_agnostic_nms = bool(
            value('enable_class_agnostic_nms')
        )
        self._class_agnostic_nms_iou = float(
            value('class_agnostic_nms_iou')
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
        self._tf_tracker = ToolSpatialTfSelector(
            max_tools_per_class=self._tf_track_max_active_per_class,
            reset_stamp_jump_sec=self._tf_track_reset_stamp_jump_sec,
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
            PlanarPoseEstimator,
            RigidTransform,
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
            )
        )
        detector.load()
        self._detection_postprocessor = DetectionPostprocessor(
            DetectionPostprocessorConfig(
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
            PlanarPoseEstimator(),
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
        self._registrar = registrar_type(depth_camera, color_camera, transform)
        self._registrar_key = key
        with self._condition:
            self._registrar_ready = True
        return self._registrar

    def _process_frame(self, pending: PendingPoseFrame) -> None:
        """Detect on RGB always; sample depth and estimate pose only if present."""
        started = time.perf_counter()
        try:
            rgb = decode_rgb(pending.rgb)
            decode_ms = (time.perf_counter() - started) * 1000.0
            with self._condition:
                self._sequence += 1
                sequence = self._sequence
            detect_started = time.perf_counter()
            detections = self._algorithm.detect(
                rgb, 'BGR', self._confidence_threshold
            )
            detect_ms = (time.perf_counter() - detect_started) * 1000.0

            aligned_depth_m = None
            registration_ms = 0.0
            pose_result = None
            inference_pose_ms = 0.0
            with self._condition:
                extrinsics_ready = bool(
                    pending.depth_to_color_extrinsics is not None
                    and pending.extrinsics_revision
                    == self._extrinsics_revision
                    and self._active_extrinsics_locked()
                    is pending.depth_to_color_extrinsics
                )
            if (
                pending.depth is not None
                and pending.color_info is not None
                and pending.depth_info is not None
                and extrinsics_ready
            ):
                native_depth = self._algorithm_symbols['decode_depth'](
                    pending.depth.data, pending.depth.format
                )
                registrar_started = time.perf_counter()
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
                registration_ms = (
                    time.perf_counter() - registrar_started
                ) * 1000.0
                if rgb.shape[:2] == registration.aligned_depth_m.shape:
                    aligned_depth_m = registration.aligned_depth_m
                    pose_started = time.perf_counter()
                    pose_result = self._algorithm.pose_estimator.estimate(
                        detections,
                        aligned_depth_m,
                        registrar.color_camera,
                        self._support_plane,
                        frame_key=f'{self._view}:{sequence}',
                    )
                    inference_pose_ms = (
                        time.perf_counter() - pose_started
                    ) * 1000.0

            observation_array = to_observation_array_from_detections(
                detections=detections,
                header=pending.rgb.header,
                sequence=sequence,
                view=self._view,
                aligned_depth_m=aligned_depth_m,
            )
            self._observation_publisher.publish(observation_array)
            selector_labels = spatial_tool_child_frames(
                self._workspace_zone, list(observation_array.instances)
            )
            selector_u_by_instance_id = {
                int(item.frame_local_instance_id): selector_horizontal_u_px(item)
                for item in observation_array.instances
            }

            valid_count = 0
            if pose_result is not None:
                pose_array, _unused = to_pose_and_observation_arrays(
                    result=pose_result,
                    header=pending.rgb.header,
                    sequence=sequence,
                    view=self._view,
                    support_plane=self._support_plane,
                    additional_status_flags=tuple(self._additional_status_flags),
                    degrade_for_additional_flags=True,
                )
                self._pose_publisher.publish(pose_array)
                tf_report = self._publish_constrained_tool_tf(
                    pose_array,
                    selector_u_by_instance_id=selector_u_by_instance_id,
                )
                self._publish_pose_overlay(
                    rgb, pose_result, registrar.color_camera, pending.rgb.header
                )
                valid_count = sum(
                    item.validity == ToolPose.VALIDITY_VALID
                    for item in pose_array.tools
                )
            else:
                tf_report = {
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
                }

            self._publish_detection_overlay(
                rgb,
                detections,
                observation_array,
                selector_labels,
                pending.rgb.header,
            )

            total_ms = (time.perf_counter() - started) * 1000.0
            diagnostics = {
                'schema': 'pnu.native_depth_tool_pose_diagnostics.v1',
                'view': self._view,
                'sequence': sequence,
                'source_stamp_sec': int(pending.rgb.header.stamp.sec),
                'source_stamp_nanosec': int(pending.rgb.header.stamp.nanosec),
                'frame_id': pending.rgb.header.frame_id,
                'rgb_depth_delta_ns': pending.rgb_depth_delta_ns,
                'depth_sampled': aligned_depth_m is not None,
                'depth_to_color_extrinsics_ready': extrinsics_ready,
                'depth_to_color_extrinsics_revision': (
                    pending.extrinsics_revision
                ),
                'decode_latency_ms': decode_ms,
                'detect_latency_ms': detect_ms,
                'depth_registration_latency_ms': registration_ms,
                'inference_pose_latency_ms': inference_pose_ms,
                'total_latency_ms': total_ms,
                'instance_count': len(detections.instances),
                'valid_pose_count': valid_count,
                'confidence_threshold': self._confidence_threshold,
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
                'tf_published_count': tf_report['published_count'],
                'tf_skipped_count': tf_report['skipped_count'],
                'tf_source_age_sec': tf_report['source_age_sec'],
                'tf_skip_reason': tf_report['skip_reason'],
                'tf_active_track_count': tf_report['active_track_count'],
                'tf_track_created_total': tf_report['track_created_total'],
                'tf_track_expired_total': tf_report['track_expired_total'],
                'tf_track_rejected_total': tf_report['track_rejected_total'],
                'tf_track_reset_total': tf_report['track_reset_total'],
                'error_code': '',
            }
            self._diagnostics_publisher.publish(
                String(data=json.dumps(diagnostics, separators=(',', ':')))
            )
            now_monotonic = time.monotonic()
            if now_monotonic - self._last_publish_log_monotonic >= 1.0:
                self.get_logger().info(
                    f'published v1.6 tools: {len(detections.instances)} '
                    f'instances; valid_poses={valid_count}; '
                    f'depth_sampled={aligned_depth_m is not None}'
                )
                self._last_publish_log_monotonic = now_monotonic
            with self._condition:
                self._processed_frames += 1
                self._last_success_monotonic = time.monotonic()
                self._last_error_code = ''
                self._last_error_message = ''
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
        """Emit current valid constrained tool transforms on dynamic ``/tf``.

        The input pose contract remains planar 4-DoF: position and yaw are
        measured while roll/pitch are completed by the configured support-plane
        normal.  A TF consumer receives the complete SE(3) representation but
        never an invalid, degraded, or stale transform masquerading as current.
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
        if source_age_sec > self._tf_stale_after_sec:
            return self._record_tf_skip(report, 'SOURCE_STALE')
        if source_age_sec < -self._tf_max_future_sec:
            return self._record_tf_skip(report, 'SOURCE_TIMESTAMP_IN_FUTURE')

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
            depth_ok = (
                bool(self._registrar_ready) and extrinsics_ready
                if self._require_depth
                else True
            )
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
                'color_camera_info_ready': self._color_info is not None,
                'depth_camera_info_ready': self._depth_info is not None,
                'depth_registrar_ready': self._registrar_ready,
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
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)


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
