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

from pnu_surgical_perception.native_depth_sync import (
    ApproximateRgbDepthPairer,
    RgbDepthPair,
)
from pnu_surgical_perception.pose_message_mapping import (
    to_observation_array_from_detections,
    to_pose_and_observation_arrays,
)

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)

from sensor_msgs.msg import CameraInfo, CompressedImage

from std_msgs.msg import Bool, String

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


def reliable_qos(depth: int = 5) -> QoSProfile:
    """Return reliable volatile QoS for compact pose results."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
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
    rgb_depth_delta_ns: int | None
    received_monotonic: float


class NativeDepthPoseNode(Node):
    """Latest-frame ROS adapter for native depth registration and tool pose."""

    def __init__(self) -> None:
        """Initialize configured subscriptions, publishers, and worker."""
        super().__init__('native_depth_tool_pose')
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

        self._condition = threading.Condition()
        self._pairer = ApproximateRgbDepthPairer(
            self._maximum_stamp_delta_ns, self._sync_queue_size
        )
        self._color_info: CameraInfo | None = None
        self._depth_info: CameraInfo | None = None
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
        self._support_plane = None
        self._algorithm_symbols: dict[str, Any] = {}

        self._subscriptions = [
            self.create_subscription(
                CameraInfo,
                self._color_info_topic,
                self._receive_color_info,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                CameraInfo,
                self._depth_info_topic,
                self._receive_depth_info,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                CompressedImage,
                self._rgb_topic,
                self._receive_rgb,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                CompressedImage,
                self._depth_topic,
                self._receive_depth,
                qos_profile_sensor_data,
            ),
        ]
        if self._processing_gate_topic:
            self._subscriptions.append(
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
            f'max_delta_ns={self._maximum_stamp_delta_ns}'
        )

    def _declare_parameters(self) -> None:
        """Declare transport, model, and geometry parameters."""
        prefix = '/synced/cam_4'
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
            'pose_topic', '/surgery/perception/cam4/tool_poses'
        )
        self.declare_parameter(
            'observation_topic', '/surgery/perception/cam4/observations'
        )
        self.declare_parameter(
            'overlay_topic',
            '/surgery/images/cam4/detection_overlay/compressed',
        )
        self.declare_parameter(
            'pose_overlay_topic',
            '/surgery/images/cam4/pose_overlay/compressed',
        )
        self.declare_parameter(
            'diagnostics_topic', '/surgery/perception/rfdetr/diagnostics/json'
        )
        self.declare_parameter(
            'health_topic', '/surgery/perception/rfdetr/health'
        )
        self.declare_parameter('view', 'cam4')
        self.declare_parameter('maximum_stamp_delta_ns', 1_000_000)
        self.declare_parameter('sync_queue_size', 8)
        self.declare_parameter('latest_frame_only', True)
        self.declare_parameter('input_freshness_sec', 2.0)
        self.declare_parameter('processing_enabled', True)
        self.declare_parameter('processing_gate_topic', '')
        self.declare_parameter('require_depth', False)

        self.declare_parameter(
            'algorithm_python_path', DEFAULT_ALGORITHM_PATH
        )
        self.declare_parameter('checkpoint', DEFAULT_CHECKPOINT)
        self.declare_parameter('ontology', DEFAULT_ONTOLOGY)
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('enable_class_agnostic_nms', True)
        self.declare_parameter('class_agnostic_nms_iou', 0.8)
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
            'depth_to_color_translation_m', [float('nan')] * 3
        )
        self.declare_parameter('expected_color_frame', '')
        self.declare_parameter('expected_depth_frame', '')
        self.declare_parameter('calibration_version', '')

        self.declare_parameter('support_plane_normal', [float('nan')] * 3)
        self.declare_parameter('support_plane_offset_m', float('nan'))
        self.declare_parameter('support_plane_config_version', '')
        self.declare_parameter('support_plane_inlier_ratio', 0.0)
        self.declare_parameter('support_plane_residual_p95_m', 0.0)

    def _read_parameters(self) -> None:
        """Read and fail closed on incomplete metric geometry configuration."""
        def value(name: str) -> Any:
            return self.get_parameter(name).value

        self._rgb_topic = str(value('rgb_topic'))
        self._color_info_topic = str(value('color_camera_info_topic'))
        self._depth_topic = str(value('depth_topic'))
        self._depth_info_topic = str(value('depth_camera_info_topic'))
        self._pose_topic = str(value('pose_topic'))
        self._observation_topic = str(value('observation_topic'))
        self._overlay_topic = str(value('overlay_topic'))
        self._pose_overlay_topic = str(value('pose_overlay_topic'))
        self._diagnostics_topic = str(value('diagnostics_topic'))
        self._health_topic = str(value('health_topic'))
        self._view = str(value('view'))
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
        self._confidence_threshold = float(value('confidence_threshold'))
        self._enable_class_agnostic_nms = bool(
            value('enable_class_agnostic_nms')
        )
        self._class_agnostic_nms_iou = float(
            value('class_agnostic_nms_iou')
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
        self._rotation = finite_vector(
            'depth_to_color_rotation', value('depth_to_color_rotation'), 9
        ).reshape(3, 3)
        self._translation = finite_vector(
            'depth_to_color_translation_m',
            value('depth_to_color_translation_m'),
            3,
        )
        self._expected_color_frame = str(value('expected_color_frame'))
        self._expected_depth_frame = str(value('expected_depth_frame'))
        self._calibration_version = str(value('calibration_version'))
        if not self._calibration_version:
            raise ValueError('calibration_version is required')

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

    def _receive_processing_gate(self, message: Bool) -> None:
        """Enable or pause inference while keeping the model preloaded."""
        enabled = bool(message.data)
        with self._condition:
            changed = self._processing_enabled != enabled
            self._processing_enabled = enabled
            if not enabled:
                self._pending = None
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
            DetectorConfig,
            PlanarPoseEstimator,
            RigidTransform,
            SupportPlane,
            SurgicalToolAlgorithm,
            SurgicalToolDetector,
            draw_pose_axes_bgr,
        )

        detector = SurgicalToolDetector(
            DetectorConfig(
                checkpoint_path=self._checkpoint,
                ontology_path=self._ontology,
                confidence_threshold=self._confidence_threshold,
                enable_class_agnostic_nms=self._enable_class_agnostic_nms,
                class_agnostic_nms_iou=self._class_agnostic_nms_iou,
                optimize=self._optimize,
                jit_compile=self._jit_compile,
                fp16=self._fp16,
                checkpoint_color_order='BGR',
            )
        )
        detector.load()
        self._algorithm = SurgicalToolAlgorithm(
            detector, PlanarPoseEstimator()
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
        self, color_info: CameraInfo, depth_info: CameraInfo
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
            *tuple(self._rotation.ravel()),
            *tuple(self._translation),
        )
        if key == self._registrar_key and self._registrar is not None:
            return self._registrar

        color_camera = self._camera_calibration(color_info, 'color')
        depth_camera = self._camera_calibration(depth_info, 'depth')
        transform_type = self._algorithm_symbols['RigidTransform']
        transform = transform_type(
            rotation=self._rotation,
            translation_m=self._translation,
            source_frame=depth_frame,
            target_frame=color_frame,
            calibration_version=f'{self._calibration_version}:depth_to_color',
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
            if (
                pending.depth is not None
                and pending.color_info is not None
                and pending.depth_info is not None
            ):
                native_depth = self._algorithm_symbols['decode_depth'](
                    pending.depth.data, pending.depth.format
                )
                registrar_started = time.perf_counter()
                registrar = self._get_registrar(
                    pending.color_info, pending.depth_info
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
                self._publish_pose_overlay(
                    rgb, pose_result, registrar.color_camera, pending.rgb.header
                )
                valid_count = sum(
                    item.validity == ToolPose.VALIDITY_VALID
                    for item in pose_array.tools
                )

            self._publish_detection_overlay(
                rgb, detections, observation_array, pending.rgb.header
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
                'pose_mode': 'PLANAR_4DOF_WITH_NORMAL_PRIOR',
                'error_code': '',
            }
            self._diagnostics_publisher.publish(
                String(data=json.dumps(diagnostics, separators=(',', ':')))
            )
            now_monotonic = time.monotonic()
            if now_monotonic - self._last_publish_log_monotonic >= 1.0:
                self.get_logger().info(
                    f'published v1.4 tools: {len(detections.instances)} '
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

    def _publish_detection_overlay(
        self,
        rgb: np.ndarray,
        detections: Any,
        observation_array: ToolObservation2DArray,
        header: Any,
    ) -> None:
        """Publish mask overlay with longitudinal-axis midpoints."""
        output = rgb.copy()
        for item, observation in zip(
            detections.instances, observation_array.instances, strict=False
        ):
            color = (
                (45, 210, 245)
                if observation.observation_point_depth_valid
                else (30, 140, 255)
            )
            layer = output.copy()
            layer[item.mask] = color
            output = cv2.addWeighted(output, 0.72, layer, 0.28, 0.0)
            x0, y0, x1, y1 = (
                int(round(value)) for value in item.bbox_xyxy_px
            )
            cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
            label = f'{item.class_name} {item.class_confidence:.2f}'
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
            color = (
                (45, 210, 245)
                if item.validity == 'VALID'
                else (30, 140, 255)
            )
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
        with self._condition:
            input_fresh = (
                self._last_pair_monotonic is not None
                and now - self._last_pair_monotonic
                <= self._input_freshness_sec
            )
            depth_ok = bool(self._registrar_ready) if self._require_depth else True
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
                'require_depth': self._require_depth,
                'input_fresh': input_fresh,
                'processing_enabled': self._processing_enabled,
                'depth_scale_verified': self._depth_scale_verified,
                'metric_calibration_verified': (
                    not self._additional_status_flags
                ),
                'available_pose_mode': 'PLANAR_4DOF_WITH_NORMAL_PRIOR',
                'full_6d_available': False,
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
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
