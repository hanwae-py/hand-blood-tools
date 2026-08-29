#!/usr/bin/env python3
"""ROS 2 lifecycle adapter for the supplied RF-DETR Seg-Small Blood model.

RGB recognition publishes 2D masks always. When a fresh compressedDepth frame
matches the RGB stamp, each mask centroid is sampled for metric depth in the
RGB frame: same HxW is used directly, otherwise native depth is registered
with color/depth CameraInfo and the CAM4 depth-to-color extrinsics. Missing
or unmappable depth skips those fields. Set require_depth to skip frames that
have no usable depth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String


def reliable_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def image_reader_qos() -> QoSProfile:
    """Latest frame only; compatible with local ingress BEST_EFFORT fan-out."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def camera_info_qos() -> QoSProfile:
    """Reliable CameraInfo is distinct from the lossy image transport."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=np.uint8)
    flat = binary.reshape(-1, order="F") != 0
    if flat.size == 0:
        counts = [0]
    else:
        starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
        lengths = np.diff(np.r_[starts, flat.size]).astype(np.int64)
        counts = lengths.tolist()
        if bool(flat[0]):
            counts.insert(0, 0)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def mask_centroid(mask: np.ndarray) -> list[float] | None:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def sample_centroid_depth_m(
    depth_m: np.ndarray, centroid_xy: list[float] | None
) -> float | None:
    if centroid_xy is None or depth_m.ndim != 2:
        return None
    u = int(round(centroid_xy[0]))
    v = int(round(centroid_xy[1]))
    if v < 0 or u < 0 or v >= depth_m.shape[0] or u >= depth_m.shape[1]:
        return None
    value = float(depth_m[v, u])
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def image_quality_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    """Small exposure/readiness summary used to fail closed on dark FLIR input."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty BGR image")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    p01, median, p99 = np.percentile(gray, [1.0, 50.0, 99.0])
    return {
        "gray_p01": float(p01),
        "gray_median": float(median),
        "gray_p99": float(p99),
        "gray_dynamic_range": float(p99 - p01),
        "gray_max": float(np.max(gray)),
    }


@dataclass(frozen=True)
class _FrameJob:
    """One immutable RGB job consumed by exactly one inference worker.

    The ROS executor only writes the latest slot.  This prevents camera input
    callbacks from accumulating unbounded work while keeping RF-DETR on a
    single GPU-owning thread.
    """

    generation: int
    camera: str
    source: CompressedImage
    depth: CompressedImage | None
    color_info: CameraInfo | None
    depth_info: CameraInfo | None
    require_depth: bool
    reject_low_quality: bool
    minimum_gray_p99: float
    minimum_gray_dynamic_range: float
    confidence_threshold: float
    maximum_stamp_delta_ns: int
    depth_scale_m_per_unit: float
    depth_to_color_rotation: tuple[float, ...]
    depth_to_color_translation_m: tuple[float, ...]
    calibration_version: str


@dataclass
class _CompletedFrame:
    """A worker-built result that is published later by the ROS executor."""

    generation: int
    kind: str
    mask: Image | None = None
    overlay: CompressedImage | None = None
    semantics: String | None = None
    image_quality: dict[str, float] | None = None
    rejection_reason: str = ""
    process_ms: float | None = None
    instances: int = 0
    observation_valid: bool = False
    error: str = ""


class BloodDetectionNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("blood_detection_node")
        self.declare_parameter("camera", "cam_4")
        camera = str(self.get_parameter("camera").value).strip() or "cam_4"
        synced = f"/synced/{camera}"
        out = f"/perception/{camera}/blood"
        self.declare_parameter(
            "color_topic", f"{synced}/color/image_raw/compressed"
        )
        self.declare_parameter(
            "depth_topic", f"{synced}/depth/image_rect_raw/compressedDepth"
        )
        self.declare_parameter(
            "color_camera_info_topic", f"{synced}/color/camera_info"
        )
        self.declare_parameter(
            "depth_camera_info_topic", f"{synced}/depth/camera_info"
        )
        self.declare_parameter(
            "checkpoint", str(Path.home() / "models" / "blood_detection.pth")
        )
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("optimize", True)
        self.declare_parameter("require_depth", False)
        self.declare_parameter("reject_low_quality_input", False)
        self.declare_parameter("minimum_gray_p99", 20.0)
        self.declare_parameter("minimum_gray_dynamic_range", 12.0)
        self.declare_parameter("max_input_age_sec", 1.0)
        self.declare_parameter("maximum_stamp_delta_ns", 1_000_000)
        self.declare_parameter("depth_scale_m_per_unit", 0.001)
        self.declare_parameter("depth_to_color_rotation", [float("nan")] * 9)
        self.declare_parameter("depth_to_color_translation_m", [float("nan")] * 3)
        self.declare_parameter("calibration_version", "")
        self.declare_parameter("mask_topic", f"{out}/mask")
        self.declare_parameter("overlay_topic", f"{out}/overlay/compressed")
        self.declare_parameter("semantics_topic", f"{out}/semantics")
        self.declare_parameter("health_topic", f"{out}/health")
        self.declare_parameter("diagnostics_topic", f"{out}/diagnostics")
        # Preserve the existing coordinator-controlled default.  The new
        # local-ingress concurrent runner opts in explicitly so Tool, Hand,
        # and Blood can all remain active for the unified Debug overlay.
        self.declare_parameter("autostart", False)

        self._active = False
        self._model = None
        self._torch = None
        self._state_lock = Lock()
        self._worker_event = Event()
        self._worker_stop = Event()
        self._worker_idle = Event()
        self._worker_idle.set()
        self._worker: Thread | None = None
        self._worker_busy = False
        self._worker_generation = 0
        self._pending_job: _FrameJob | None = None
        self._completed_frame: _CompletedFrame | None = None
        self._frames_received = 0
        self._frames_dropped_latest = 0
        self._frames_dropped_completed = 0
        self._frames_skipped_no_depth = 0
        self._frames_processed = 0
        self._errors = 0
        self._last_process_ms: float | None = None
        self._last_instances = 0
        self._frames_rejected = 0
        self._last_error = ""
        self._last_image_quality: dict[str, float] = {}
        self._image_quality_ready = False
        self._last_rejection_reason = "NO_IMAGE_YET"
        self._last_input_at: float | None = None
        self._last_output_at: float | None = None
        self._last_observation_valid = False
        self._latest_depth: CompressedImage | None = None
        self._color_info: CameraInfo | None = None
        self._depth_info: CameraInfo | None = None
        self._registrar = None
        self._registrar_key: tuple[object, ...] | None = None
        self._mask_pub = None
        self._overlay_pub = None
        self._semantics_pub = None
        self._health_pub = None
        self._diagnostics_pub = None
        image_qos = image_reader_qos()
        info_qos = camera_info_qos()
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("color_topic").value),
            self._on_color,
            image_qos,
        )
        self.create_subscription(
            CompressedImage,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            image_qos,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("color_camera_info_topic").value),
            self._on_color_info,
            info_qos,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("depth_camera_info_topic").value),
            self._on_depth_info,
            info_qos,
        )
        # Worker code never publishes.  This executor timer owns all lifecycle
        # publisher access, including during deactivate/cleanup transitions.
        self.create_timer(0.02, self._drain_completed_frame)
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info("blood_detection_node created (unconfigured)")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        try:
            self._reset_observation_state()
            threshold = float(self.get_parameter("confidence_threshold").value)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence_threshold must be in [0, 1]")
            checkpoint = str(self.get_parameter("checkpoint").value)
            import torch
            from rfdetr import RFDETRSegSmall

            self.get_logger().info("configuring: loading RF-DETR Seg-Small Blood model")
            model = RFDETRSegSmall.from_checkpoint(checkpoint)
            if bool(self.get_parameter("optimize").value) and torch.cuda.is_available():
                model.optimize_for_inference(
                    compile=True, batch_size=1, dtype=torch.float16, inplace=False
                )
            self._model = model
            self._torch = torch
            self._mask_pub = self.create_lifecycle_publisher(
                Image, str(self.get_parameter("mask_topic").value), reliable_qos(5)
            )
            self._overlay_pub = self.create_lifecycle_publisher(
                CompressedImage, str(self.get_parameter("overlay_topic").value), reliable_qos(5)
            )
            self._semantics_pub = self.create_lifecycle_publisher(
                String, str(self.get_parameter("semantics_topic").value), reliable_qos(5)
            )
            self._health_pub = self.create_lifecycle_publisher(
                String, str(self.get_parameter("health_topic").value), reliable_qos(1)
            )
            self._diagnostics_pub = self.create_lifecycle_publisher(
                String, str(self.get_parameter("diagnostics_topic").value), reliable_qos(1)
            )
            self._start_worker()
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            self.get_logger().info(f"configured on {device}; waiting for activation")
            return TransitionCallbackReturn.SUCCESS
        except Exception as exc:
            self._last_error = str(exc)
            self.get_logger().error(f"Blood configuration failed: {exc}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        result = super().on_activate(state)
        if result == TransitionCallbackReturn.SUCCESS:
            with self._state_lock:
                self._worker_generation += 1
                self._pending_job = None
                self._completed_frame = None
                self._active = True
            self.get_logger().info(
                "ACTIVE: processing RGB frames for Blood masks "
                f"(require_depth={bool(self.get_parameter('require_depth').value)})"
            )
        return result

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if not self._quiesce_worker():
            self.get_logger().error("Blood worker did not quiesce before deactivate")
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info(f"INACTIVE after {self._frames_processed} processed frames")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        if not self._stop_worker():
            self.get_logger().error("Blood worker did not stop before cleanup")
            return TransitionCallbackReturn.FAILURE
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._reset_observation_state()
        self.get_logger().info("cleaned up: Blood model released")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        stopped = self._stop_worker()
        if stopped:
            self._model = None
        self._reset_observation_state()
        return TransitionCallbackReturn.SUCCESS if stopped else TransitionCallbackReturn.FAILURE

    def _reset_observation_state(self) -> None:
        self._last_image_quality = {}
        self._image_quality_ready = False
        self._last_rejection_reason = "NO_IMAGE_YET"
        self._last_input_at = None
        self._last_output_at = None
        self._last_observation_valid = False
        self._latest_depth = None
        self._color_info = None
        self._depth_info = None
        self._registrar = None
        self._registrar_key = None
        self._last_process_ms = None
        self._last_instances = 0
        self._last_error = ""
        self._pending_job = None
        self._completed_frame = None
        self._worker_busy = False
        self._frames_received = 0
        self._frames_dropped_latest = 0
        self._frames_dropped_completed = 0
        self._frames_skipped_no_depth = 0

    def _on_depth(self, message: CompressedImage) -> None:
        with self._state_lock:
            self._latest_depth = message

    def _on_color_info(self, message: CameraInfo) -> None:
        with self._state_lock:
            self._color_info = message
            self._registrar = None
            self._registrar_key = None

    def _on_depth_info(self, message: CameraInfo) -> None:
        with self._state_lock:
            self._depth_info = message
            self._registrar = None
            self._registrar_key = None

    def _depth_to_color_registrar(
        self,
        rgb_height: int,
        rgb_width: int,
        native_shape: tuple[int, ...],
        *,
        color_info: CameraInfo | None = None,
        depth_info: CameraInfo | None = None,
        depth_to_color_rotation: tuple[float, ...] | None = None,
        depth_to_color_translation_m: tuple[float, ...] | None = None,
        calibration_version: str | None = None,
    ):
        if len(native_shape) != 2:
            return None
        try:
            from pnu_surgical_tool.depth_registration import (
                finite_vector_or_none,
                registrar_from_camera_messages,
            )
        except ImportError as exc:
            self.get_logger().warn(
                f"Blood depth-to-color registration unavailable: {exc}",
                throttle_duration_sec=10.0,
            )
            return None
        if color_info is None or depth_info is None:
            with self._state_lock:
                if color_info is None:
                    color_info = self._color_info
                if depth_info is None:
                    depth_info = self._depth_info
        rotation = finite_vector_or_none(
            self.get_parameter("depth_to_color_rotation").value
            if depth_to_color_rotation is None
            else depth_to_color_rotation,
            9,
        )
        translation = finite_vector_or_none(
            self.get_parameter("depth_to_color_translation_m").value
            if depth_to_color_translation_m is None
            else depth_to_color_translation_m,
            3,
        )
        version = (
            str(self.get_parameter("calibration_version").value).strip()
            if calibration_version is None
            else calibration_version.strip()
        )
        if (
            color_info is None
            or depth_info is None
            or rotation is None
            or translation is None
            or not version
        ):
            return None
        if int(color_info.width) != rgb_width or int(color_info.height) != rgb_height:
            return None
        if (int(depth_info.height), int(depth_info.width)) != native_shape:
            return None
        key = (
            int(color_info.width),
            int(color_info.height),
            tuple(np.asarray(color_info.k, dtype=np.float64).ravel()),
            tuple(np.asarray(color_info.d, dtype=np.float64).ravel()),
            str(color_info.header.frame_id),
            int(depth_info.width),
            int(depth_info.height),
            tuple(np.asarray(depth_info.k, dtype=np.float64).ravel()),
            tuple(np.asarray(depth_info.d, dtype=np.float64).ravel()),
            str(depth_info.header.frame_id),
            tuple(rotation.ravel()),
            tuple(translation.ravel()),
            version,
        )
        with self._state_lock:
            if self._registrar is not None and self._registrar_key == key:
                return self._registrar
        try:
            registrar = registrar_from_camera_messages(
                color_info, depth_info, rotation, translation, version
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(
                f"Blood depth-to-color registrar skipped: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        with self._state_lock:
            self._registrar = registrar
            self._registrar_key = key
        return registrar

    def _aligned_depth_m(
        self,
        rgb: CompressedImage,
        height: int,
        width: int,
        *,
        depth_message: CompressedImage | None = None,
        color_info: CameraInfo | None = None,
        depth_info: CameraInfo | None = None,
        maximum_stamp_delta_ns: int | None = None,
        depth_scale_m_per_unit: float | None = None,
        depth_to_color_rotation: tuple[float, ...] | None = None,
        depth_to_color_translation_m: tuple[float, ...] | None = None,
        calibration_version: str | None = None,
    ) -> np.ndarray | None:
        if depth_message is None:
            with self._state_lock:
                depth_msg = self._latest_depth
        else:
            depth_msg = depth_message
        if depth_msg is None:
            return None
        maximum_delta_ns = (
            int(self.get_parameter("maximum_stamp_delta_ns").value)
            if maximum_stamp_delta_ns is None
            else maximum_stamp_delta_ns
        )
        if abs(stamp_ns(rgb) - stamp_ns(depth_msg)) > maximum_delta_ns:
            return None
        try:
            from pnu_surgical_tool.depth_registration import (
                decode_compressed_depth_16uc1,
                metric_depth_in_rgb_frame,
            )

            native = decode_compressed_depth_16uc1(depth_msg.data, depth_msg.format)
        except Exception as exc:
            self.get_logger().warn(
                f"Blood depth decode skipped: {exc}",
                throttle_duration_sec=5.0,
            )
            return None
        registrar = None
        if native.shape != (height, width):
            registrar = self._depth_to_color_registrar(
                height,
                width,
                native.shape,
                color_info=color_info,
                depth_info=depth_info,
                depth_to_color_rotation=depth_to_color_rotation,
                depth_to_color_translation_m=depth_to_color_translation_m,
                calibration_version=calibration_version,
            )
        aligned = metric_depth_in_rgb_frame(
            native,
            height,
            width,
            float(self.get_parameter("depth_scale_m_per_unit").value)
            if depth_scale_m_per_unit is None
            else depth_scale_m_per_unit,
            registrar,
        )
        if aligned is None and native.shape != (height, width):
            self.get_logger().warn(
                "Blood native depth HxW differs from RGB and could not be "
                "registered; centroid depth skipped",
                throttle_duration_sec=5.0,
            )
        return aligned

    def _start_worker(self) -> None:
        """Start one GPU-owning worker after the model is configured."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker_stop.clear()
        self._worker_event.clear()
        self._worker_idle.set()
        self._worker = Thread(
            target=self._worker_loop,
            name=f"{self.get_name()}-inference",
            daemon=True,
        )
        self._worker.start()

    def _quiesce_worker(self, timeout_sec: float = 20.0) -> bool:
        """Invalidate queued work and wait until any in-flight GPU call ends."""
        with self._state_lock:
            self._active = False
            self._worker_generation += 1
            self._pending_job = None
            self._completed_frame = None
            self._worker_event.set()
        return self._worker_idle.wait(timeout=max(0.1, timeout_sec))

    def _stop_worker(self, timeout_sec: float = 30.0) -> bool:
        """Stop the worker before releasing the model or lifecycle publishers."""
        worker = self._worker
        if worker is not None and not worker.is_alive():
            with self._state_lock:
                self._active = False
                self._worker_generation += 1
                self._pending_job = None
                self._completed_frame = None
                self._worker = None
                self._worker_busy = False
                self._worker_idle.set()
            return True
        self._quiesce_worker(timeout_sec=timeout_sec)
        worker = self._worker
        if worker is None:
            return True
        self._worker_stop.set()
        self._worker_event.set()
        worker.join(timeout=max(0.1, timeout_sec))
        if worker.is_alive():
            return False
        with self._state_lock:
            self._worker = None
            self._worker_busy = False
            self._worker_idle.set()
        return True

    def destroy_node(self) -> None:
        # Tests and process teardown may bypass lifecycle cleanup.
        self._stop_worker(timeout_sec=5.0)
        return super().destroy_node()

    def _on_color(self, message: CompressedImage) -> None:
        """Replace the pending frame in O(1); inference runs off the executor."""
        with self._state_lock:
            if not self._active or self._model is None:
                return
        require_depth = bool(self.get_parameter("require_depth").value)
        try:
            rotation = tuple(
                float(value)
                for value in self.get_parameter("depth_to_color_rotation").value
            )
            translation = tuple(
                float(value)
                for value in self.get_parameter(
                    "depth_to_color_translation_m"
                ).value
            )
        except (TypeError, ValueError) as exc:
            self._errors += 1
            self._last_error = f"invalid depth calibration parameter: {exc}"
            return
        if self._worker is None or not self._worker.is_alive():
            self._start_worker()
        with self._state_lock:
            if not self._active or self._model is None:
                return
            self._frames_received += 1
            self._last_input_at = time.monotonic()
            # RGB-only operation deliberately does not snapshot or decode depth.
            depth = self._latest_depth if require_depth else None
            color_info = self._color_info if require_depth else None
            depth_info = self._depth_info if require_depth else None
            if self._pending_job is not None:
                self._frames_dropped_latest += 1
            self._pending_job = _FrameJob(
                generation=self._worker_generation,
                camera=str(self.get_parameter("camera").value),
                source=message,
                depth=depth,
                color_info=color_info,
                depth_info=depth_info,
                require_depth=require_depth,
                reject_low_quality=bool(
                    self.get_parameter("reject_low_quality_input").value
                ),
                minimum_gray_p99=float(
                    self.get_parameter("minimum_gray_p99").value
                ),
                minimum_gray_dynamic_range=float(
                    self.get_parameter("minimum_gray_dynamic_range").value
                ),
                confidence_threshold=float(
                    self.get_parameter("confidence_threshold").value
                ),
                maximum_stamp_delta_ns=int(
                    self.get_parameter("maximum_stamp_delta_ns").value
                ),
                depth_scale_m_per_unit=float(
                    self.get_parameter("depth_scale_m_per_unit").value
                ),
                depth_to_color_rotation=rotation,
                depth_to_color_translation_m=translation,
                calibration_version=str(
                    self.get_parameter("calibration_version").value
                ).strip(),
            )
            self._worker_event.set()

    def _worker_loop(self) -> None:
        """Run decode, optional depth, inference, RLE, and JPEG sequentially."""
        while not self._worker_stop.is_set():
            self._worker_event.wait()
            self._worker_event.clear()
            while not self._worker_stop.is_set():
                with self._state_lock:
                    job = self._pending_job
                    self._pending_job = None
                    if job is None:
                        self._worker_busy = False
                        self._worker_idle.set()
                        break
                    self._worker_busy = True
                    self._worker_idle.clear()
                completed = self._process_frame(job)
                with self._state_lock:
                    if (
                        job.generation == self._worker_generation
                        and self._active
                        and not self._worker_stop.is_set()
                    ):
                        if self._completed_frame is not None:
                            self._frames_dropped_completed += 1
                        self._completed_frame = completed
            # A frame can arrive after the last slot check but before waiting;
            # its event remains set and causes the next loop to run immediately.
        with self._state_lock:
            self._worker_busy = False
            self._worker_idle.set()

    def _process_frame(self, job: _FrameJob) -> _CompletedFrame:
        """Build a result without calling ROS publishers from the worker."""
        try:
            data = np.frombuffer(job.source.data, dtype=np.uint8)
            image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError("failed to decode compressed RGB image")
            height, width = image_bgr.shape[:2]
            image_quality = image_quality_metrics(image_bgr)
            image_quality_ready = bool(
                image_quality["gray_p99"] >= job.minimum_gray_p99
                and image_quality["gray_dynamic_range"]
                >= job.minimum_gray_dynamic_range
            )
            if job.reject_low_quality and not image_quality_ready:
                return self._build_rejected_frame(job, image_bgr, image_quality)

            # This is intentionally the only depth path.  In RGB-only mode,
            # require_depth=False means no depth snapshot, decode, registration,
            # or centroid sampling work is performed.
            depth_m = None
            if job.require_depth:
                depth_m = self._aligned_depth_m(
                    job.source,
                    height,
                    width,
                    depth_message=job.depth,
                    color_info=job.color_info,
                    depth_info=job.depth_info,
                    maximum_stamp_delta_ns=job.maximum_stamp_delta_ns,
                    depth_scale_m_per_unit=job.depth_scale_m_per_unit,
                    depth_to_color_rotation=job.depth_to_color_rotation,
                    depth_to_color_translation_m=job.depth_to_color_translation_m,
                    calibration_version=job.calibration_version,
                )
                if depth_m is None:
                    return _CompletedFrame(
                        generation=job.generation,
                        kind="skipped_no_depth",
                        image_quality=image_quality,
                    )

            with self._state_lock:
                if (
                    not self._active
                    or job.generation != self._worker_generation
                    or self._model is None
                    or self._torch is None
                ):
                    return _CompletedFrame(generation=job.generation, kind="discarded")
                model = self._model
                torch = self._torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            detections = model.predict(
                image_bgr,
                threshold=job.confidence_threshold,
                include_source_image=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            process_ms = (time.perf_counter() - started) * 1000.0
            return self._build_result(
                job, image_bgr, detections, depth_m, image_quality, process_ms
            )
        except Exception as exc:
            return _CompletedFrame(
                generation=job.generation,
                kind="error",
                error=str(exc),
            )

    def _drain_completed_frame(self) -> None:
        """Publish a completed worker result from the ROS executor only."""
        with self._state_lock:
            completed = self._completed_frame
            self._completed_frame = None
            active = self._active
            generation = self._worker_generation
        if completed is None or not active or completed.generation != generation:
            return
        if completed.kind == "discarded":
            return
        if completed.kind == "error":
            self._errors += 1
            self._last_error = completed.error
            self.get_logger().error(
                f"Blood processing failed: {completed.error}",
                throttle_duration_sec=2.0,
            )
            return
        if completed.kind == "skipped_no_depth":
            self._frames_skipped_no_depth += 1
            self._last_image_quality = completed.image_quality or {}
            self._image_quality_ready = bool(
                self._last_image_quality
                and self._last_image_quality["gray_p99"]
                >= float(self.get_parameter("minimum_gray_p99").value)
                and self._last_image_quality["gray_dynamic_range"]
                >= float(self.get_parameter("minimum_gray_dynamic_range").value)
            )
            self._last_rejection_reason = ""
            return
        if completed.mask is not None:
            self._mask_pub.publish(completed.mask)
        if completed.overlay is not None:
            self._overlay_pub.publish(completed.overlay)
        if completed.semantics is not None:
            self._semantics_pub.publish(completed.semantics)
        self._frames_processed += 1
        if completed.kind == "rejected":
            self._frames_rejected += 1
        self._last_instances = completed.instances
        self._last_process_ms = completed.process_ms
        self._last_output_at = time.monotonic()
        self._last_observation_valid = completed.observation_valid
        self._last_image_quality = completed.image_quality or {}
        self._image_quality_ready = bool(
            self._last_image_quality
            and self._last_image_quality["gray_p99"]
            >= float(self.get_parameter("minimum_gray_p99").value)
            and self._last_image_quality["gray_dynamic_range"]
            >= float(self.get_parameter("minimum_gray_dynamic_range").value)
        )
        self._last_rejection_reason = completed.rejection_reason
        self._last_error = ""
        self.get_logger().info(
            f"published Blood masks: {completed.instances} instances",
            throttle_duration_sec=1.0,
        )

    def _build_rejected_frame(
        self,
        job: _FrameJob,
        image_bgr: np.ndarray,
        image_quality: dict[str, float],
    ) -> _CompletedFrame:
        """Build explicit UNKNOWN evidence without publishing from the worker."""
        source = job.source
        height, width = image_bgr.shape[:2]
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (width, min(height, 86)), (15, 45, 75), -1)
        cv2.putText(
            overlay,
            "FLIR INPUT DARK - BLOOD UNKNOWN",
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 190, 255),
            3,
            cv2.LINE_AA,
        )
        success, encoded = cv2.imencode(".jpg", overlay)
        if not success:
            raise RuntimeError("failed to JPEG encode rejected Blood overlay")
        overlay_msg = CompressedImage()
        overlay_msg.header = source.header
        overlay_msg.format = "jpeg"
        overlay_msg.data = encoded.tobytes()

        payload = {
            "schema": "pnu.surgical_blood_observations.v1",
            "header": {
                "stamp_sec": source.header.stamp.sec,
                "stamp_nanosec": source.header.stamp.nanosec,
                "frame_id": source.header.frame_id,
            },
            "camera": job.camera,
            "image": {"width": width, "height": height},
            "model": "RF-DETR Seg-Small",
            "classes": ["blood"],
            "depth_sampled": False,
            "instances": [],
            "observation_valid": False,
            "mask_published": False,
            "rejection_reason": "LOW_LIGHT_OR_LOW_DYNAMIC_RANGE",
            "image_quality": image_quality,
        }
        return _CompletedFrame(
            generation=job.generation,
            kind="rejected",
            overlay=overlay_msg,
            semantics=String(data=json.dumps(payload, separators=(",", ":"))),
            image_quality=image_quality,
            rejection_reason="LOW_LIGHT_OR_LOW_DYNAMIC_RANGE",
            observation_valid=False,
        )

    def _build_result(
        self,
        job: _FrameJob,
        image_bgr: np.ndarray,
        detections,
        depth_m: np.ndarray | None,
        image_quality: dict[str, float],
        process_ms: float,
    ) -> _CompletedFrame:
        source = job.source
        height, width = image_bgr.shape[:2]
        raw_masks = getattr(detections, "mask", None)
        if raw_masks is None:
            raise RuntimeError("Blood checkpoint returned no segmentation masks")
        union_mask = np.zeros((height, width), dtype=bool)
        overlay = image_bgr.copy()
        instances: list[dict[str, object]] = []
        for item_id, (box, class_id, confidence) in enumerate(
            zip(detections.xyxy, detections.class_id, detections.confidence, strict=True)
        ):
            if int(class_id) != 0:
                raise RuntimeError(f"unexpected Blood class index: {class_id}")
            mask = np.asarray(raw_masks[item_id], dtype=bool)
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            union_mask |= mask
            colored = overlay.copy()
            colored[mask] = (230, 80, 30)  # blue, BGR
            overlay = cv2.addWeighted(overlay, 0.70, colored, 0.30, 0.0)
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (230, 80, 30), 2)
            cv2.putText(overlay, f"blood {float(confidence):.2f}", (x0, max(18, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 80, 30), 2, cv2.LINE_AA)
            centroid = mask_centroid(mask)
            instance: dict[str, object] = {
                "instance_id": item_id,
                "class_id": 1,
                "class_name": "blood",
                "confidence": float(confidence),
                "bbox_xyxy_px": [float(value) for value in box],
                "centroid_xy_px": centroid,
                "mask_rle": encode_coco_rle(mask),
            }
            centroid_depth_m = (
                sample_centroid_depth_m(depth_m, centroid) if depth_m is not None else None
            )
            if centroid_depth_m is not None:
                instance["centroid_depth_m"] = centroid_depth_m
            instances.append(instance)

        mask_msg = Image()
        mask_msg.header = source.header
        mask_msg.height = height
        mask_msg.width = width
        mask_msg.encoding = "mono8"
        mask_msg.is_bigendian = False
        mask_msg.step = width
        mask_msg.data = (union_mask * 255).astype(np.uint8).tobytes()

        success, encoded = cv2.imencode(".jpg", overlay)
        if not success:
            raise RuntimeError("failed to JPEG encode Blood overlay")
        overlay_msg = CompressedImage()
        overlay_msg.header = source.header
        overlay_msg.format = "jpeg"
        overlay_msg.data = encoded.tobytes()

        union_centroid = mask_centroid(union_mask)
        union_centroid_depth_m = (
            sample_centroid_depth_m(depth_m, union_centroid) if depth_m is not None else None
        )
        payload: dict[str, object] = {
            "schema": "pnu.surgical_blood_observations.v1",
            "header": {
                "stamp_sec": source.header.stamp.sec,
                "stamp_nanosec": source.header.stamp.nanosec,
                "frame_id": source.header.frame_id,
            },
            "image": {"width": width, "height": height},
            "camera": job.camera,
            "model": "RF-DETR Seg-Small",
            "classes": ["blood"],
            "confidence_threshold": job.confidence_threshold,
            "inference_latency_ms": process_ms,
            "depth_sampled": depth_m is not None,
            "instances": instances,
            "observation_valid": True,
            "mask_published": True,
            "rejection_reason": "",
            "image_quality": image_quality,
            "combined_blood_mask_rle": encode_coco_rle(union_mask),
            "combined_blood_centroid_xy_px": union_centroid,
        }
        if union_centroid_depth_m is not None:
            payload["combined_blood_centroid_depth_m"] = union_centroid_depth_m
        return _CompletedFrame(
            generation=job.generation,
            kind="published",
            mask=mask_msg,
            overlay=overlay_msg,
            semantics=String(data=json.dumps(payload, separators=(",", ":"))),
            image_quality=image_quality,
            process_ms=process_ms,
            instances=len(instances),
            observation_valid=True,
        )

    def _publish_status(self) -> None:
        if self._health_pub is None or self._diagnostics_pub is None:
            return
        state = "active" if self._active else "inactive"
        now = time.monotonic()
        reject_low_quality = bool(
            self.get_parameter("reject_low_quality_input").value)
        max_input_age = max(
            0.1, float(self.get_parameter("max_input_age_sec").value))
        input_age = (
            None if self._last_input_at is None
            else max(0.0, now - self._last_input_at))
        output_age = (
            None if self._last_output_at is None
            else max(0.0, now - self._last_output_at))
        input_fresh = input_age is not None and input_age <= max_input_age
        with self._state_lock:
            worker_busy = self._worker_busy
            pending_frame = self._pending_job is not None
            frames_received = self._frames_received
            frames_dropped_latest = self._frames_dropped_latest
            frames_dropped_completed = self._frames_dropped_completed
            frames_skipped_no_depth = self._frames_skipped_no_depth
        self._health_pub.publish(String(data=json.dumps({
            "node": self.get_name(),
            "camera": str(self.get_parameter("camera").value),
            "ready": bool(
                self._active
                and self._model is not None
                and input_fresh
                and (not reject_low_quality or self._image_quality_ready)
                and self._last_observation_valid
                and not self._last_error
            ),
            "lifecycle_state": state, "explicit_classes": ["blood"],
            "background_is_implicit": True,
            "require_depth": bool(self.get_parameter("require_depth").value),
            "image_quality_ready": self._image_quality_ready,
            "image_quality": self._last_image_quality,
            "input_fresh": input_fresh,
            "input_age_sec": (
                None if input_age is None else round(input_age, 3)),
            "output_age_sec": (
                None if output_age is None else round(output_age, 3)),
            "max_input_age_sec": max_input_age,
            "worker_busy": worker_busy,
            "pending_frame": pending_frame,
            "last_observation_valid": self._last_observation_valid,
            "rejection_reason": self._last_rejection_reason,
            "last_error": self._last_error,
        })))
        self._diagnostics_pub.publish(String(data=json.dumps({
            "node": self.get_name(),
            "camera": str(self.get_parameter("camera").value),
            "lifecycle_state": state,
            "frames_processed": self._frames_processed,
            "frames_rejected": self._frames_rejected,
            "frames_inferred": self._frames_processed - self._frames_rejected,
            "frames_received": frames_received,
            "frames_dropped_latest": frames_dropped_latest,
            "frames_dropped_completed": frames_dropped_completed,
            "frames_skipped_no_depth": frames_skipped_no_depth,
            "worker_busy": worker_busy,
            "pending_frame": pending_frame,
            "output_age_sec": (
                None if output_age is None else round(output_age, 3)),
            "blood_instances_last_frame": self._last_instances,
            "last_process_ms": self._last_process_ms, "errors": self._errors,
        })))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = BloodDetectionNode()
        if bool(node.get_parameter("autostart").value):
            node.get_logger().info(
                "autostart:=true -- self-configuring and activating"
            )
            if node.trigger_configure() != TransitionCallbackReturn.SUCCESS:
                raise RuntimeError("Blood lifecycle configure failed")
            if node.trigger_activate() != TransitionCallbackReturn.SUCCESS:
                raise RuntimeError("Blood lifecycle activate failed")
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
