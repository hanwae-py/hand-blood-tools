"""Native-depth decoding and depth-to-color registration.

This module is ROS-independent.  Transport adapters may pass the ``data`` and
``format`` fields of a ROS ``sensor_msgs/CompressedImage`` directly to
``decode_compressed_depth_16uc1`` and then register the decoded depth with a
cached ``DepthToColorRegistrar``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

from .types import CameraCalibration


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform satisfying ``P_target = R @ P_source + t``."""

    rotation: np.ndarray
    translation_m: np.ndarray
    source_frame: str
    target_frame: str
    calibration_version: str

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64).reshape(-1)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_m must be a finite 3-vector")
        orthogonality_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
        )
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > 1e-4 or abs(determinant - 1.0) > 1e-4:
            raise ValueError("rotation must be a proper orthonormal rotation matrix")
        if not self.source_frame or not self.target_frame or not self.calibration_version:
            raise ValueError(
                "source_frame, target_frame and calibration_version are required"
            )
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)


@dataclass(frozen=True)
class DepthRegistrationResult:
    """RGB-sized metric z-depth and registration diagnostics."""

    aligned_depth_m: np.ndarray
    source_valid_pixels: int
    projected_points: int
    aligned_valid_pixels: int
    z_buffer_collisions: int
    depth_scale_m_per_unit: float
    source_frame: str
    target_frame: str

    @property
    def aligned_valid_ratio(self) -> float:
        return float(
            self.aligned_valid_pixels / max(int(self.aligned_depth_m.size), 1)
        )


def decode_compressed_depth_16uc1(
    payload: bytes | bytearray | memoryview | np.ndarray,
    message_format: str,
) -> np.ndarray:
    """Decode ROS image_transport ``16UC1; compressedDepth png`` payload.

    The standard transport prepends a small codec header to the PNG.  Locating
    the PNG signature rather than assuming a fixed header size keeps this
    decoder compatible with the payload recorded in the reference MCAP.
    Float inverse-depth ``32FC1`` transport is deliberately rejected because
    it has different codec semantics and is not present in the reference bag.
    """

    declared_encoding = message_format.split(";", 1)[0].strip().upper()
    if declared_encoding != "16UC1" or "compresseddepth" not in message_format.lower():
        raise ValueError(
            "expected ROS compressedDepth format '16UC1; compressedDepth ...'"
        )
    encoded = bytes(payload)
    signature_offset = encoded.find(PNG_SIGNATURE)
    if signature_offset < 0:
        raise ValueError("compressedDepth payload does not contain a PNG signature")
    depth = cv2.imdecode(
        np.frombuffer(encoded[signature_offset:], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if depth is None:
        raise ValueError("OpenCV could not decode the compressedDepth PNG")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(
            f"decoded compressedDepth must be uint16 HxW, got {depth.dtype} {depth.shape}"
        )
    return depth


def validate_rgb_depth_timestamps(
    rgb_stamp_ns: int,
    depth_stamp_ns: int,
    maximum_delta_ns: int = 1_000_000,
) -> int:
    """Return absolute stamp delta or raise when a pair exceeds tolerance."""

    if maximum_delta_ns < 0:
        raise ValueError("maximum_delta_ns must be non-negative")
    delta_ns = abs(int(rgb_stamp_ns) - int(depth_stamp_ns))
    if delta_ns > int(maximum_delta_ns):
        raise ValueError(
            f"RGB_DEPTH_TIMESTAMP_MISMATCH: delta_ns={delta_ns} "
            f"> maximum_delta_ns={maximum_delta_ns}"
        )
    return delta_ns


class DepthToColorRegistrar:
    """Register repeated native-depth frames into an RGB camera image.

    Depth-camera rays are cached at construction time.  Every ``register``
    call performs metric unprojection, the supplied rigid transform, color
    projection with distortion, and nearest-z buffering.
    """

    def __init__(
        self,
        depth_camera: CameraCalibration,
        color_camera: CameraCalibration,
        color_from_depth: RigidTransform,
        *,
        backend: str = "numpy",
        allow_sticky_numpy_fallback: bool = False,
        cuda_library_path: str | None = None,
        cuda_device_id: int = 0,
    ) -> None:
        if color_from_depth.source_frame != depth_camera.frame_name:
            raise ValueError(
                "color_from_depth.source_frame must match depth_camera.frame_name"
            )
        if color_from_depth.target_frame != color_camera.frame_name:
            raise ValueError(
                "color_from_depth.target_frame must match color_camera.frame_name"
            )
        self.depth_camera = depth_camera
        self.color_camera = color_camera
        self.color_from_depth = color_from_depth
        if len(color_camera.distortion) not in (0, 4, 5, 8, 12):
            raise ValueError(
                "color distortion must use OpenCV plumb_bob coefficients "
                "of length 0, 4, 5, 8 or 12"
            )
        rows, columns = np.indices(
            (depth_camera.height, depth_camera.width), dtype=np.float64
        )
        pixels = np.column_stack((columns.ravel(), rows.ravel()))
        normalized = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            depth_camera.k,
            depth_camera.distortion,
        ).reshape(-1, 2)
        self._depth_rays = np.column_stack(
            (normalized, np.ones(len(normalized), dtype=np.float64))
        ).astype(np.float32)
        requested_backend = str(backend).strip().lower()
        if requested_backend not in ("numpy", "cuda"):
            raise ValueError("depth registration backend must be 'numpy' or 'cuda'")
        self.requested_backend = requested_backend
        self.allow_sticky_numpy_fallback = bool(allow_sticky_numpy_fallback)
        self.backend_name = "numpy_reference"
        self.backend_version = str(np.__version__)
        self.fallback_active = False
        self.fallback_count = 0
        self.last_backend_error = ""
        self.last_registration_ms = 0.0
        self.last_gpu_ms = 0.0
        self._closed = False
        self._cuda_backend = None
        if requested_backend == "cuda":
            try:
                from ._depth_registration_cuda import CudaDepthRegistrar

                self._cuda_backend = CudaDepthRegistrar(
                    depth_rays=self._depth_rays,
                    depth_width=depth_camera.width,
                    depth_height=depth_camera.height,
                    color_width=color_camera.width,
                    color_height=color_camera.height,
                    rotation=color_from_depth.rotation,
                    translation_m=color_from_depth.translation_m,
                    color_k=color_camera.k,
                    distortion=color_camera.distortion,
                    library_path=cuda_library_path,
                    device_id=int(cuda_device_id),
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                if not self.allow_sticky_numpy_fallback:
                    raise RuntimeError(
                        f"CUDA depth registration initialization failed: {exc}"
                    ) from exc
                self._activate_numpy_fallback(exc)
            else:
                self.backend_name = "cuda_cabi_v1"
                self.backend_version = self._cuda_backend.library_version

    def _activate_numpy_fallback(self, error: BaseException) -> None:
        """Switch once to the reference path and never retry per frame."""

        if self._cuda_backend is not None:
            self._cuda_backend.close()
            self._cuda_backend = None
        if not self.fallback_active:
            self.fallback_count += 1
        self.fallback_active = True
        self.last_backend_error = f"{type(error).__name__}: {error}"
        self.backend_name = "numpy_reference"
        self.backend_version = str(np.__version__)

    def close(self) -> None:
        """Release native resources deterministically; safe to call repeatedly."""

        if self._cuda_backend is not None:
            self._cuda_backend.close()
            self._cuda_backend = None
        self._closed = True

    def register(
        self,
        native_depth: np.ndarray,
        depth_scale_m_per_unit: float,
        minimum_depth_m: float = 0.05,
        maximum_depth_m: float = 10.0,
    ) -> DepthRegistrationResult:
        """Return color-camera z-depth in metres with invalid pixels as NaN."""

        if self._closed:
            raise RuntimeError("depth registration context is closed")
        source = np.asarray(native_depth)
        expected = (self.depth_camera.height, self.depth_camera.width)
        if source.ndim != 2 or source.shape != expected:
            raise ValueError(f"native_depth shape {source.shape} != {expected}")
        if not np.issubdtype(source.dtype, np.number):
            raise TypeError("native_depth must use a numeric dtype")
        scale = float(depth_scale_m_per_unit)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("depth_scale_m_per_unit must be finite and positive")
        minimum = float(minimum_depth_m)
        maximum = float(maximum_depth_m)
        if not 0.0 <= minimum < maximum or not np.isfinite(maximum):
            raise ValueError("depth limits must satisfy 0 <= minimum < maximum")

        registration_started = time.perf_counter()
        if self._cuda_backend is not None:
            try:
                cuda_result = self._cuda_backend.register(
                    source,
                    scale,
                    minimum,
                    maximum,
                )
            except RuntimeError as exc:
                if not self.allow_sticky_numpy_fallback:
                    raise
                self._activate_numpy_fallback(exc)
            else:
                self.last_gpu_ms = cuda_result.gpu_elapsed_ms
                self.last_registration_ms = (
                    time.perf_counter() - registration_started
                ) * 1000.0
                return DepthRegistrationResult(
                    aligned_depth_m=cuda_result.aligned_depth_m,
                    source_valid_pixels=cuda_result.source_valid_pixels,
                    projected_points=cuda_result.projected_points,
                    aligned_valid_pixels=cuda_result.aligned_valid_pixels,
                    z_buffer_collisions=(
                        cuda_result.projected_points
                        - cuda_result.aligned_valid_pixels
                    ),
                    depth_scale_m_per_unit=scale,
                    source_frame=self.depth_camera.frame_name,
                    target_frame=self.color_camera.frame_name,
                )

        depth_m = source.reshape(-1).astype(np.float32, copy=False) * scale
        valid = (
            np.isfinite(depth_m)
            & (depth_m >= minimum)
            & (depth_m <= maximum)
        )
        source_valid_pixels = int(np.count_nonzero(valid))
        output_shape = (self.color_camera.height, self.color_camera.width)
        if source_valid_pixels == 0:
            result = DepthRegistrationResult(
                aligned_depth_m=np.full(output_shape, np.nan, dtype=np.float32),
                source_valid_pixels=0,
                projected_points=0,
                aligned_valid_pixels=0,
                z_buffer_collisions=0,
                depth_scale_m_per_unit=scale,
                source_frame=self.depth_camera.frame_name,
                target_frame=self.color_camera.frame_name,
            )
            self.last_registration_ms = (
                time.perf_counter() - registration_started
            ) * 1000.0
            self.last_gpu_ms = 0.0
            return result

        rays = self._depth_rays[valid]
        depth_values = depth_m[valid]
        x_depth = rays[:, 0] * depth_values
        y_depth = rays[:, 1] * depth_values
        z_depth = depth_values
        rotation = self.color_from_depth.rotation.astype(np.float32)
        translation = self.color_from_depth.translation_m.astype(np.float32)
        x_color = (
            rotation[0, 0] * x_depth
            + rotation[0, 1] * y_depth
            + rotation[0, 2] * z_depth
            + translation[0]
        )
        y_color = (
            rotation[1, 0] * x_depth
            + rotation[1, 1] * y_depth
            + rotation[1, 2] * z_depth
            + translation[1]
        )
        z_color = (
            rotation[2, 0] * x_depth
            + rotation[2, 1] * y_depth
            + rotation[2, 2] * z_depth
            + translation[2]
        )
        in_front = (
            np.isfinite(x_color)
            & np.isfinite(y_color)
            & np.isfinite(z_color)
            & (z_color > 0.0)
        )
        x_color = x_color[in_front]
        y_color = y_color[in_front]
        z_color = z_color[in_front]
        if len(z_color):
            normalized_x = x_color / z_color
            normalized_y = y_color / z_color
            squared_x = normalized_x * normalized_x
            squared_y = normalized_y * normalized_y
            radius2 = squared_x + squared_y
            radius4 = radius2 * radius2
            radius6 = radius4 * radius2
            coefficients = np.zeros(12, dtype=np.float32)
            coefficients[: len(self.color_camera.distortion)] = (
                self.color_camera.distortion.astype(np.float32)
            )
            k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = coefficients
            radial = (
                1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6
            ) / (1.0 + k4 * radius2 + k5 * radius4 + k6 * radius6)
            xy = normalized_x * normalized_y
            distorted_x = (
                normalized_x * radial
                + 2.0 * p1 * xy
                + p2 * (radius2 + 2.0 * squared_x)
                + s1 * radius2
                + s2 * radius4
            )
            distorted_y = (
                normalized_y * radial
                + p1 * (radius2 + 2.0 * squared_y)
                + 2.0 * p2 * xy
                + s3 * radius2
                + s4 * radius4
            )
            camera_matrix = self.color_camera.k.astype(np.float32)
            pixel_x = np.rint(
                camera_matrix[0, 0] * distorted_x
                + camera_matrix[0, 1] * distorted_y
                + camera_matrix[0, 2]
            ).astype(np.int64)
            pixel_y = np.rint(
                camera_matrix[1, 0] * distorted_x
                + camera_matrix[1, 1] * distorted_y
                + camera_matrix[1, 2]
            ).astype(np.int64)
            inside = (
                (pixel_x >= 0)
                & (pixel_x < self.color_camera.width)
                & (pixel_y >= 0)
                & (pixel_y < self.color_camera.height)
            )
            pixels = np.column_stack((pixel_x[inside], pixel_y[inside]))
            z_color = z_color[inside]
        else:
            pixels = np.empty((0, 2), dtype=np.int64)
            z_color = np.empty(0, dtype=np.float32)

        linear = pixels[:, 1] * self.color_camera.width + pixels[:, 0]
        aligned_flat = np.full(
            self.color_camera.height * self.color_camera.width,
            np.inf,
            dtype=np.float32,
        )
        np.minimum.at(aligned_flat, linear, z_color)
        aligned_valid = np.isfinite(aligned_flat)
        aligned_valid_pixels = int(np.count_nonzero(aligned_valid))
        aligned_flat[~aligned_valid] = np.nan
        projected_points = int(len(linear))
        result = DepthRegistrationResult(
            aligned_depth_m=aligned_flat.reshape(output_shape),
            source_valid_pixels=source_valid_pixels,
            projected_points=projected_points,
            aligned_valid_pixels=aligned_valid_pixels,
            z_buffer_collisions=projected_points - aligned_valid_pixels,
            depth_scale_m_per_unit=scale,
            source_frame=self.depth_camera.frame_name,
            target_frame=self.color_camera.frame_name,
        )
        self.last_registration_ms = (
            time.perf_counter() - registration_started
        ) * 1000.0
        self.last_gpu_ms = 0.0
        return result


def rigid_transform_from_realsense_extrinsics(
    rotation_column_major: Any,
    translation_m: Any,
    source_frame: str,
    target_frame: str,
    calibration_version: str,
) -> RigidTransform:
    """Build ``T_target_from_source`` from RealSense Extrinsics fields.

    ``realsense2_camera_msgs/Extrinsics.rotation`` is a flat column-major
    vector.  Callers that have already validated and expanded that vector may
    pass a 3x3 matrix directly; it must not be transposed a second time.
    """

    raw_rotation = np.asarray(rotation_column_major, dtype=np.float64)
    if raw_rotation.shape == (3, 3):
        rotation = raw_rotation
    else:
        rotation = raw_rotation.reshape((3, 3), order="F")

    return RigidTransform(
        rotation=rotation,
        translation_m=np.asarray(translation_m, dtype=np.float64),
        source_frame=source_frame,
        target_frame=target_frame,
        calibration_version=calibration_version,
    )


def registrar_from_camera_fields(
    *,
    color_width: int,
    color_height: int,
    color_k: Any,
    color_d: Any,
    color_frame: str,
    depth_width: int,
    depth_height: int,
    depth_k: Any,
    depth_d: Any,
    depth_frame: str,
    rotation: Any,
    translation_m: Any,
    calibration_version: str,
    backend: str = "numpy",
    allow_sticky_numpy_fallback: bool = False,
    cuda_library_path: str | None = None,
    cuda_device_id: int = 0,
) -> DepthToColorRegistrar:
    """Build a depth-to-color registrar from CameraInfo-equivalent fields."""

    color_camera = CameraCalibration(
        width=int(color_width),
        height=int(color_height),
        k=np.asarray(color_k, dtype=np.float64).reshape(3, 3),
        distortion=np.asarray(color_d, dtype=np.float64),
        frame_name=str(color_frame),
        calibration_version=f"{calibration_version}:color",
    )
    depth_camera = CameraCalibration(
        width=int(depth_width),
        height=int(depth_height),
        k=np.asarray(depth_k, dtype=np.float64).reshape(3, 3),
        distortion=np.asarray(depth_d, dtype=np.float64),
        frame_name=str(depth_frame),
        calibration_version=f"{calibration_version}:depth",
    )
    transform = rigid_transform_from_realsense_extrinsics(
        rotation,
        translation_m,
        source_frame=depth_camera.frame_name,
        target_frame=color_camera.frame_name,
        calibration_version=f"{calibration_version}:depth_to_color",
    )
    return DepthToColorRegistrar(
        depth_camera,
        color_camera,
        transform,
        backend=backend,
        allow_sticky_numpy_fallback=allow_sticky_numpy_fallback,
        cuda_library_path=cuda_library_path,
        cuda_device_id=cuda_device_id,
    )


def finite_vector_or_none(values: Any, length: int) -> np.ndarray | None:
    """Return a finite 1-D vector, or None when the shape/values are unusable."""

    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def registrar_from_camera_messages(
    color_info: Any,
    depth_info: Any,
    rotation: Any,
    translation_m: Any,
    calibration_version: str,
    *,
    backend: str = "numpy",
    allow_sticky_numpy_fallback: bool = False,
    cuda_library_path: str | None = None,
    cuda_device_id: int = 0,
) -> DepthToColorRegistrar:
    """Build a registrar from ROS ``CameraInfo``-like messages."""

    color_frame = str(color_info.header.frame_id) or "color_optical_frame"
    depth_frame = str(depth_info.header.frame_id) or "depth_optical_frame"
    return registrar_from_camera_fields(
        color_width=int(color_info.width),
        color_height=int(color_info.height),
        color_k=color_info.k,
        color_d=color_info.d,
        color_frame=color_frame,
        depth_width=int(depth_info.width),
        depth_height=int(depth_info.height),
        depth_k=depth_info.k,
        depth_d=depth_info.d,
        depth_frame=depth_frame,
        rotation=rotation,
        translation_m=translation_m,
        calibration_version=calibration_version,
        backend=backend,
        allow_sticky_numpy_fallback=allow_sticky_numpy_fallback,
        cuda_library_path=cuda_library_path,
        cuda_device_id=cuda_device_id,
    )


def metric_depth_in_rgb_frame(
    native_depth: np.ndarray,
    rgb_height: int,
    rgb_width: int,
    depth_scale_m_per_unit: float,
    registrar: DepthToColorRegistrar | None = None,
) -> np.ndarray | None:
    """Return RGB-sized metric z-depth, or None when that mapping is unsafe.

    A supplied registrar always wins, even when native depth and RGB happen to
    share the same HxW.  Equal resolution does not prove equal optical frames
    or intrinsics.  With no registrar, same HxW is accepted only for callers
    that have already established an aligned-color-grid contract.
    """

    source = np.asarray(native_depth)
    if source.ndim != 2:
        return None
    rgb_shape = (int(rgb_height), int(rgb_width))
    if registrar is not None:
        expected_depth = (
            registrar.depth_camera.height,
            registrar.depth_camera.width,
        )
        expected_color = (
            registrar.color_camera.height,
            registrar.color_camera.width,
        )
        if source.shape != expected_depth or expected_color != rgb_shape:
            return None
        try:
            return registrar.register(
                source, depth_scale_m_per_unit
            ).aligned_depth_m
        except (TypeError, ValueError):
            return None
    if source.shape != rgb_shape:
        return None
    scale = float(depth_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    depth_m = source.astype(np.float32) * scale
    depth_m[source == 0] = 0.0
    return depth_m
