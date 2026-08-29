"""Private ctypes adapter for the versioned CUDA depth-registration C ABI.

The native library deliberately has no Python, NumPy or OpenCV ABI dependency.
This module owns all ndarray normalization and returns a new Python-owned output
array on every call.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import weakref

import numpy as np


ABI_VERSION = 1
INPUT_U16 = 1
INPUT_F32 = 2
ERROR_CAPACITY = 1024
LIBRARY_ENV = "PNU_DEPTH_REGISTRATION_CUDA_LIBRARY"


class CudaDepthRegistrationError(RuntimeError):
    """The CUDA registration backend could not be created or executed."""


class _ConfigV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_id", ctypes.c_int32),
        ("depth_width", ctypes.c_int32),
        ("depth_height", ctypes.c_int32),
        ("color_width", ctypes.c_int32),
        ("color_height", ctypes.c_int32),
        ("distortion_count", ctypes.c_int32),
        ("rotation_row_major", ctypes.c_float * 9),
        ("translation_m", ctypes.c_float * 3),
        ("color_k_row_major", ctypes.c_float * 9),
        ("distortion", ctypes.c_float * 12),
    ]


class _DiagnosticsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("source_valid_pixels", ctypes.c_uint64),
        ("projected_points", ctypes.c_uint64),
        ("aligned_valid_pixels", ctypes.c_uint64),
        ("gpu_elapsed_ms", ctypes.c_float),
    ]


@dataclass(frozen=True)
class CudaRegistrationOutput:
    aligned_depth_m: np.ndarray
    source_valid_pixels: int
    projected_points: int
    aligned_valid_pixels: int
    gpu_elapsed_ms: float


def _candidate_library_paths(explicit_path: str | os.PathLike[str] | None):
    if explicit_path:
        yield Path(explicit_path).expanduser()
        return
    configured = os.environ.get(LIBRARY_ENV, "").strip()
    if configured:
        yield Path(configured).expanduser()
        return
    install_lib = (
        Path.home()
        / "opt"
        / "pnu-depth-registration-cuda-0.1.0-cuda12.8-sm86"
        / "lib"
    )
    yield install_lib / "libpnu_depth_registration_cuda.so.1"
    yield install_lib / "libpnu_depth_registration_cuda.so"


def _load_library(explicit_path: str | os.PathLike[str] | None):
    attempted = []
    for candidate in _candidate_library_paths(explicit_path):
        attempted.append(str(candidate))
        if not candidate.is_file():
            continue
        try:
            library = ctypes.CDLL(
                str(candidate), mode=getattr(ctypes, "RTLD_LOCAL", 0)
            )
        except OSError as exc:
            raise CudaDepthRegistrationError(
                f"could not load CUDA depth-registration library {candidate}: {exc}"
            ) from exc
        try:
            _configure_library(library)
            abi = int(library.pnu_dcr_abi_version())
        except AttributeError as exc:
            raise CudaDepthRegistrationError(
                f"CUDA depth-registration library {candidate} is missing "
                f"required ABI symbols: {exc}"
            ) from exc
        if abi != ABI_VERSION:
            raise CudaDepthRegistrationError(
                f"CUDA depth-registration ABI {abi} != expected {ABI_VERSION}"
            )
        return library, candidate
    raise CudaDepthRegistrationError(
        "CUDA depth-registration library was not found; tried "
        + ", ".join(attempted)
    )


def _configure_library(library) -> None:
    library.pnu_dcr_abi_version.argtypes = []
    library.pnu_dcr_abi_version.restype = ctypes.c_uint32
    library.pnu_dcr_library_version.argtypes = []
    library.pnu_dcr_library_version.restype = ctypes.c_char_p
    library.pnu_dcr_create_v1.argtypes = [
        ctypes.POINTER(_ConfigV1),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.pnu_dcr_create_v1.restype = ctypes.c_int
    library.pnu_dcr_register_v1.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(_DiagnosticsV1),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.pnu_dcr_register_v1.restype = ctypes.c_int
    library.pnu_dcr_destroy_v1.argtypes = [ctypes.c_void_p]
    library.pnu_dcr_destroy_v1.restype = None


def _error_text(buffer: ctypes.Array[ctypes.c_char]) -> str:
    return bytes(buffer.value).decode("utf-8", errors="replace") or "unknown error"


def _destroy_context(library, handle_value: int, owner_pid: int) -> None:
    if handle_value and os.getpid() == owner_pid:
        library.pnu_dcr_destroy_v1(ctypes.c_void_p(handle_value))


class CudaDepthRegistrar:
    """Persistent CUDA context for one fixed depth/color calibration."""

    def __init__(
        self,
        *,
        depth_rays: np.ndarray,
        depth_width: int,
        depth_height: int,
        color_width: int,
        color_height: int,
        rotation: np.ndarray,
        translation_m: np.ndarray,
        color_k: np.ndarray,
        distortion: np.ndarray,
        library_path: str | os.PathLike[str] | None = None,
        device_id: int = 0,
    ) -> None:
        self._library, loaded_path = _load_library(library_path)
        self.library_path = str(loaded_path)
        self.library_version = (
            self._library.pnu_dcr_library_version().decode(
                "utf-8", errors="replace"
            )
        )
        self.depth_shape = (int(depth_height), int(depth_width))
        self.color_shape = (int(color_height), int(color_width))
        ray_array = np.asarray(depth_rays, dtype=np.float32)
        if ray_array.shape != (depth_width * depth_height, 3):
            raise ValueError(
                f"depth_rays shape {ray_array.shape} does not match depth image"
            )
        ray_xy = np.ascontiguousarray(ray_array[:, :2], dtype=np.float32)
        rotation_array = np.asarray(rotation, dtype=np.float32).reshape(9)
        translation_array = np.asarray(translation_m, dtype=np.float32).reshape(3)
        k_array = np.asarray(color_k, dtype=np.float32).reshape(9)
        distortion_array = np.asarray(distortion, dtype=np.float32).reshape(-1)
        if distortion_array.size not in (0, 4, 5, 8, 12):
            raise ValueError("unsupported color distortion coefficient count")
        padded_distortion = np.zeros(12, dtype=np.float32)
        padded_distortion[: distortion_array.size] = distortion_array

        config = _ConfigV1()
        config.struct_size = ctypes.sizeof(_ConfigV1)
        config.abi_version = ABI_VERSION
        config.device_id = int(device_id)
        config.depth_width = int(depth_width)
        config.depth_height = int(depth_height)
        config.color_width = int(color_width)
        config.color_height = int(color_height)
        config.distortion_count = int(distortion_array.size)
        config.rotation_row_major[:] = rotation_array.tolist()
        config.translation_m[:] = translation_array.tolist()
        config.color_k_row_major[:] = k_array.tolist()
        config.distortion[:] = padded_distortion.tolist()

        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        status = self._library.pnu_dcr_create_v1(
            ctypes.byref(config),
            ctypes.c_void_p(ray_xy.ctypes.data),
            ray_xy.shape[0],
            ctypes.byref(handle),
            error,
            len(error),
        )
        if status != 0 or not handle.value:
            raise CudaDepthRegistrationError(
                f"CUDA context creation failed ({status}): {_error_text(error)}"
            )
        self._owner_pid = os.getpid()
        self._handle = handle
        self._lock = threading.Lock()
        self._finalizer = weakref.finalize(
            self,
            _destroy_context,
            self._library,
            int(handle.value),
            self._owner_pid,
        )
        self.last_gpu_ms = 0.0

    def close(self) -> None:
        with self._lock:
            self._finalizer()
            self._handle = ctypes.c_void_p()

    def register(
        self,
        native_depth: np.ndarray,
        depth_scale_m_per_unit: float,
        minimum_depth_m: float,
        maximum_depth_m: float,
    ) -> CudaRegistrationOutput:
        source = np.asarray(native_depth)
        if source.shape != self.depth_shape:
            raise ValueError(
                f"native_depth shape {source.shape} != {self.depth_shape}"
            )
        if source.dtype == np.uint16:
            contiguous = np.ascontiguousarray(source)
            input_type = INPUT_U16
        else:
            contiguous = np.ascontiguousarray(source, dtype=np.float32)
            input_type = INPUT_F32
        output = np.empty(self.color_shape, dtype=np.float32)
        diagnostics = _DiagnosticsV1()
        diagnostics.struct_size = ctypes.sizeof(_DiagnosticsV1)
        diagnostics.abi_version = ABI_VERSION
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        with self._lock:
            if not self._finalizer.alive or not self._handle.value:
                raise CudaDepthRegistrationError(
                    "CUDA depth-registration context is closed"
                )
            if os.getpid() != self._owner_pid:
                raise CudaDepthRegistrationError(
                    "CUDA depth-registration context cannot be used after fork"
                )
            status = self._library.pnu_dcr_register_v1(
                self._handle,
                ctypes.c_void_p(contiguous.ctypes.data),
                contiguous.size,
                input_type,
                float(depth_scale_m_per_unit),
                float(minimum_depth_m),
                float(maximum_depth_m),
                ctypes.c_void_p(output.ctypes.data),
                output.size,
                ctypes.byref(diagnostics),
                error,
                len(error),
            )
        if status != 0:
            raise CudaDepthRegistrationError(
                f"CUDA registration failed ({status}): {_error_text(error)}"
            )
        self.last_gpu_ms = float(diagnostics.gpu_elapsed_ms)
        return CudaRegistrationOutput(
            aligned_depth_m=output,
            source_valid_pixels=int(diagnostics.source_valid_pixels),
            projected_points=int(diagnostics.projected_points),
            aligned_valid_pixels=int(diagnostics.aligned_valid_pixels),
            gpu_elapsed_ms=self.last_gpu_ms,
        )
