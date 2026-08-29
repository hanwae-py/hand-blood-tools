from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pnu_surgical_tool.depth_registration import registrar_from_camera_fields


def _registrar(
    *,
    backend: str = "numpy",
    allow_fallback: bool = False,
    library_path: str | None = None,
    distortion=(),
    translation=(0.01, 0.0, 0.0),
    color_width: int = 16,
    color_height: int = 12,
    depth_width: int = 8,
    depth_height: int = 6,
):
    return registrar_from_camera_fields(
        color_width=color_width,
        color_height=color_height,
        color_k=[
            [80.0, 0.125, color_width / 2.0],
            [0.0625, 79.0, color_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        color_d=distortion,
        color_frame="color",
        depth_width=depth_width,
        depth_height=depth_height,
        depth_k=[
            [40.0, 0.0, depth_width / 2.0],
            [0.0, 39.5, depth_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        depth_d=[0.001, -0.0002, 0.0001, -0.0001, 0.00001],
        depth_frame="depth",
        rotation=np.eye(3),
        translation_m=translation,
        calibration_version="backend-test",
        backend=backend,
        allow_sticky_numpy_fallback=allow_fallback,
        cuda_library_path=library_path,
    )


def _installed_cuda_library() -> str | None:
    configured = os.environ.get("PNU_DEPTH_REGISTRATION_CUDA_LIBRARY", "")
    candidates = [
        Path(configured) if configured else None,
        Path.home()
        / "opt"
        / "pnu-depth-registration-cuda-0.1.0-cuda12.8-sm86"
        / "lib"
        / "libpnu_depth_registration_cuda.so.1",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    return None


def _assert_equivalent(reference, candidate) -> None:
    assert candidate.aligned_depth_m.shape == reference.aligned_depth_m.shape
    assert candidate.aligned_depth_m.dtype == np.float32
    assert candidate.source_valid_pixels == reference.source_valid_pixels
    assert candidate.projected_points == reference.projected_points
    assert candidate.aligned_valid_pixels == reference.aligned_valid_pixels
    assert candidate.z_buffer_collisions == reference.z_buffer_collisions
    assert candidate.depth_scale_m_per_unit == reference.depth_scale_m_per_unit
    assert candidate.source_frame == reference.source_frame
    assert candidate.target_frame == reference.target_frame
    reference_mask = np.isfinite(reference.aligned_depth_m)
    candidate_mask = np.isfinite(candidate.aligned_depth_m)
    np.testing.assert_array_equal(candidate_mask, reference_mask)
    np.testing.assert_allclose(
        candidate.aligned_depth_m[candidate_mask],
        reference.aligned_depth_m[reference_mask],
        atol=1e-5,
        rtol=2e-6,
    )


def test_default_backend_remains_numpy_reference() -> None:
    registrar = _registrar()
    assert registrar.requested_backend == "numpy"
    assert registrar.backend_name == "numpy_reference"
    assert not registrar.fallback_active
    assert registrar.fallback_count == 0


def test_missing_cuda_library_sticky_fallback_is_visible(tmp_path: Path) -> None:
    missing = tmp_path / "libdoes-not-exist.so"
    fallback = _registrar(
        backend="cuda", allow_fallback=True, library_path=str(missing)
    )
    reference = _registrar()
    native = np.full((6, 8), 1200, dtype=np.uint16)

    assert fallback.requested_backend == "cuda"
    assert fallback.backend_name == "numpy_reference"
    assert fallback.fallback_active
    assert fallback.fallback_count == 1
    assert "was not found" in fallback.last_backend_error
    _assert_equivalent(
        reference.register(native, 0.001),
        fallback.register(native, 0.001),
    )
    assert fallback.fallback_count == 1


def test_missing_cuda_library_is_fail_closed_without_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="initialization failed"):
        _registrar(
            backend="cuda",
            allow_fallback=False,
            library_path=str(tmp_path / "missing.so"),
        )


def test_wrong_shared_library_uses_visible_sticky_fallback() -> None:
    wrong_library = Path("/lib/x86_64-linux-gnu/libm.so.6")
    if not wrong_library.is_file():
        pytest.skip("known non-registration shared library is unavailable")
    fallback = _registrar(
        backend="cuda",
        allow_fallback=True,
        library_path=str(wrong_library),
    )
    assert fallback.backend_name == "numpy_reference"
    assert fallback.fallback_active
    assert fallback.fallback_count == 1
    assert "missing required ABI symbols" in fallback.last_backend_error


@pytest.mark.parametrize(
    "distortion",
    [
        [],
        [0.01, -0.004, 0.0003, -0.0002],
        [0.01, -0.004, 0.0003, -0.0002, 0.0007],
        [0.01, -0.004, 0.0003, -0.0002, 0.0007, 0.001, -0.0005, 0.0002],
        [
            0.01,
            -0.004,
            0.0003,
            -0.0002,
            0.0007,
            0.001,
            -0.0005,
            0.0002,
            0.0001,
            -0.00002,
            0.00008,
            -0.00001,
        ],
    ],
)
def test_cuda_matches_numpy_distortion_and_collision(distortion) -> None:
    library = _installed_cuda_library()
    if library is None:
        pytest.skip("CUDA registration library is not installed")
    reference = _registrar(distortion=distortion)
    candidate = _registrar(
        backend="cuda", library_path=library, distortion=distortion
    )
    rng = np.random.default_rng(20260826)
    native = rng.integers(0, 4000, size=(6, 8), dtype=np.uint16)
    native[0, :4] = 0
    native[1, 0] = 50
    native[1, 1] = 10_000

    reference_result = reference.register(native, 0.001, 0.05, 10.0)
    cuda_result = candidate.register(native, 0.001, 0.05, 10.0)
    _assert_equivalent(reference_result, cuda_result)
    assert candidate.backend_name == "cuda_cabi_v1"
    assert not candidate.fallback_active
    assert candidate.last_gpu_ms >= 0.0


def test_cuda_matches_numpy_float_invalid_limits_and_noncontiguous() -> None:
    library = _installed_cuda_library()
    if library is None:
        pytest.skip("CUDA registration library is not installed")
    reference = _registrar()
    candidate = _registrar(backend="cuda", library_path=library)
    backing = np.full((6, 16), 1.25, dtype=np.float64)
    native = backing[:, ::2]
    native[0, 0] = np.nan
    native[0, 1] = np.inf
    native[0, 2] = -1.0
    native[0, 3] = 0.0
    native[0, 4] = 0.05
    native[0, 5] = 10.0

    _assert_equivalent(
        reference.register(native, 1.0, 0.05, 10.0),
        candidate.register(native, 1.0, 0.05, 10.0),
    )


def test_cuda_repeated_output_is_deterministic() -> None:
    library = _installed_cuda_library()
    if library is None:
        pytest.skip("CUDA registration library is not installed")
    candidate = _registrar(backend="cuda", library_path=library)
    native = np.arange(48, dtype=np.uint16).reshape(6, 8) * 17 + 50
    first = candidate.register(native, 0.001, 0.05, 10.0)
    for _ in range(10):
        repeated = candidate.register(native, 0.001, 0.05, 10.0)
        np.testing.assert_array_equal(
            repeated.aligned_depth_m, first.aligned_depth_m
        )
        assert repeated.projected_points == first.projected_points
        assert repeated.z_buffer_collisions == first.z_buffer_collisions


def test_cuda_close_is_idempotent_and_blocks_later_registration() -> None:
    library = _installed_cuda_library()
    if library is None:
        pytest.skip("CUDA registration library is not installed")
    candidate = _registrar(backend="cuda", library_path=library)
    native = np.full((6, 8), 1000, dtype=np.uint16)
    candidate.close()
    candidate.close()
    with pytest.raises(RuntimeError, match="context is closed"):
        candidate.register(native, 0.001)
