"""TensorRT engine metadata and compatibility checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any


ENGINE_METADATA_SCHEMA = "pnu.rfdetr_tensorrt_engine.v1"


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path_for_engine(engine_path: str | Path) -> Path:
    """Return the sidecar metadata path for an engine TorchScript file."""
    path = Path(engine_path)
    return Path(f"{path}.json")


def load_engine_metadata(engine_path: str | Path) -> dict[str, Any]:
    """Load the JSON sidecar belonging to ``engine_path``."""
    metadata_path = metadata_path_for_engine(engine_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("TensorRT engine metadata must be a JSON object")
    return payload


def validate_engine_metadata(
    metadata: dict[str, Any],
    *,
    engine_path: str | Path,
    checkpoint_path: str | Path,
    model_size: str,
    required_max_batch: int,
    torch_version: str,
    torch_tensorrt_version: str,
    tensorrt_version: str,
    compute_capability: tuple[int, int],
) -> None:
    """Reject stale or binary-incompatible TensorRT engine artifacts."""
    expected = {
        "schema": ENGINE_METADATA_SCHEMA,
        "engine_format": "tensorrt_plan",
        "model_size": str(model_size),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "engine_sha256": sha256_file(engine_path),
        "torch_version": str(torch_version),
        "torch_tensorrt_version": str(torch_tensorrt_version),
        "tensorrt_version": str(tensorrt_version),
        "compute_capability": list(compute_capability),
    }
    mismatches = [
        f"{key}: metadata={metadata.get(key)!r}, runtime={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    maximum_batch = int(metadata.get("maximum_batch_size", 0))
    if maximum_batch < int(required_max_batch):
        mismatches.append(
            f"maximum_batch_size: metadata={maximum_batch}, "
            f"required={required_max_batch}"
        )
    if int(metadata.get("tensorrt_engine_count", 0)) < 1:
        mismatches.append("tensorrt_engine_count must be at least one")
    if mismatches:
        raise RuntimeError(
            "TensorRT engine metadata is incompatible: " + "; ".join(mismatches)
        )


class TensorRtPlanRunner:
    """Execute one dynamic-batch TensorRT plan directly on PyTorch CUDA tensors."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        input_name: str,
        output_names: list[str] | tuple[str, ...],
    ) -> None:
        import numpy as np
        import tensorrt
        import torch

        self._torch = torch
        self._tensorrt = tensorrt
        logger = tensorrt.Logger(tensorrt.Logger.WARNING)
        self._runtime = tensorrt.Runtime(logger)
        serialized = Path(engine_path).read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(serialized)
        if self._engine is None:
            raise RuntimeError(f"could not deserialize TensorRT plan: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("could not create TensorRT execution context")
        io_names = tuple(
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        )
        self.input_name = str(input_name)
        self.output_names = tuple(str(name) for name in output_names)
        if self.input_name not in io_names:
            raise RuntimeError(
                f"TensorRT input {self.input_name!r} not found in {io_names}"
            )
        if any(name not in io_names for name in self.output_names):
            raise RuntimeError(
                f"TensorRT outputs {self.output_names!r} not found in {io_names}"
            )
        self._torch_dtypes = {
            name: torch.from_numpy(
                np.empty((), dtype=tensorrt.nptype(self._engine.get_tensor_dtype(name)))
            ).dtype
            for name in io_names
        }
        # TensorRT warns and inserts extra synchronization when enqueueV3 uses
        # CUDA's legacy default stream. One private stream is sufficient
        # because this runner owns one execution context and is called by one
        # batch worker at a time.
        self._stream = torch.cuda.Stream()
        self._input_started = torch.cuda.Event(enable_timing=True)
        self._input_completed = torch.cuda.Event(enable_timing=True)
        self._execution_started = torch.cuda.Event(enable_timing=True)
        self._execution_completed = torch.cuda.Event(enable_timing=True)
        self.last_runtime_diagnostics: dict[str, float] = {}

    def __call__(self, input_tensor: Any) -> tuple[Any, ...]:
        torch = self._torch
        runner_started = time.perf_counter()
        upstream_stream = torch.cuda.current_stream()
        self._stream.wait_stream(upstream_stream)
        with torch.cuda.stream(self._stream):
            self._input_started.record(self._stream)
            frame_batch = input_tensor.to(
                device="cuda",
                dtype=self._torch_dtypes[self.input_name],
            ).contiguous()
            self._input_completed.record(self._stream)
            if not self._context.set_input_shape(
                self.input_name, tuple(frame_batch.shape)
            ):
                raise RuntimeError(
                    f"TensorRT rejected input shape {tuple(frame_batch.shape)}"
                )
            unresolved = self._context.infer_shapes()
            if unresolved:
                raise RuntimeError(
                    f"TensorRT unresolved dynamic tensors: {unresolved}"
                )
            outputs = []
            for name in self.output_names:
                shape = tuple(self._context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(
                        f"TensorRT output {name!r} has unresolved shape {shape}"
                    )
                outputs.append(
                    torch.empty(
                        shape,
                        device="cuda",
                        dtype=self._torch_dtypes[name],
                    )
                )
            self._context.set_tensor_address(
                self.input_name, int(frame_batch.data_ptr())
            )
            for name, output in zip(self.output_names, outputs, strict=True):
                self._context.set_tensor_address(name, int(output.data_ptr()))
            self._execution_started.record(self._stream)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            self._execution_completed.record(self._stream)
        # TensorRT uses external pointers unknown to the PyTorch allocator.
        # Synchronizing here prevents input storage reuse before execution ends.
        self._stream.synchronize()
        runner_wall_ms = (time.perf_counter() - runner_started) * 1000.0
        input_gpu_ms = float(
            self._input_started.elapsed_time(self._input_completed)
        )
        execution_gpu_ms = float(
            self._execution_started.elapsed_time(self._execution_completed)
        )
        self.last_runtime_diagnostics = {
            "runner_wall_ms": runner_wall_ms,
            "input_prepare_gpu_ms": input_gpu_ms,
            "trt_execute_gpu_ms": execution_gpu_ms,
            "runner_unattributed_wall_ms": max(
                0.0, runner_wall_ms - input_gpu_ms - execution_gpu_ms
            ),
        }
        return tuple(outputs)
