#!/usr/bin/env python3
"""Compile a fine-tuned RF-DETR checkpoint into a dynamic-batch TRT plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


MODEL_CLASSES = {
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-batch-size", type=int, default=1)
    parser.add_argument("--optimum-batch-size", type=int, default=3)
    parser.add_argument("--maximum-batch-size", type=int, default=3)
    parser.add_argument("--workspace-gib", type=float, default=3.0)
    parser.add_argument(
        "--allow-pytorch-fallback",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The ONNX parser always builds one "
            "full TensorRT plan and never embeds PyTorch fallback regions."
        ),
    )
    parser.add_argument("--skip-verification", action="store_true")
    return parser.parse_args()


def _flatten_tensors(value: Any) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[Any] = []
        for key in sorted(value):
            result.extend(_flatten_tensors(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    raise TypeError(f"unsupported output type: {type(value)!r}")


def _build_plan(
    onnx_path: Path,
    *,
    resolution: int,
    minimum_batch_size: int,
    optimum_batch_size: int,
    maximum_batch_size: int,
    workspace_bytes: int,
) -> tuple[bytes, str, list[str], int]:
    import tensorrt

    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(logger)
    network = builder.create_network(
        1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = tensorrt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed: " + " | ".join(errors))
    if network.num_inputs != 1:
        raise RuntimeError(f"expected one RF-DETR input, got {network.num_inputs}")
    input_tensor = network.get_input(0)
    input_name = str(input_tensor.name)
    output_names = [
        str(network.get_output(index).name)
        for index in range(network.num_outputs)
    ]
    if output_names != ["dets", "labels", "masks"]:
        raise RuntimeError(f"unexpected RF-DETR outputs: {output_names}")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (minimum_batch_size, 3, resolution, resolution),
        (optimum_batch_size, 3, resolution, resolution),
        (maximum_batch_size, 3, resolution, resolution),
    )
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(
        tensorrt.MemoryPoolType.WORKSPACE, int(workspace_bytes)
    )
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("GPU does not advertise fast FP16 TensorRT support")
    config.set_flag(tensorrt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    layer_count = int(network.num_layers)
    return bytes(serialized), input_name, output_names, layer_count


def _verify(
    reference: Any,
    runner: Any,
    *,
    resolution: int,
    batch_sizes: list[int],
) -> list[dict[str, Any]]:
    import torch

    reports: list[dict[str, Any]] = []
    generator = torch.Generator(device="cuda").manual_seed(5020)
    for batch_size in batch_sizes:
        sample = torch.randn(
            batch_size,
            3,
            resolution,
            resolution,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        torch.cuda.synchronize()
        with torch.inference_mode():
            expected = _flatten_tensors(reference(sample))
            actual = _flatten_tensors(runner(sample))
        torch.cuda.synchronize()
        if len(expected) != len(actual):
            raise RuntimeError(
                f"output count mismatch at batch {batch_size}: "
                f"{len(expected)} != {len(actual)}"
            )
        tensor_reports = []
        for index, (expected_tensor, actual_tensor) in enumerate(
            zip(expected, actual, strict=True)
        ):
            if expected_tensor.shape != actual_tensor.shape:
                raise RuntimeError(
                    f"output {index} shape mismatch at batch {batch_size}: "
                    f"{tuple(expected_tensor.shape)} != {tuple(actual_tensor.shape)}"
                )
            difference = (expected_tensor.float() - actual_tensor.float()).abs()
            if not torch.all(torch.isfinite(actual_tensor)):
                raise RuntimeError(
                    f"output {index} contains non-finite values at batch {batch_size}"
                )
            tensor_reports.append(
                {
                    "index": index,
                    "shape": list(actual_tensor.shape),
                    "dtype": str(actual_tensor.dtype),
                    "max_abs_error": (
                        float(difference.max()) if difference.numel() else 0.0
                    ),
                    "mean_abs_error": (
                        float(difference.mean()) if difference.numel() else 0.0
                    ),
                }
            )
        reports.append({"batch_size": batch_size, "outputs": tensor_reports})
    return reports


def main() -> int:
    args = _arguments()
    if not (
        1
        <= args.minimum_batch_size
        <= args.optimum_batch_size
        <= args.maximum_batch_size
    ):
        raise ValueError("batch sizes must satisfy 1 <= min <= optimum <= max")
    if args.workspace_gib <= 0.0:
        raise ValueError("workspace-gib must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import onnx
    import rfdetr
    import tensorrt
    import torch
    import torch_tensorrt

    from pnu_surgical_tool.trt_engine import (
        ENGINE_METADATA_SCHEMA,
        TensorRtPlanRunner,
        metadata_path_for_engine,
        sha256_file,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT compilation requires a CUDA GPU")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_class = getattr(rfdetr, MODEL_CLASSES[args.model_size])
    started = time.perf_counter()
    model = model_class.from_checkpoint(str(checkpoint), device="cuda")
    resolution = int(model.model.resolution)

    with tempfile.TemporaryDirectory(
        prefix="rfdetr-onnx-", dir=output.parent
    ) as temporary_directory:
        onnx_path = Path(
            model.export(
                output_dir=temporary_directory,
                shape=(resolution, resolution),
                batch_size=args.optimum_batch_size,
                dynamic_batch=True,
                opset_version=17,
                verbose=False,
                format="onnx",
            )
        )
        serialized, input_name, output_names, layer_count = _build_plan(
            onnx_path,
            resolution=resolution,
            minimum_batch_size=args.minimum_batch_size,
            optimum_batch_size=args.optimum_batch_size,
            maximum_batch_size=args.maximum_batch_size,
            workspace_bytes=int(args.workspace_gib * 1024**3),
        )

    temporary_engine = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_engine.write_bytes(serialized)
    del serialized
    gc.collect()
    torch.cuda.empty_cache()

    verification: list[dict[str, Any]] = []
    if not args.skip_verification:
        reference = model.model.model.eval()
        reference.export()
        runner = TensorRtPlanRunner(
            temporary_engine,
            input_name=input_name,
            output_names=output_names,
        )
        verification = _verify(
            reference,
            runner,
            resolution=resolution,
            batch_sizes=list(
                range(args.minimum_batch_size, args.maximum_batch_size + 1)
            ),
        )
        del runner
    os.replace(temporary_engine, output)

    capability = torch.cuda.get_device_capability(0)
    metadata = {
        "schema": ENGINE_METADATA_SCHEMA,
        "engine_format": "tensorrt_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "engine_path": str(output),
        "engine_sha256": sha256_file(output),
        "engine_bytes": output.stat().st_size,
        "model_size": args.model_size,
        "model_class": MODEL_CLASSES[args.model_size],
        "resolution": resolution,
        "input_name": input_name,
        "output_names": output_names,
        "minimum_batch_size": args.minimum_batch_size,
        "optimum_batch_size": args.optimum_batch_size,
        "maximum_batch_size": args.maximum_batch_size,
        "precision": "fp16",
        "input_dtype": "float32",
        "compilation_frontend": "rfdetr_dynamic_onnx_tensorrt_parser",
        "require_full_compilation": True,
        "pytorch_fallback_regions": 0,
        "tensorrt_engine_count": 1,
        "network_layer_count": layer_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_tensorrt_version": torch_tensorrt.__version__,
        "tensorrt_version": tensorrt.__version__,
        "onnx_version": onnx.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "workspace_gib": args.workspace_gib,
        "compile_elapsed_sec": time.perf_counter() - started,
        "verification": verification,
    }
    metadata_path = metadata_path_for_engine(output)
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"TensorRT compilation failed: {error}", file=sys.stderr)
        raise
