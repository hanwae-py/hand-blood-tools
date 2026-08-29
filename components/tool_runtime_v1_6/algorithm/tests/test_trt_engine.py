from __future__ import annotations

import pytest

from pnu_surgical_tool.trt_engine import (
    ENGINE_METADATA_SCHEMA,
    sha256_file,
    validate_engine_metadata,
)


def _metadata(engine, checkpoint):
    return {
        "schema": ENGINE_METADATA_SCHEMA,
        "engine_format": "tensorrt_plan",
        "model_size": "xlarge",
        "checkpoint_sha256": sha256_file(checkpoint),
        "engine_sha256": sha256_file(engine),
        "maximum_batch_size": 3,
        "torch_version": "2.7.0+cu118",
        "torch_tensorrt_version": "2.7.0",
        "tensorrt_version": "10.9.0.34",
        "compute_capability": [8, 6],
        "tensorrt_engine_count": 1,
    }


def _validate(metadata, engine, checkpoint):
    validate_engine_metadata(
        metadata,
        engine_path=engine,
        checkpoint_path=checkpoint,
        model_size="xlarge",
        required_max_batch=3,
        torch_version="2.7.0+cu118",
        torch_tensorrt_version="2.7.0",
        tensorrt_version="10.9.0.34",
        compute_capability=(8, 6),
    )


def test_engine_metadata_accepts_exact_runtime_identity(tmp_path):
    engine = tmp_path / "engine.ts"
    checkpoint = tmp_path / "checkpoint.pth"
    engine.write_bytes(b"engine")
    checkpoint.write_bytes(b"checkpoint")

    _validate(_metadata(engine, checkpoint), engine, checkpoint)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "torch_version"])
def test_engine_metadata_rejects_stale_hash_or_runtime_version(tmp_path, field):
    engine = tmp_path / "engine.ts"
    checkpoint = tmp_path / "checkpoint.pth"
    engine.write_bytes(b"engine")
    checkpoint.write_bytes(b"checkpoint")
    metadata = _metadata(engine, checkpoint)
    metadata[field] = "stale"

    with pytest.raises(RuntimeError, match=field):
        _validate(metadata, engine, checkpoint)
