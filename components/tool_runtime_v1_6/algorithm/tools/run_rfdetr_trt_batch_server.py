#!/usr/bin/env python3
"""Serve one dynamic-batch RF-DETR TensorRT engine to local camera workers."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import json
from multiprocessing.connection import Connection, Listener
import os
from pathlib import Path
from types import SimpleNamespace
import signal
import threading
import time
from typing import Any

import numpy as np

from pnu_surgical_tool.trt_batch import AUTHKEY, PROTOCOL_SCHEMA, pack_prediction
from pnu_surgical_tool.trt_engine import (
    TensorRtPlanRunner,
    load_engine_metadata,
    validate_engine_metadata,
)
from pnu_surgical_tool.trt_fast_predict import predict_thresholded_masks


MODEL_CLASSES = {
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
}


@dataclass
class PendingRequest:
    """One client request waiting to be included in a GPU batch."""

    payload: dict[str, Any]
    received_monotonic: float
    completed: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


def arguments() -> argparse.Namespace:
    """Parse server configuration."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES), default="xlarge")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--maximum-batch-size", type=int, default=3)
    parser.add_argument("--batch-window-ms", type=float, default=0.0)
    return parser.parse_args()


class TrtBatchServer:
    """Batch requests from independent latest-frame camera workers."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.engine_path = args.engine.expanduser().resolve()
        self.checkpoint_path = args.checkpoint.expanduser().resolve()
        self.model_size = str(args.model_size)
        self.socket_path = args.socket.expanduser().resolve()
        self.stats_path = args.stats.expanduser().resolve()
        self.maximum_batch_size = int(args.maximum_batch_size)
        self.batch_window_sec = float(args.batch_window_ms) / 1000.0
        if self.maximum_batch_size < 1:
            raise ValueError("maximum-batch-size must be positive")
        if self.batch_window_sec < 0.0:
            raise ValueError("batch-window-ms must be non-negative")
        self._condition = threading.Condition()
        self._queue: deque[PendingRequest] = deque()
        self._stopping = False
        self._listener: Listener | None = None
        self._client_threads: list[threading.Thread] = []
        self._worker: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._request_total = 0
        self._batch_total = 0
        self._failed_batch_total = 0
        self._batch_size_counts: Counter[int] = Counter()
        self._batched_camera_frame_total = 0
        self._inference_ms_total = 0.0
        self._last_batch: dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        import rfdetr
        import tensorrt
        import torch
        import torch_tensorrt

        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT server requires CUDA")
        metadata = load_engine_metadata(self.engine_path)
        validate_engine_metadata(
            metadata,
            engine_path=self.engine_path,
            checkpoint_path=self.checkpoint_path,
            model_size=self.model_size,
            required_max_batch=self.maximum_batch_size,
            torch_version=torch.__version__,
            torch_tensorrt_version=torch_tensorrt.__version__,
            tensorrt_version=tensorrt.__version__,
            compute_capability=torch.cuda.get_device_capability(0),
        )
        model_class = getattr(rfdetr, MODEL_CLASSES[self.model_size])
        # Build only the lightweight preprocessing/postprocessing context on
        # CPU. The duplicate PyTorch network is released after the TRT module
        # is attached, so runtime VRAM contains one shared engine.
        model = model_class.from_checkpoint(
            str(self.checkpoint_path), device="cpu"
        )
        engine = TensorRtPlanRunner(
            self.engine_path,
            input_name=str(metadata["input_name"]),
            output_names=tuple(metadata["output_names"]),
        )
        model.model.inference_model = engine
        model.model.device = torch.device("cuda")
        model.model.model = None
        model._is_optimized_for_inference = True
        model._optimized_has_been_compiled = False
        model._optimized_batch_size = None
        model._optimized_resolution = int(metadata["resolution"])
        model._optimized_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
        }[str(metadata["input_dtype"])]
        model._optimized_inplace = True
        self._model = model
        self._engine_runner = engine
        self._torch = torch
        self._metadata = metadata
        self._identity = {
            "model_size": self.model_size,
            "resolution": int(metadata["resolution"]),
            "maximum_batch_size": self.maximum_batch_size,
            "engine_sha256": metadata["engine_sha256"],
            "tensorrt_engine_count": metadata["tensorrt_engine_count"],
        }

    def _response(
        self, request: dict[str, Any], *, ok: bool, **fields: Any
    ) -> dict[str, Any]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "request_id": request.get("request_id"),
            "ok": bool(ok),
            **fields,
        }

    def _handle_client(self, connection: Connection) -> None:
        try:
            while not self._stopping:
                request = connection.recv()
                if not isinstance(request, dict):
                    raise ValueError("request must be a dict")
                if request.get("schema") != PROTOCOL_SCHEMA:
                    raise ValueError("protocol schema mismatch")
                operation = request.get("operation")
                if operation == "ping":
                    connection.send(self._response(
                        request, ok=True, **self._identity
                    ))
                    continue
                if operation != "predict":
                    connection.send(self._response(
                        request, ok=False, error="unsupported operation"
                    ))
                    continue
                pending = PendingRequest(request, time.monotonic())
                with self._condition:
                    self._queue.append(pending)
                    self._condition.notify()
                pending.completed.wait()
                if pending.response is None:
                    raise RuntimeError("batch worker completed without response")
                connection.send(pending.response)
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            pass
        except Exception as error:
            try:
                connection.send({
                    "schema": PROTOCOL_SCHEMA,
                    "request_id": None,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                })
            except Exception:
                pass
        finally:
            connection.close()

    def _next_batch(self) -> list[PendingRequest] | None:
        with self._condition:
            while not self._queue and not self._stopping:
                self._condition.wait()
            if self._stopping:
                return None
            batch = [self._queue.popleft()]
            if self.batch_window_sec == 0.0:
                while self._queue and len(batch) < self.maximum_batch_size:
                    batch.append(self._queue.popleft())
                return batch
            deadline = time.monotonic() + self.batch_window_sec
            while len(batch) < self.maximum_batch_size:
                while self._queue and len(batch) < self.maximum_batch_size:
                    batch.append(self._queue.popleft())
                if len(batch) >= self.maximum_batch_size:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            return batch

    @staticmethod
    def _filter_prediction(prediction: Any, threshold: float) -> Any:
        confidence = np.asarray(prediction.confidence, dtype=np.float32)
        keep = confidence > float(threshold)
        return SimpleNamespace(
            xyxy=np.asarray(prediction.xyxy)[keep],
            class_id=np.asarray(prediction.class_id)[keep],
            confidence=confidence[keep],
            mask=np.asarray(prediction.mask, dtype=bool)[keep],
        )

    def _run_batch(self, batch: list[PendingRequest]) -> None:
        batch_started = time.monotonic()
        try:
            images = []
            thresholds = []
            cameras = []
            for pending in batch:
                request = pending.payload
                image = np.asarray(request["image"])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("request image must be uint8 HxWx3")
                threshold = float(request["confidence_threshold"])
                if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError("confidence_threshold must be in [0, 1]")
                camera = str(request["camera_key"]).strip()
                if not camera:
                    raise ValueError("camera_key must not be empty")
                images.append(image)
                thresholds.append(threshold)
                cameras.append(camera)
            minimum_threshold = min(thresholds)
            self._torch.cuda.synchronize()
            inference_started = time.perf_counter()
            predictions, fast_predict_diagnostics = predict_thresholded_masks(
                self._model,
                self._engine_runner,
                images,
                minimum_threshold,
            )
            self._torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if len(batch) == 1:
                predictions = [predictions]
            if len(predictions) != len(batch):
                raise RuntimeError(
                    f"prediction count {len(predictions)} != batch {len(batch)}"
                )
            for index, (pending, prediction, threshold) in enumerate(
                zip(batch, predictions, thresholds, strict=True)
            ):
                response_started = time.perf_counter()
                filtered = self._filter_prediction(prediction, threshold)
                packed_prediction = pack_prediction(filtered)
                response_build_ms = (
                    time.perf_counter() - response_started
                ) * 1000.0
                runner_diagnostics = dict(
                    self._engine_runner.last_runtime_diagnostics
                )
                runner_wall_ms = float(
                    runner_diagnostics.get("runner_wall_ms", 0.0)
                )
                pending.response = self._response(
                    pending.payload,
                    ok=True,
                    prediction=packed_prediction,
                    diagnostics={
                        "backend": "tensorrt_shared_dynamic_batch",
                        "batch_size": len(batch),
                        "batch_index": index,
                        "server_inference_ms": inference_ms,
                        "server_model_wrapper_ms": max(
                            0.0, inference_ms - runner_wall_ms
                        ),
                        "server_response_build_ms": response_build_ms,
                        "server_queue_wait_ms": (
                            batch_started - pending.received_monotonic
                        ) * 1000.0,
                        "engine_sha256": self._metadata["engine_sha256"],
                        **fast_predict_diagnostics,
                        **runner_diagnostics,
                    },
                )
            with self._stats_lock:
                self._request_total += len(batch)
                self._batch_total += 1
                self._batch_size_counts[len(batch)] += 1
                self._batched_camera_frame_total += len(batch)
                self._inference_ms_total += inference_ms
                self._last_batch = {
                    "batch_size": len(batch),
                    "cameras": cameras,
                    "inference_ms": inference_ms,
                    "completed_monotonic": time.monotonic(),
                }
        except Exception as error:
            with self._stats_lock:
                self._failed_batch_total += 1
            for pending in batch:
                pending.response = self._response(
                    pending.payload,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
        finally:
            for pending in batch:
                pending.completed.set()
            self._write_stats()

    def _worker_loop(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            self._run_batch(batch)

    def _write_stats(self) -> None:
        with self._stats_lock:
            payload = {
                "schema": "pnu.rfdetr_trt_batch_server_stats.v1",
                **self._identity,
                "socket": str(self.socket_path),
                "batch_window_ms": self.batch_window_sec * 1000.0,
                "request_total": self._request_total,
                "batch_total": self._batch_total,
                "failed_batch_total": self._failed_batch_total,
                "batch_size_counts": {
                    str(key): value
                    for key, value in sorted(self._batch_size_counts.items())
                },
                "mean_batch_size": (
                    self._batched_camera_frame_total / self._batch_total
                    if self._batch_total
                    else 0.0
                ),
                "mean_inference_ms": (
                    self._inference_ms_total / self._batch_total
                    if self._batch_total
                    else 0.0
                ),
                "last_batch": self._last_batch,
            }
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.stats_path.with_name(
            f".{self.stats_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(temporary, self.stats_path)

    def run(self) -> None:
        """Load the socket listener and serve until terminated."""
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        old_umask = os.umask(0o077)
        try:
            self._listener = Listener(
                str(self.socket_path), family="AF_UNIX", authkey=AUTHKEY
            )
        finally:
            os.umask(old_umask)
        self.socket_path.chmod(0o600)
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="rfdetr-trt-gpu-batcher",
            daemon=True,
        )
        self._worker.start()
        self._write_stats()
        print(
            "RF-DETR TensorRT batch server ready: "
            f"socket={self.socket_path} model={self.model_size} "
            f"batch=1..{self.maximum_batch_size} "
            f"window_ms={self.batch_window_sec * 1000.0:.3f}",
            flush=True,
        )
        while not self._stopping:
            connection = self._listener.accept()
            thread = threading.Thread(
                target=self._handle_client,
                args=(connection,),
                name="rfdetr-trt-client",
                daemon=True,
            )
            self._client_threads.append(thread)
            thread.start()

    def stop(self) -> None:
        """Wake the batch worker and release the local socket."""
        with self._condition:
            self._stopping = True
            while self._queue:
                pending = self._queue.popleft()
                pending.response = self._response(
                    pending.payload,
                    ok=False,
                    error="TensorRT batch server is stopping",
                )
                pending.completed.set()
            self._condition.notify_all()
        if self._listener is not None:
            self._listener.close()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        self.socket_path.unlink(missing_ok=True)


def main() -> int:
    """Run the TensorRT batch server."""
    server = TrtBatchServer(arguments())

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
