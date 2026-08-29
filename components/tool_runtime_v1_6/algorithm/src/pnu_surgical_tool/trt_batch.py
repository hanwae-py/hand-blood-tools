"""Local IPC contract for shared, batched RF-DETR TensorRT inference."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Client, Connection
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


PROTOCOL_SCHEMA = "pnu.rfdetr_trt_batch_ipc.v1"
AUTHKEY = b"pnu-rfdetr-trt-v1"


@dataclass(frozen=True)
class RemotePrediction:
    """Minimal supervision-compatible prediction returned by the server."""

    xyxy: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray
    mask: np.ndarray


def pack_prediction(prediction: Any) -> dict[str, Any]:
    """Pack boolean masks while retaining exact boxes/classes/confidences."""
    masks = np.asarray(getattr(prediction, "mask", None), dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"segmentation masks must be NxHxW, got {masks.shape}")
    return {
        "xyxy": np.asarray(prediction.xyxy, dtype=np.float32),
        "class_id": np.asarray(prediction.class_id, dtype=np.int32),
        "confidence": np.asarray(prediction.confidence, dtype=np.float32),
        "mask_shape": tuple(int(value) for value in masks.shape),
        "mask_bits": np.packbits(
            masks.reshape(-1), bitorder="little"
        ).tobytes(),
    }


def unpack_prediction(payload: dict[str, Any]) -> RemotePrediction:
    """Reconstruct one exact boolean-mask prediction from an IPC payload."""
    shape = tuple(int(value) for value in payload["mask_shape"])
    if len(shape) != 3 or any(value < 0 for value in shape):
        raise ValueError(f"invalid mask_shape: {shape}")
    count = int(np.prod(shape, dtype=np.int64))
    bits = np.frombuffer(payload["mask_bits"], dtype=np.uint8)
    masks = np.unpackbits(bits, count=count, bitorder="little").reshape(shape)
    return RemotePrediction(
        xyxy=np.asarray(payload["xyxy"], dtype=np.float32),
        class_id=np.asarray(payload["class_id"], dtype=np.int32),
        confidence=np.asarray(payload["confidence"], dtype=np.float32),
        mask=masks.astype(bool, copy=False),
    )


class BatchInferenceClient:
    """Persistent synchronous client used by one latest-frame ROS worker."""

    def __init__(
        self,
        socket_path: str | Path,
        camera_key: str,
        *,
        timeout_sec: float = 10.0,
        expected_model_size: str = "xlarge",
    ) -> None:
        self.socket_path = str(Path(socket_path))
        self.camera_key = str(camera_key).strip()
        self.timeout_sec = float(timeout_sec)
        self.expected_model_size = str(expected_model_size)
        if not self.socket_path or not self.camera_key:
            raise ValueError("socket_path and camera_key are required")
        if self.timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        self._connection: Connection | None = None
        self._lock = threading.Lock()
        self._request_id = 0
        self.last_diagnostics: dict[str, Any] = {}

    def _close_locked(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def close(self) -> None:
        """Close the persistent local connection."""
        with self._lock:
            self._close_locked()

    def _connect_locked(self) -> Connection:
        if self._connection is not None:
            return self._connection
        deadline = time.monotonic() + self.timeout_sec
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self._connection = Client(
                    self.socket_path, family="AF_UNIX", authkey=AUTHKEY
                )
                return self._connection
            except (FileNotFoundError, ConnectionRefusedError, OSError) as error:
                last_error = error
                time.sleep(0.05)
        raise TimeoutError(
            f"TensorRT batch server unavailable at {self.socket_path}: "
            f"{last_error}"
        )

    def _round_trip_locked(self, request: dict[str, Any]) -> dict[str, Any]:
        connection = self._connect_locked()
        try:
            connection.send(request)
            if not connection.poll(self.timeout_sec):
                raise TimeoutError(
                    f"TensorRT request timed out after {self.timeout_sec:.3f}s"
                )
            response = connection.recv()
        except (EOFError, BrokenPipeError, ConnectionError, OSError, TimeoutError):
            self._close_locked()
            raise
        if not isinstance(response, dict):
            raise RuntimeError("TensorRT server returned a non-dict response")
        if response.get("schema") != PROTOCOL_SCHEMA:
            raise RuntimeError("TensorRT server protocol schema mismatch")
        if response.get("request_id") != request.get("request_id"):
            raise RuntimeError("TensorRT server response request_id mismatch")
        if not response.get("ok", False):
            raise RuntimeError(
                f"TensorRT server error: {response.get('error', 'unknown')}"
            )
        return response

    def ping(self) -> dict[str, Any]:
        """Validate server availability and its shared model identity."""
        with self._lock:
            self._request_id += 1
            response = self._round_trip_locked({
                "schema": PROTOCOL_SCHEMA,
                "operation": "ping",
                "request_id": self._request_id,
                "camera_key": self.camera_key,
            })
        model_size = str(response.get("model_size", ""))
        if model_size != self.expected_model_size:
            raise RuntimeError(
                f"TensorRT server model_size={model_size!r} != expected "
                f"{self.expected_model_size!r}"
            )
        return response

    def predict(
        self, image: np.ndarray, confidence_threshold: float
    ) -> RemotePrediction:
        """Submit one RGB frame and wait for its batched prediction."""
        frame = np.ascontiguousarray(image, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("TensorRT IPC image must be uint8 HxWx3")
        with self._lock:
            self._request_id += 1
            started = time.perf_counter()
            response = self._round_trip_locked({
                "schema": PROTOCOL_SCHEMA,
                "operation": "predict",
                "request_id": self._request_id,
                "camera_key": self.camera_key,
                "confidence_threshold": float(confidence_threshold),
                "image": frame,
            })
            round_trip_ms = (time.perf_counter() - started) * 1000.0
        self.last_diagnostics = dict(response.get("diagnostics", {}))
        self.last_diagnostics["ipc_round_trip_ms"] = round_trip_ms
        return unpack_prediction(response["prediction"])
