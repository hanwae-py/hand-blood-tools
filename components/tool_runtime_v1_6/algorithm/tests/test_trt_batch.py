from __future__ import annotations

from collections import deque
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np

from pnu_surgical_tool.trt_batch import (
    AUTHKEY,
    PROTOCOL_SCHEMA,
    BatchInferenceClient,
    pack_prediction,
    unpack_prediction,
)


def _prediction(count: int = 2, height: int = 7, width: int = 9):
    masks = np.zeros((count, height, width), dtype=bool)
    if count:
        masks[0, 1:4, 2:8] = True
    return SimpleNamespace(
        xyxy=np.arange(count * 4, dtype=np.float32).reshape(count, 4),
        class_id=np.arange(count, dtype=np.int32),
        confidence=np.linspace(0.6, 0.9, count, dtype=np.float32),
        mask=masks,
    )


def test_prediction_mask_pack_round_trip_is_exact():
    for count in (0, 2):
        source = _prediction(count=count)
        restored = unpack_prediction(pack_prediction(source))

        np.testing.assert_array_equal(restored.xyxy, source.xyxy)
        np.testing.assert_array_equal(restored.class_id, source.class_id)
        np.testing.assert_array_equal(restored.confidence, source.confidence)
        np.testing.assert_array_equal(restored.mask, source.mask)
        assert restored.mask.shape == (count, 7, 9)


def test_persistent_unix_client_validates_ping_and_prediction(tmp_path: Path):
    socket_path = tmp_path / "trt.sock"
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            connection = listener.accept()
            ping = connection.recv()
            connection.send({
                "schema": PROTOCOL_SCHEMA,
                "request_id": ping["request_id"],
                "ok": True,
                "model_size": "xlarge",
                "maximum_batch_size": 3,
                "engine_sha256": "test-engine",
            })
            request = connection.recv()
            assert request["camera_key"] == "cam_3"
            assert request["image"].shape == (4, 5, 3)
            connection.send({
                "schema": PROTOCOL_SCHEMA,
                "request_id": request["request_id"],
                "ok": True,
                "prediction": pack_prediction(_prediction(1, 4, 5)),
                "diagnostics": {"batch_size": 3, "batch_index": 0},
            })
            connection.close()
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    client = BatchInferenceClient(
        socket_path, "cam_3", timeout_sec=1.0, expected_model_size="xlarge"
    )
    try:
        assert client.ping()["maximum_batch_size"] == 3
        result = client.predict(np.zeros((4, 5, 3), dtype=np.uint8), 0.3)
        assert result.mask.shape == (1, 4, 5)
        assert client.last_diagnostics["batch_size"] == 3
        assert client.last_diagnostics["ipc_round_trip_ms"] >= 0.0
    finally:
        client.close()
        thread.join(timeout=2.0)
        listener.close()
    assert not thread.is_alive()
    assert not errors


def test_zero_window_batches_only_requests_already_waiting():
    from run_rfdetr_trt_batch_server import PendingRequest, TrtBatchServer

    server = object.__new__(TrtBatchServer)
    server.maximum_batch_size = 3
    server.batch_window_sec = 0.0
    server._condition = threading.Condition()
    server._queue = deque(
        PendingRequest({'request_id': value}, time.monotonic())
        for value in ('A', 'B', 'C', 'D')
    )
    server._stopping = False

    batch = server._next_batch()

    assert [item.payload['request_id'] for item in batch] == ['A', 'B', 'C']
    assert [item.payload['request_id'] for item in server._queue] == ['D']
