from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _FakePredictionModel:
    def __init__(self) -> None:
        self.observed_image: np.ndarray | None = None

    def predict(self, image, threshold, include_source_image):
        del threshold, include_source_image
        self.observed_image = np.asarray(image).copy()
        masks = np.zeros((2, 2, 2), dtype=bool)
        masks[:, 0, 0] = True
        return SimpleNamespace(
            xyxy=np.asarray([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=float),
            class_id=np.asarray([0, 1], dtype=int),
            confidence=np.asarray([0.9, 0.8], dtype=float),
            mask=masks,
        )


@pytest.mark.parametrize(
    ("model_size", "expected_class", "expected_color", "expected_instances"),
    [
        ("small", "RFDETRSegSmall", [1, 2, 3], 1),
        ("medium", "RFDETRSegMedium", [3, 2, 1], 2),
        ("large", "RFDETRSegLarge", [3, 2, 1], 2),
        ("xlarge", "RFDETRSegXLarge", [3, 2, 1], 2),
    ],
)
def test_model_selection_color_contract_and_default_nms(
    tmp_path,
    monkeypatch,
    model_size,
    expected_class,
    expected_color,
    expected_instances,
):
    from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector

    loaded_classes: list[str] = []
    prediction_models: list[_FakePredictionModel] = []

    def loader_class(name: str):
        class _Loader:
            @classmethod
            def from_checkpoint(cls, path, *, device):
                del cls
                assert path == str(tmp_path / "model.pth")
                assert device == "cpu"
                loaded_classes.append(name)
                model = _FakePredictionModel()
                prediction_models.append(model)
                return model

        return _Loader

    fake_rfdetr = SimpleNamespace(
        RFDETRSegSmall=loader_class("RFDETRSegSmall"),
        RFDETRSegMedium=loader_class("RFDETRSegMedium"),
        RFDETRSegLarge=loader_class("RFDETRSegLarge"),
        RFDETRSegXLarge=loader_class("RFDETRSegXLarge"),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            synchronize=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "rfdetr", fake_rfdetr)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    ontology = Path(__file__).resolve().parents[1] / "model" / "ontology.json"
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=checkpoint,
            ontology_path=ontology,
            model_size=model_size,
            optimize=False,
        )
    )
    bgr = np.asarray([[[1, 2, 3], [4, 5, 6]]] * 2, dtype=np.uint8)

    result = detector.predict(bgr, color_order="BGR")

    assert loaded_classes == [expected_class]
    assert prediction_models[0].observed_image[0, 0].tolist() == expected_color
    assert len(result.instances) == expected_instances


def test_detector_config_defaults_to_xlarge():
    from pnu_surgical_tool import DetectorConfig

    config = DetectorConfig(
        checkpoint_path="unused.pth",
        ontology_path="unused.json",
    )

    assert config.model_size == "xlarge"


def test_bipolar_containment_uses_larger_bbox_for_near_equal_confidence():
    from pnu_surgical_tool.rfdetr_inference import (
        same_class_mask_containment_indices,
    )

    complete = np.zeros((12, 8), dtype=bool)
    complete[1:11, 1:7] = True
    partial = np.zeros_like(complete)
    partial[3:9, 2:6] = True

    keep = same_class_mask_containment_indices(
        [complete, partial],
        np.asarray([4, 4]),
        np.asarray([0.88, 0.90]),
        boxes_xyxy=np.asarray([[0, 0, 8, 12], [2, 3, 6, 9]], dtype=float),
        prefer_larger_bbox_class_ids={4},
        larger_bbox_max_confidence_gap=0.03,
    )

    assert keep.tolist() == [0]


def test_bipolar_containment_rejects_low_confidence_larger_bbox():
    from pnu_surgical_tool.rfdetr_inference import (
        same_class_mask_containment_indices,
    )

    complete = np.zeros((12, 8), dtype=bool)
    complete[1:11, 1:7] = True
    partial = np.zeros_like(complete)
    partial[3:9, 2:6] = True

    keep = same_class_mask_containment_indices(
        [complete, partial],
        np.asarray([4, 4]),
        np.asarray([0.70, 0.90]),
        boxes_xyxy=np.asarray([[0, 0, 8, 12], [2, 3, 6, 9]], dtype=float),
        prefer_larger_bbox_class_ids={4},
        larger_bbox_max_confidence_gap=0.03,
    )

    assert keep.tolist() == [1]


def test_non_bipolar_containment_still_prefers_confidence():
    from pnu_surgical_tool.rfdetr_inference import (
        same_class_mask_containment_indices,
    )

    complete = np.zeros((12, 8), dtype=bool)
    complete[1:11, 1:7] = True
    partial = np.zeros_like(complete)
    partial[3:9, 2:6] = True

    keep = same_class_mask_containment_indices(
        [complete, partial],
        np.asarray([3, 3]),
        np.asarray([0.45, 0.90]),
        boxes_xyxy=np.asarray([[0, 0, 8, 12], [2, 3, 6, 9]], dtype=float),
        prefer_larger_bbox_class_ids={4},
        larger_bbox_max_confidence_gap=0.03,
    )

    assert keep.tolist() == [1]


def test_containment_keeps_different_classes_and_crossing_instances():
    from pnu_surgical_tool.rfdetr_inference import (
        same_class_mask_containment_indices,
    )

    horizontal = np.zeros((12, 12), dtype=bool)
    horizontal[4:8, 1:11] = True
    vertical = np.zeros_like(horizontal)
    vertical[1:11, 4:8] = True
    contained_other_class = np.zeros_like(horizontal)
    contained_other_class[5:7, 3:9] = True

    keep = same_class_mask_containment_indices(
        [horizontal, vertical, contained_other_class],
        np.asarray([4, 4, 5]),
        np.asarray([0.9, 0.8, 0.7]),
    )

    assert keep.tolist() == [0, 1, 2]


def test_rejects_unknown_model_size(tmp_path):
    from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector

    ontology = Path(__file__).resolve().parents[1] / "model" / "ontology.json"
    with pytest.raises(ValueError, match="model_size"):
        SurgicalToolDetector(
            DetectorConfig(
                checkpoint_path=tmp_path / "model.pth",
                ontology_path=ontology,
                model_size="2xlarge",  # type: ignore[arg-type]
            )
        )


def test_shared_trt_backend_does_not_import_or_load_local_rfdetr(
    tmp_path, monkeypatch
):
    import builtins

    from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector
    from pnu_surgical_tool import rfdetr_inference
    from pnu_surgical_tool.trt_batch import RemotePrediction

    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    ontology = tmp_path / "ontology.json"
    ontology.write_text(
        '{"schema":"test","canonical_tool_classes":['
        + ",".join(
            f'{{"canonical_id":{index},"canonical_name":"tool-{index}"}}'
            for index in range(1, 9)
        )
        + "]}",
        encoding="utf-8",
    )
    imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rfdetr" or name.startswith("rfdetr."):
            imports.append(name)
            raise AssertionError("shared TRT client must not import RF-DETR")
        return real_import(name, *args, **kwargs)

    class FakeClient:
        def __init__(self, socket, camera, **kwargs):
            assert socket == "/tmp/shared.sock"
            assert camera == "cam_4"
            assert kwargs["expected_model_size"] == "xlarge"
            self.last_diagnostics = {}

        def ping(self):
            return {
                "model_size": "xlarge",
                "maximum_batch_size": 3,
                "engine_sha256": "engine",
            }

        def predict(self, image, threshold):
            assert image.shape == (2, 3, 3)
            assert threshold == pytest.approx(0.3)
            self.last_diagnostics = {"batch_size": 3}
            return RemotePrediction(
                xyxy=np.asarray([[0, 0, 3, 2]], dtype=np.float32),
                class_id=np.asarray([0], dtype=np.int32),
                confidence=np.asarray([0.9], dtype=np.float32),
                mask=np.ones((1, 2, 3), dtype=bool),
            )

        def close(self):
            pass

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(rfdetr_inference, "BatchInferenceClient", FakeClient)
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=checkpoint,
            ontology_path=ontology,
            model_size="xlarge",
            trt_server_socket="/tmp/shared.sock",
            trt_camera_key="cam_4",
        )
    )

    result = detector.predict(
        np.zeros((2, 3, 3), dtype=np.uint8), color_order="RGB"
    )

    assert imports == []
    assert detector.runtime_backend == "tensorrt_shared_dynamic_batch"
    assert detector.last_runtime_diagnostics["batch_size"] == 3
    assert len(result.instances) == 1
