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
            def from_checkpoint(cls, path):
                del cls
                assert path == str(tmp_path / "model.pth")
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
