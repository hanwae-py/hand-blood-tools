from __future__ import annotations

import numpy as np
import pytest

from pnu_surgical_tool import (
    DetectionBatch,
    DetectionInstance,
    DetectionPostprocessor,
    DetectionPostprocessorConfig,
    SmallComponentCleanupConfig,
    TemporalClassConfig,
    WorkspaceRoiConfig,
)
from pnu_surgical_tool.detection_postprocessing import _mask_iou


CLASSES = {
    0: (1, "Scalpel"),
    3: (4, "Adson Forceps"),
    4: (5, "Bipolar Forceps"),
}


def instance(
    class_index: int,
    mask: np.ndarray,
    confidence: float = 0.8,
    instance_id: int = 0,
) -> DetectionInstance:
    canonical_id, name = CLASSES[class_index]
    ys, xs = np.nonzero(mask)
    bbox = (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )
    return DetectionInstance(
        frame_local_instance_id=instance_id,
        canonical_class_id=canonical_id,
        model_class_index=class_index,
        class_name=name,
        class_confidence=confidence,
        bbox_xyxy_px=bbox,
        mask=mask,
    )


def batch(*items: DetectionInstance) -> DetectionBatch:
    return DetectionBatch(
        image_width=20,
        image_height=20,
        model_version="test",
        ontology_version="test",
        instances=list(items),
    )


def square(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((20, 20), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_workspace_roi_filters_by_mask_overlap_and_centroid() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            workspace_roi=WorkspaceRoiConfig(
                enabled=True,
                polygon_norm_xy=(0.0, 0.0, 0.55, 0.0, 0.55, 1.0, 0.0, 1.0),
                minimum_mask_overlap=0.6,
                require_mask_centroid_inside=True,
            )
        )
    )
    inside = instance(3, square(1, 2, 7, 8), instance_id=1)
    outside = instance(4, square(13, 2, 19, 8), instance_id=2)

    result = processor.process(batch(inside, outside))

    assert [item.frame_local_instance_id for item in result.instances] == [1]
    assert processor.last_diagnostics["roi_rejected_instances"] == 1


def test_majority_workspace_roi_rejects_small_partial_overlap() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            workspace_roi=WorkspaceRoiConfig(
                enabled=True,
                polygon_norm_xy=(
                    0.0, 0.0, 0.55, 0.0, 0.55, 1.0, 0.0, 1.0,
                ),
                minimum_mask_overlap=0.7,
                require_mask_centroid_inside=True,
            )
        )
    )
    fully_inside = instance(3, square(1, 2, 7, 8), instance_id=1)
    crosses_boundary = instance(4, square(8, 2, 13, 8), instance_id=2)

    result = processor.process(batch(fully_inside, crosses_boundary))

    assert [item.frame_local_instance_id for item in result.instances] == [1]
    assert processor.last_diagnostics["roi_rejected_instances"] == 1


def test_roi_configuration_rejects_invalid_polygon() -> None:
    with pytest.raises(ValueError, match="three"):
        WorkspaceRoiConfig(enabled=True, polygon_norm_xy=(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        WorkspaceRoiConfig(
            enabled=True,
            polygon_norm_xy=(0.0, 0.0, 1.2, 0.0, 0.0, 1.0),
        )


def test_class_confidence_threshold_filters_only_named_class() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            class_confidence_thresholds=(("Adson Forceps", 0.45),)
        )
    )
    result = processor.process(
        batch(
            instance(3, square(1, 2, 7, 8), 0.44, instance_id=1),
            instance(3, square(7, 2, 13, 8), 0.45, instance_id=2),
            instance(0, square(13, 2, 19, 8), 0.31, instance_id=3),
        )
    )

    assert [item.frame_local_instance_id for item in result.instances] == [2, 3]
    assert processor.last_diagnostics[
        "class_confidence_rejected_instances"
    ] == 1


def test_class_confidence_threshold_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DetectionPostprocessorConfig(
            class_confidence_thresholds=(("Adson Forceps", 1.1),)
        )


def test_lower_named_threshold_preserves_higher_default_for_other_classes() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            default_class_confidence_threshold=0.30,
            class_confidence_thresholds=(("Adson Forceps", 0.25),),
        )
    )
    result = processor.process(
        batch(
            instance(3, square(1, 2, 7, 8), 0.26, instance_id=1),
            instance(0, square(13, 2, 19, 8), 0.29, instance_id=2),
        )
    )

    assert [item.frame_local_instance_id for item in result.instances] == [1]
    assert processor.last_diagnostics[
        "class_confidence_rejected_instances"
    ] == 1


def test_single_frame_class_flicker_is_overridden_on_same_geometry() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            temporal_class=TemporalClassConfig(
                enabled=True,
                history_size=5,
                minimum_switch_frames=3,
            )
        )
    )
    geometry = square(4, 4, 10, 12)
    first = processor.process(batch(instance(3, geometry, 0.8))).instances[0]
    flicker_result = processor.process(batch(instance(4, geometry, 0.9)))
    flicker = flicker_result.instances[0]
    recovered = processor.process(batch(instance(3, geometry, 0.7))).instances[0]

    assert first.class_name == "Adson Forceps"
    assert flicker.class_name == "Adson Forceps"
    assert flicker_result.instances[0].class_confidence == pytest.approx(0.8)
    assert recovered.class_name == "Adson Forceps"
    assert processor.last_diagnostics["raw_class_transitions"] == 1
    assert processor.last_diagnostics["class_overrides"] == 0


def test_sustained_class_evidence_switches_after_minimum_frames() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            temporal_class=TemporalClassConfig(
                enabled=True,
                history_size=5,
                minimum_switch_frames=3,
                switch_score_margin=0.1,
            )
        )
    )
    geometry = square(4, 4, 10, 12)
    processor.process(batch(instance(3, geometry, 0.4)))
    names = []
    for _ in range(3):
        result = processor.process(batch(instance(4, geometry, 0.9)))
        names.append(result.instances[0].class_name)

    assert names == ["Adson Forceps", "Adson Forceps", "Bipolar Forceps"]
    assert processor.last_diagnostics["class_switches"] == 1


def test_association_is_class_independent_but_spatially_one_to_one() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            temporal_class=TemporalClassConfig(
                enabled=True,
                history_size=5,
                minimum_switch_frames=3,
                maximum_centroid_distance_norm=0.1,
            )
        )
    )
    left = square(1, 2, 6, 8)
    right = square(14, 2, 19, 8)
    processor.process(batch(instance(3, left), instance(4, right)))
    result = processor.process(
        batch(instance(4, left), instance(3, right))
    )

    assert [item.class_name for item in result.instances] == [
        "Adson Forceps",
        "Bipolar Forceps",
    ]
    assert processor.last_diagnostics["active_tracks"] == 2
    assert processor.last_diagnostics["class_overrides"] == 2


def test_large_area_change_does_not_inherit_previous_class() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            temporal_class=TemporalClassConfig(
                enabled=True,
                maximum_mask_area_ratio=2.0,
            )
        )
    )
    processor.process(batch(instance(3, square(6, 6, 10, 10))))
    result = processor.process(batch(instance(4, square(2, 2, 16, 16))))

    assert result.instances[0].class_name == "Bipolar Forceps"
    assert processor.last_diagnostics["tracks_created"] == 1
    assert processor.last_diagnostics["class_overrides"] == 0


def test_mask_iou_matches_full_frame_boolean_reference() -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        left = rng.random((37, 53)) > 0.85
        right = rng.random((37, 53)) > 0.80
        intersection = int(np.count_nonzero(left & right))
        union = int(np.count_nonzero(left | right))

        assert _mask_iou(left, right) == pytest.approx(
            intersection / union if union else 0.0
        )


def test_temporal_geometry_uses_mask_extent_not_declared_bbox() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            temporal_class=TemporalClassConfig(enabled=True)
        )
    )
    first_mask = square(1, 1, 5, 5)
    second_mask = first_mask.copy()
    first = instance(3, first_mask)
    second = instance(4, second_mask)
    first.bbox_xyxy_px = (10.0, 10.0, 11.0, 11.0)
    second.bbox_xyxy_px = (10.0, 10.0, 11.0, 11.0)

    processor.process(batch(first))
    result = processor.process(batch(second))

    assert result.instances[0].class_name == "Adson Forceps"
    assert processor.last_diagnostics["tracks_matched"] == 1


def test_small_disconnected_component_is_removed_inside_bbox_crop() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            small_component_cleanup=SmallComponentCleanupConfig(
                enabled=True,
                minimum_area_px=4,
                minimum_area_ratio=0.0,
            )
        )
    )
    mask = square(4, 4, 10, 10)
    mask[2, 2] = True

    result = processor.process(batch(instance(3, mask)))

    assert not result.instances[0].mask[2, 2]
    assert np.count_nonzero(result.instances[0].mask) == 36
    assert result.instances[0].bbox_xyxy_px == (4.0, 4.0, 10.0, 10.0)
    assert processor.last_diagnostics["small_components_removed"] == 1


def test_cleanup_preserves_multiple_substantial_components() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            small_component_cleanup=SmallComponentCleanupConfig(
                enabled=True,
                minimum_area_px=4,
                minimum_area_ratio=0.0,
            )
        )
    )
    mask = square(2, 2, 6, 6)
    mask[10:13, 10:13] = True

    original = instance(3, mask)
    result = processor.process(batch(original))

    assert result.instances[0] is original
    assert np.array_equal(result.instances[0].mask, mask)
    assert processor.last_diagnostics["small_components_removed"] == 0


def test_cleanup_always_preserves_largest_component() -> None:
    processor = DetectionPostprocessor(
        DetectionPostprocessorConfig(
            small_component_cleanup=SmallComponentCleanupConfig(
                enabled=True,
                minimum_area_px=100,
                minimum_area_ratio=1.0,
            )
        )
    )
    mask = square(5, 5, 8, 8)
    mask[2, 2] = True

    result = processor.process(batch(instance(3, mask)))

    assert np.count_nonzero(result.instances[0].mask) == 9
    assert processor.last_diagnostics["small_components_removed"] == 1


def test_disabled_cleanup_reuses_original_instance_and_mask() -> None:
    processor = DetectionPostprocessor(DetectionPostprocessorConfig())
    mask = square(4, 4, 10, 10)
    mask[2, 2] = True
    original = instance(3, mask)

    result = processor.process(batch(original))

    assert result.instances[0] is original
    assert result.instances[0].mask is original.mask
    assert result.instances[0].bbox_xyxy_px == original.bbox_xyxy_px
