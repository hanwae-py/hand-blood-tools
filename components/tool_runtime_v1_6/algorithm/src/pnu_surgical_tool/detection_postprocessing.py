"""Workspace ROI filtering and temporal class stabilization for detections."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
import math

import cv2
import numpy as np

from .types import DetectionBatch, DetectionInstance


@dataclass(frozen=True)
class WorkspaceRoiConfig:
    """Filter instances to a calibrated workspace polygon."""

    enabled: bool = False
    polygon_norm_xy: tuple[float, ...] = ()
    minimum_mask_overlap: float = 0.5
    require_mask_centroid_inside: bool = True

    def __post_init__(self) -> None:
        polygon = tuple(float(value) for value in self.polygon_norm_xy)
        if self.enabled and (len(polygon) < 6 or len(polygon) % 2):
            raise ValueError(
                "enabled ROI requires at least three normalized x/y points"
            )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in polygon):
            raise ValueError("ROI polygon coordinates must be finite and in [0, 1]")
        if not 0.0 <= self.minimum_mask_overlap <= 1.0:
            raise ValueError("minimum_mask_overlap must be in [0, 1]")
        object.__setattr__(self, "polygon_norm_xy", polygon)


@dataclass(frozen=True)
class TemporalClassConfig:
    """Associate without class labels and stabilize recent class evidence."""

    enabled: bool = False
    history_size: int = 7
    minimum_switch_frames: int = 3
    switch_score_margin: float = 0.2
    minimum_mask_iou: float = 0.10
    minimum_bbox_iou: float = 0.20
    maximum_centroid_distance_norm: float = 0.06
    maximum_mask_area_ratio: float = 3.0
    max_missed_frames: int = 3

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if not 1 <= self.minimum_switch_frames <= self.history_size:
            raise ValueError(
                "minimum_switch_frames must be in [1, history_size]"
            )
        if not math.isfinite(self.switch_score_margin) or self.switch_score_margin < 0.0:
            raise ValueError("switch_score_margin must be finite and non-negative")
        for name, value in (
            ("minimum_mask_iou", self.minimum_mask_iou),
            ("minimum_bbox_iou", self.minimum_bbox_iou),
            ("maximum_centroid_distance_norm", self.maximum_centroid_distance_norm),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not math.isfinite(self.maximum_mask_area_ratio)
            or self.maximum_mask_area_ratio < 1.0
        ):
            raise ValueError(
                "maximum_mask_area_ratio must be finite and at least 1"
            )
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")


@dataclass(frozen=True)
class DetectionPostprocessorConfig:
    workspace_roi: WorkspaceRoiConfig = field(default_factory=WorkspaceRoiConfig)
    temporal_class: TemporalClassConfig = field(default_factory=TemporalClassConfig)
    default_class_confidence_threshold: float = 0.0
    class_confidence_thresholds: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        default_threshold = float(self.default_class_confidence_threshold)
        if (
            not math.isfinite(default_threshold)
            or not 0.0 <= default_threshold <= 1.0
        ):
            raise ValueError(
                "default class confidence threshold must be finite and in [0, 1]"
            )
        object.__setattr__(
            self, "default_class_confidence_threshold", default_threshold
        )
        normalized: list[tuple[str, float]] = []
        names: set[str] = set()
        for raw_name, raw_threshold in self.class_confidence_thresholds:
            name = str(raw_name).strip()
            threshold = float(raw_threshold)
            if not name:
                raise ValueError("class confidence threshold name is required")
            if name in names:
                raise ValueError(f"duplicate class confidence threshold: {name}")
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "class confidence thresholds must be finite and in [0, 1]"
                )
            names.add(name)
            normalized.append((name, threshold))
        object.__setattr__(self, "class_confidence_thresholds", tuple(normalized))


@dataclass
class _ClassEvidence:
    canonical_class_id: int
    model_class_index: int
    class_name: str
    confidence: float


@dataclass
class _Track:
    track_id: int
    mask: np.ndarray
    bbox_xyxy_px: tuple[float, float, float, float]
    centroid_xy_px: tuple[float, float]
    stable_class: _ClassEvidence
    history: deque[_ClassEvidence]
    missed_frames: int = 0


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def _mask_centroid(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size:
        return float(xs.mean()), float(ys.mean())
    return (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5


class DetectionPostprocessor:
    """Apply workspace filtering followed by class-independent tracking.

    Track IDs are deliberately internal. The public detection contract keeps
    ``frame_local_instance_id`` semantics while class labels are stabilized.
    """

    def __init__(self, config: DetectionPostprocessorConfig) -> None:
        self.config = config
        self._roi_cache: dict[tuple[int, int], np.ndarray] = {}
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._class_confidence_thresholds = dict(
            config.class_confidence_thresholds
        )
        self.last_diagnostics: dict[str, int | float | bool] = {
            "enabled": bool(
                self._class_confidence_thresholds
                or config.default_class_confidence_threshold > 0.0
                or config.workspace_roi.enabled
                or config.temporal_class.enabled
            ),
            "input_instances": 0,
            "class_confidence_rejected_instances": 0,
            "roi_rejected_instances": 0,
            "output_instances": 0,
            "active_tracks": 0,
            "tracks_created": 0,
            "tracks_matched": 0,
            "raw_class_transitions": 0,
            "class_overrides": 0,
            "class_switches": 0,
        }

    def reset(self) -> None:
        """Forget temporal state, for example after a bag/time discontinuity."""
        self._tracks.clear()
        self._next_track_id = 1

    @property
    def active_track_count(self) -> int:
        """Return the number of internal association tracks."""
        return len(self._tracks)

    def roi_polygon_pixels(self, width: int, height: int) -> np.ndarray | None:
        """Return the configured ROI polygon as OpenCV integer points."""
        if not self.config.workspace_roi.enabled:
            return None
        values = self.config.workspace_roi.polygon_norm_xy
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        points[:, 0] *= max(width - 1, 0)
        points[:, 1] *= max(height - 1, 0)
        return np.rint(points).astype(np.int32)

    def _roi_mask(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self._roi_cache.get(key)
        if cached is not None:
            return cached
        polygon = self.roi_polygon_pixels(width, height)
        if polygon is None:
            mask = np.ones((height, width), dtype=bool)
        else:
            mask_u8 = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask_u8, [polygon], 1)
            mask = mask_u8.astype(bool)
        self._roi_cache[key] = mask
        return mask

    def _inside_roi(self, item: DetectionInstance, roi: np.ndarray) -> bool:
        mask = np.asarray(item.mask, dtype=bool)
        if mask.shape != roi.shape:
            raise ValueError(
                f"instance mask shape {mask.shape} does not match ROI {roi.shape}"
            )
        area = int(np.count_nonzero(mask))
        if area == 0:
            return False
        overlap = int(np.count_nonzero(mask & roi)) / area
        if overlap < self.config.workspace_roi.minimum_mask_overlap:
            return False
        if self.config.workspace_roi.require_mask_centroid_inside:
            centroid_x, centroid_y = _mask_centroid(mask, item.bbox_xyxy_px)
            x = min(max(int(round(centroid_x)), 0), roi.shape[1] - 1)
            y = min(max(int(round(centroid_y)), 0), roi.shape[0] - 1)
            if not roi[y, x]:
                return False
        return True

    def _filter_roi(self, batch: DetectionBatch) -> list[DetectionInstance]:
        if not self.config.workspace_roi.enabled:
            return list(batch.instances)
        roi = self._roi_mask(batch.image_width, batch.image_height)
        return [item for item in batch.instances if self._inside_roi(item, roi)]

    def _filter_class_confidence(
        self, instances: list[DetectionInstance]
    ) -> list[DetectionInstance]:
        if not self._class_confidence_thresholds:
            return list(instances)
        return [
            item
            for item in instances
            if item.class_confidence >= self._class_confidence_thresholds.get(
                item.class_name,
                self.config.default_class_confidence_threshold,
            )
        ]

    def _association_score(
        self,
        track: _Track,
        item: DetectionInstance,
        width: int,
        height: int,
    ) -> float | None:
        config = self.config.temporal_class
        mask_iou = _mask_iou(track.mask, item.mask)
        bbox_iou = _bbox_iou(track.bbox_xyxy_px, item.bbox_xyxy_px)
        track_area = int(np.count_nonzero(track.mask))
        item_area = int(np.count_nonzero(item.mask))
        minimum_area = min(track_area, item_area)
        area_ratio = (
            max(track_area, item_area) / minimum_area
            if minimum_area > 0
            else math.inf
        )
        centroid = _mask_centroid(item.mask, item.bbox_xyxy_px)
        diagonal = math.hypot(width, height)
        distance_norm = (
            math.hypot(
                centroid[0] - track.centroid_xy_px[0],
                centroid[1] - track.centroid_xy_px[1],
            )
            / diagonal
            if diagonal > 0.0
            else 1.0
        )
        if area_ratio > config.maximum_mask_area_ratio:
            return None
        if distance_norm > config.maximum_centroid_distance_norm:
            return None
        if (
            mask_iou < config.minimum_mask_iou
            and bbox_iou < config.minimum_bbox_iou
        ):
            return None
        distance_similarity = max(
            0.0,
            1.0
            - distance_norm / max(config.maximum_centroid_distance_norm, 1e-9),
        )
        return 0.55 * mask_iou + 0.30 * bbox_iou + 0.15 * distance_similarity

    @staticmethod
    def _evidence(item: DetectionInstance) -> _ClassEvidence:
        return _ClassEvidence(
            canonical_class_id=item.canonical_class_id,
            model_class_index=item.model_class_index,
            class_name=item.class_name,
            confidence=float(item.class_confidence),
        )

    def _new_track(self, item: DetectionInstance) -> _Track:
        evidence = self._evidence(item)
        track = _Track(
            track_id=self._next_track_id,
            mask=np.asarray(item.mask, dtype=bool).copy(),
            bbox_xyxy_px=tuple(item.bbox_xyxy_px),
            centroid_xy_px=_mask_centroid(item.mask, item.bbox_xyxy_px),
            stable_class=evidence,
            history=deque(
                [evidence], maxlen=self.config.temporal_class.history_size
            ),
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    def _update_track(
        self, track: _Track, item: DetectionInstance
    ) -> tuple[DetectionInstance, bool, bool, bool]:
        evidence = self._evidence(item)
        raw_class_transition = (
            bool(track.history)
            and track.history[-1].model_class_index
            != evidence.model_class_index
        )
        track.history.append(evidence)
        track.mask = np.asarray(item.mask, dtype=bool).copy()
        track.bbox_xyxy_px = tuple(item.bbox_xyxy_px)
        track.centroid_xy_px = _mask_centroid(item.mask, item.bbox_xyxy_px)
        track.missed_frames = 0

        scores: Counter[int] = Counter()
        counts: Counter[int] = Counter()
        metadata: dict[int, _ClassEvidence] = {}
        confidence_values: dict[int, list[float]] = {}
        for sample in track.history:
            index = sample.model_class_index
            scores[index] += sample.confidence
            counts[index] += 1
            metadata[index] = sample
            confidence_values.setdefault(index, []).append(sample.confidence)

        stable_index = track.stable_class.model_class_index
        candidate_index = max(scores, key=lambda index: (scores[index], counts[index]))
        switched = False
        if (
            candidate_index != stable_index
            and counts[candidate_index]
            >= self.config.temporal_class.minimum_switch_frames
            and scores[candidate_index]
            >= scores[stable_index]
            + self.config.temporal_class.switch_score_margin
        ):
            track.stable_class = metadata[candidate_index]
            stable_index = candidate_index
            switched = True

        stable = track.stable_class
        stable_confidence = float(np.mean(confidence_values.get(stable_index, [stable.confidence])))
        overridden = evidence.model_class_index != stable_index
        output = replace(
            item,
            canonical_class_id=stable.canonical_class_id,
            model_class_index=stable.model_class_index,
            class_name=stable.class_name,
            class_confidence=stable_confidence,
        )
        return output, overridden, switched, raw_class_transition

    def _stabilize(
        self,
        instances: list[DetectionInstance],
        width: int,
        height: int,
    ) -> tuple[list[DetectionInstance], int, int, int, int, int]:
        for track in self._tracks.values():
            track.missed_frames += 1

        candidates: list[tuple[float, int, int]] = []
        for item_index, item in enumerate(instances):
            for track_id, track in self._tracks.items():
                score = self._association_score(track, item, width, height)
                if score is not None:
                    candidates.append((score, item_index, track_id))
        candidates.sort(reverse=True)

        item_to_track: dict[int, _Track] = {}
        used_tracks: set[int] = set()
        for _score, item_index, track_id in candidates:
            if item_index in item_to_track or track_id in used_tracks:
                continue
            item_to_track[item_index] = self._tracks[track_id]
            used_tracks.add(track_id)

        created = 0
        matched = 0
        raw_transitions = 0
        overrides = 0
        switches = 0
        output: list[DetectionInstance] = []
        for item_index, item in enumerate(instances):
            track = item_to_track.get(item_index)
            if track is None:
                track = self._new_track(item)
                created += 1
                output.append(item)
                continue
            (
                stabilized,
                overridden,
                switched,
                raw_class_transition,
            ) = self._update_track(track, item)
            output.append(stabilized)
            matched += 1
            raw_transitions += int(raw_class_transition)
            overrides += int(overridden)
            switches += int(switched)

        maximum_missed = self.config.temporal_class.max_missed_frames
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed_frames <= maximum_missed
        }
        return output, created, matched, raw_transitions, overrides, switches

    def process(self, batch: DetectionBatch) -> DetectionBatch:
        """Return a new batch after configured ROI and temporal processing."""
        filtered = self._filter_class_confidence(batch.instances)
        class_confidence_rejected = len(batch.instances) - len(filtered)
        filtered = self._filter_roi(replace(batch, instances=filtered))
        roi_rejected = (
            len(batch.instances) - class_confidence_rejected - len(filtered)
        )
        created = matched = raw_transitions = overrides = switches = 0
        if self.config.temporal_class.enabled:
            (
                filtered,
                created,
                matched,
                raw_transitions,
                overrides,
                switches,
            ) = self._stabilize(filtered, batch.image_width, batch.image_height)
        output = DetectionBatch(
            image_width=batch.image_width,
            image_height=batch.image_height,
            model_version=batch.model_version,
            ontology_version=batch.ontology_version,
            instances=filtered,
            inference_latency_ms=batch.inference_latency_ms,
        )
        self.last_diagnostics = {
            "enabled": bool(
                self._class_confidence_thresholds
                or self.config.default_class_confidence_threshold > 0.0
                or self.config.workspace_roi.enabled
                or self.config.temporal_class.enabled
            ),
            "input_instances": len(batch.instances),
            "class_confidence_rejected_instances": (
                class_confidence_rejected
            ),
            "roi_rejected_instances": roi_rejected,
            "output_instances": len(output.instances),
            "active_tracks": len(self._tracks),
            "tracks_created": created,
            "tracks_matched": matched,
            "raw_class_transitions": raw_transitions,
            "class_overrides": overrides,
            "class_switches": switches,
        }
        return output
