"""Workspace ROI filtering and temporal class stabilization for detections."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
import math
import time

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
class SmallComponentCleanupConfig:
    """Remove insignificant disconnected islands inside each instance bbox."""

    enabled: bool = False
    minimum_area_px: int = 16
    minimum_area_ratio: float = 0.005

    def __post_init__(self) -> None:
        if self.minimum_area_px < 1:
            raise ValueError("minimum_area_px must be positive")
        if (
            not math.isfinite(self.minimum_area_ratio)
            or not 0.0 <= self.minimum_area_ratio <= 1.0
        ):
            raise ValueError("minimum_area_ratio must be finite and in [0, 1]")


@dataclass(frozen=True)
class DetectionPostprocessorConfig:
    small_component_cleanup: SmallComponentCleanupConfig = field(
        default_factory=SmallComponentCleanupConfig
    )
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
    mask_area: int
    mask_bounds_xyxy_px: tuple[int, int, int, int] | None
    bbox_xyxy_px: tuple[float, float, float, float]
    centroid_xy_px: tuple[float, float]
    stable_class: _ClassEvidence
    history: deque[_ClassEvidence]
    missed_frames: int = 0


@dataclass(frozen=True)
class _MaskGeometry:
    """Frame-local mask measurements reused by ROI and track association."""

    area: int
    bounds_xyxy_px: tuple[int, int, int, int] | None
    centroid_xy_px: tuple[float, float]


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


def _measure_mask(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> _MaskGeometry:
    """Measure a mask once instead of rescanning it for every track pair."""
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    if area:
        return _MaskGeometry(
            area=area,
            bounds_xyxy_px=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ),
            centroid_xy_px=(float(xs.mean()), float(ys.mean())),
        )
    return _MaskGeometry(
        area=0,
        bounds_xyxy_px=None,
        centroid_xy_px=(
            (bbox[0] + bbox[2]) * 0.5,
            (bbox[1] + bbox[3]) * 0.5,
        ),
    )


def _bbox_crop_bounds(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Return a clipped, one-pixel-padded integer bbox crop."""
    if width < 1 or height < 1 or not all(math.isfinite(value) for value in bbox):
        return None
    x0 = max(0, int(math.floor(min(bbox[0], bbox[2]))) - 1)
    y0 = max(0, int(math.floor(min(bbox[1], bbox[3]))) - 1)
    x1 = min(width, int(math.ceil(max(bbox[0], bbox[2]))) + 1)
    y1 = min(height, int(math.ceil(max(bbox[1], bbox[3]))) + 1)
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _geometry_from_components(
    stats: np.ndarray,
    centroids: np.ndarray,
    component_indices: np.ndarray,
    crop_x0: int,
    crop_y0: int,
) -> _MaskGeometry:
    """Combine selected connected-component statistics without a mask scan."""
    selected = np.asarray(component_indices, dtype=np.int32).reshape(-1)
    if selected.size == 0:
        return _MaskGeometry(area=0, bounds_xyxy_px=None, centroid_xy_px=(0.0, 0.0))
    areas = stats[selected, cv2.CC_STAT_AREA].astype(np.int64)
    area = int(areas.sum())
    left = int(stats[selected, cv2.CC_STAT_LEFT].min()) + crop_x0
    top = int(stats[selected, cv2.CC_STAT_TOP].min()) + crop_y0
    right = int(
        np.max(
            stats[selected, cv2.CC_STAT_LEFT]
            + stats[selected, cv2.CC_STAT_WIDTH]
        )
    ) + crop_x0
    bottom = int(
        np.max(
            stats[selected, cv2.CC_STAT_TOP]
            + stats[selected, cv2.CC_STAT_HEIGHT]
        )
    ) + crop_y0
    centroid_x = float(
        np.dot(centroids[selected, 0], areas) / max(area, 1)
    ) + crop_x0
    centroid_y = float(
        np.dot(centroids[selected, 1], areas) / max(area, 1)
    ) + crop_y0
    return _MaskGeometry(
        area=area,
        bounds_xyxy_px=(left, top, right, bottom),
        centroid_xy_px=(centroid_x, centroid_y),
    )


def _mask_iou_from_geometry(
    left: np.ndarray,
    left_geometry: _MaskGeometry,
    right: np.ndarray,
    right_geometry: _MaskGeometry,
) -> float:
    if left.shape != right.shape:
        return 0.0
    left_bounds = left_geometry.bounds_xyxy_px
    right_bounds = right_geometry.bounds_xyxy_px
    if left_bounds is None or right_bounds is None:
        return 0.0
    x0 = max(left_bounds[0], right_bounds[0])
    y0 = max(left_bounds[1], right_bounds[1])
    x1 = min(left_bounds[2], right_bounds[2])
    y1 = min(left_bounds[3], right_bounds[3])
    if x0 >= x1 or y0 >= y1:
        intersection = 0
    else:
        intersection = int(
            np.count_nonzero(
                left[y0:y1, x0:x1] & right[y0:y1, x0:x1]
            )
        )
    union = left_geometry.area + right_geometry.area - intersection
    return intersection / union if union else 0.0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return exact mask IoU; retained as a standalone test/helper API."""
    empty_bbox = (0.0, 0.0, 0.0, 0.0)
    return _mask_iou_from_geometry(
        left,
        _measure_mask(left, empty_bbox),
        right,
        _measure_mask(right, empty_bbox),
    )


def _mask_centroid(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    return _measure_mask(mask, bbox).centroid_xy_px


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
                or config.small_component_cleanup.enabled
                or config.workspace_roi.enabled
                or config.temporal_class.enabled
            ),
            "input_instances": 0,
            "class_confidence_rejected_instances": 0,
            "component_cleanup_enabled": config.small_component_cleanup.enabled,
            "component_cleanup_latency_ms": 0.0,
            "component_cleanup_modified_instances": 0,
            "small_components_removed": 0,
            "small_component_pixels_removed": 0,
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

    def _inside_roi(
        self,
        item: DetectionInstance,
        roi: np.ndarray,
        geometry: _MaskGeometry | None = None,
    ) -> bool:
        mask = np.asarray(item.mask, dtype=bool)
        if mask.shape != roi.shape:
            raise ValueError(
                f"instance mask shape {mask.shape} does not match ROI {roi.shape}"
            )
        measured = geometry or _measure_mask(mask, item.bbox_xyxy_px)
        if measured.area == 0 or measured.bounds_xyxy_px is None:
            return False
        x0, y0, x1, y1 = measured.bounds_xyxy_px
        overlap = (
            int(np.count_nonzero(mask[y0:y1, x0:x1] & roi[y0:y1, x0:x1]))
            / measured.area
        )
        if overlap < self.config.workspace_roi.minimum_mask_overlap:
            return False
        if self.config.workspace_roi.require_mask_centroid_inside:
            centroid_x, centroid_y = measured.centroid_xy_px
            x = min(max(int(round(centroid_x)), 0), roi.shape[1] - 1)
            y = min(max(int(round(centroid_y)), 0), roi.shape[0] - 1)
            if not roi[y, x]:
                return False
        return True

    def _filter_roi(
        self,
        batch: DetectionBatch,
        geometries: dict[int, _MaskGeometry] | None = None,
    ) -> list[DetectionInstance]:
        if not self.config.workspace_roi.enabled:
            return list(batch.instances)
        roi = self._roi_mask(batch.image_width, batch.image_height)
        return [
            item
            for item in batch.instances
            if self._inside_roi(
                item,
                roi,
                geometries.get(id(item)) if geometries is not None else None,
            )
        ]

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

    def _cleanup_small_components(
        self,
        item: DetectionInstance,
    ) -> tuple[DetectionInstance, _MaskGeometry, int, int]:
        """Clean one instance in its bbox crop and return reusable geometry."""
        mask = np.asarray(item.mask, dtype=bool)
        height, width = mask.shape
        bounds = _bbox_crop_bounds(item.bbox_xyxy_px, width, height)
        if bounds is None:
            return item, _measure_mask(mask, item.bbox_xyxy_px), 0, 0
        x0, y0, x1, y1 = bounds
        crop = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)
        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            crop, connectivity=8
        )
        if label_count <= 1:
            return item, _measure_mask(mask, item.bbox_xyxy_px), 0, 0

        foreground_indices = np.arange(1, label_count, dtype=np.int32)
        areas = stats[foreground_indices, cv2.CC_STAT_AREA].astype(np.int64)
        total_area = int(areas.sum())
        threshold = max(
            self.config.small_component_cleanup.minimum_area_px,
            int(math.ceil(
                total_area
                * self.config.small_component_cleanup.minimum_area_ratio
            )),
        )
        keep = areas >= threshold
        keep[int(np.argmax(areas))] = True
        kept_indices = foreground_indices[keep]
        geometry = _geometry_from_components(
            stats, centroids, kept_indices, x0, y0
        )
        removed_count = int(np.count_nonzero(~keep))
        if removed_count == 0:
            return item, geometry, 0, 0

        keep_lookup = np.zeros(label_count, dtype=bool)
        keep_lookup[kept_indices] = True
        cleaned_mask = np.zeros_like(mask, dtype=bool)
        cleaned_mask[y0:y1, x0:x1] = keep_lookup[labels]
        removed_pixels = int(areas[~keep].sum())
        assert geometry.bounds_xyxy_px is not None
        bx0, by0, bx1, by1 = geometry.bounds_xyxy_px
        cleaned_item = replace(
            item,
            mask=cleaned_mask,
            bbox_xyxy_px=(float(bx0), float(by0), float(bx1), float(by1)),
        )
        return cleaned_item, geometry, removed_count, removed_pixels

    def _association_score(
        self,
        track: _Track,
        item: DetectionInstance,
        width: int,
        height: int,
        geometry: _MaskGeometry | None = None,
    ) -> float | None:
        config = self.config.temporal_class
        measured = geometry or _measure_mask(item.mask, item.bbox_xyxy_px)
        bbox_iou = _bbox_iou(track.bbox_xyxy_px, item.bbox_xyxy_px)
        minimum_area = min(track.mask_area, measured.area)
        area_ratio = (
            max(track.mask_area, measured.area) / minimum_area
            if minimum_area > 0
            else math.inf
        )
        diagonal = math.hypot(width, height)
        distance_norm = (
            math.hypot(
                measured.centroid_xy_px[0] - track.centroid_xy_px[0],
                measured.centroid_xy_px[1] - track.centroid_xy_px[1],
            )
            / diagonal
            if diagonal > 0.0
            else 1.0
        )
        if area_ratio > config.maximum_mask_area_ratio:
            return None
        if distance_norm > config.maximum_centroid_distance_norm:
            return None
        mask_iou = _mask_iou_from_geometry(
            track.mask,
            _MaskGeometry(
                area=track.mask_area,
                bounds_xyxy_px=track.mask_bounds_xyxy_px,
                centroid_xy_px=track.centroid_xy_px,
            ),
            item.mask,
            measured,
        )
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

    def _new_track(
        self,
        item: DetectionInstance,
        geometry: _MaskGeometry | None = None,
    ) -> _Track:
        evidence = self._evidence(item)
        measured = geometry or _measure_mask(item.mask, item.bbox_xyxy_px)
        track = _Track(
            track_id=self._next_track_id,
            mask=np.asarray(item.mask, dtype=bool).copy(),
            mask_area=measured.area,
            mask_bounds_xyxy_px=measured.bounds_xyxy_px,
            bbox_xyxy_px=tuple(item.bbox_xyxy_px),
            centroid_xy_px=measured.centroid_xy_px,
            stable_class=evidence,
            history=deque(
                [evidence], maxlen=self.config.temporal_class.history_size
            ),
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    def _update_track(
        self,
        track: _Track,
        item: DetectionInstance,
        geometry: _MaskGeometry | None = None,
    ) -> tuple[DetectionInstance, bool, bool, bool]:
        evidence = self._evidence(item)
        measured = geometry or _measure_mask(item.mask, item.bbox_xyxy_px)
        raw_class_transition = (
            bool(track.history)
            and track.history[-1].model_class_index
            != evidence.model_class_index
        )
        track.history.append(evidence)
        track.mask = np.asarray(item.mask, dtype=bool).copy()
        track.mask_area = measured.area
        track.mask_bounds_xyxy_px = measured.bounds_xyxy_px
        track.bbox_xyxy_px = tuple(item.bbox_xyxy_px)
        track.centroid_xy_px = measured.centroid_xy_px
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
        geometries: dict[int, _MaskGeometry] | None = None,
    ) -> tuple[list[DetectionInstance], int, int, int, int, int]:
        for track in self._tracks.values():
            track.missed_frames += 1

        candidates: list[tuple[float, int, int]] = []
        for item_index, item in enumerate(instances):
            geometry = (
                geometries.get(id(item))
                if geometries is not None
                else _measure_mask(item.mask, item.bbox_xyxy_px)
            )
            for track_id, track in self._tracks.items():
                score = self._association_score(
                    track, item, width, height, geometry
                )
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
            geometry = (
                geometries.get(id(item))
                if geometries is not None
                else _measure_mask(item.mask, item.bbox_xyxy_px)
            )
            track = item_to_track.get(item_index)
            if track is None:
                track = self._new_track(item, geometry)
                created += 1
                output.append(item)
                continue
            (
                stabilized,
                overridden,
                switched,
                raw_class_transition,
            ) = self._update_track(track, item, geometry)
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
        cleanup_started = time.perf_counter()
        cleanup_modified = 0
        components_removed = 0
        component_pixels_removed = 0
        geometries: dict[int, _MaskGeometry] | None = None
        if self.config.small_component_cleanup.enabled:
            geometries = {}
            cleaned_instances: list[DetectionInstance] = []
            for item in filtered:
                cleaned, geometry, removed, removed_pixels = (
                    self._cleanup_small_components(item)
                )
                cleaned_instances.append(cleaned)
                geometries[id(cleaned)] = geometry
                cleanup_modified += int(cleaned is not item)
                components_removed += removed
                component_pixels_removed += removed_pixels
            filtered = cleaned_instances
        cleanup_ms = (time.perf_counter() - cleanup_started) * 1000.0
        need_geometry = (
            self.config.small_component_cleanup.enabled
            or self.config.workspace_roi.enabled
            or self.config.temporal_class.enabled
        )
        if need_geometry and geometries is None:
            geometries = {
                id(item): _measure_mask(item.mask, item.bbox_xyxy_px)
                for item in filtered
            }
        filtered = self._filter_roi(
            replace(batch, instances=filtered), geometries
        )
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
            ) = self._stabilize(
                filtered,
                batch.image_width,
                batch.image_height,
                geometries,
            )
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
                or self.config.small_component_cleanup.enabled
                or self.config.workspace_roi.enabled
                or self.config.temporal_class.enabled
            ),
            "input_instances": len(batch.instances),
            "class_confidence_rejected_instances": (
                class_confidence_rejected
            ),
            "component_cleanup_enabled": (
                self.config.small_component_cleanup.enabled
            ),
            "component_cleanup_minimum_area_px": (
                self.config.small_component_cleanup.minimum_area_px
            ),
            "component_cleanup_minimum_area_ratio": (
                self.config.small_component_cleanup.minimum_area_ratio
            ),
            "component_cleanup_latency_ms": cleanup_ms,
            "component_cleanup_modified_instances": cleanup_modified,
            "small_components_removed": components_removed,
            "small_component_pixels_removed": component_pixels_removed,
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
