"""Pure helpers for the surgical-tool observation JSON contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


CAM4_CLASS_NAMES = (
    'Scalpel',
    'Allis Forceps',
    'Mosquito',
    'Adson Forceps',
    'Bipolar Forceps',
    'Bovie',
    'Army-Navy Retractor',
    'Thyroid Retractor',
)


def mask_to_rle_counts(mask: np.ndarray) -> list[int]:
    """Return uncompressed COCO RLE counts in column-major order."""
    flat = np.asarray(mask, dtype=np.uint8).ravel(order='F')
    if flat.size == 0:
        return []

    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(boundaries).astype(int).tolist()
    if flat[0] != 0:
        counts.insert(0, 0)
    return counts


def compress_coco_rle_counts(counts: list[int]) -> str:
    """Encode counts with the compact codec used by COCO mask APIs."""
    delta_counts = list(counts)
    for index in range(3, len(delta_counts)):
        delta_counts[index] = counts[index] - counts[index - 2]

    characters = []
    for value in delta_counts:
        more = True
        while more:
            character = value & 0x1F
            value >>= 5
            more = value != -1 if character & 0x10 else value != 0
            if more:
                character |= 0x20
            characters.append(chr(character + 48))
    return ''.join(characters)


def _mask_coordinates(
    mask: np.ndarray,
    bbox_xyxy_px: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return foreground coordinates, scanning a trusted detector ROI when safe."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape={binary.shape}')

    if bbox_xyxy_px is None:
        return np.nonzero(binary)
    try:
        bbox = tuple(float(value) for value in bbox_xyxy_px)
    except (TypeError, ValueError):
        return np.nonzero(binary)
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        return np.nonzero(binary)

    height, width = binary.shape
    x0 = max(0, int(math.floor(min(bbox[0], bbox[2]))) - 1)
    y0 = max(0, int(math.floor(min(bbox[1], bbox[3]))) - 1)
    x1 = min(width, int(math.ceil(max(bbox[0], bbox[2]))) + 1)
    y1 = min(height, int(math.ceil(max(bbox[1], bbox[3]))) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.nonzero(binary)

    crop_ys, crop_xs = np.nonzero(binary[y0:y1, x0:x1])
    if crop_xs.size == 0:
        # An empty detector ROI can still accompany a malformed/stale bbox.
        # Retain the full-mask result in that exceptional path.
        return np.nonzero(binary)

    # A foreground pixel on a padded interior edge means the detector bbox is
    # not a safe mask bound. Fall back rather than truncating the lossless RLE.
    touches_unclipped_edge = bool(
        (x0 > 0 and np.any(crop_xs == 0))
        or (x1 < width and np.any(crop_xs == x1 - x0 - 1))
        or (y0 > 0 and np.any(crop_ys == 0))
        or (y1 < height and np.any(crop_ys == y1 - y0 - 1))
    )
    if touches_unclipped_edge:
        return np.nonzero(binary)
    return crop_ys + y0, crop_xs + x0


def _rle_counts_from_coordinates(
    ys: np.ndarray,
    xs: np.ndarray,
    height: int,
    width: int,
) -> list[int]:
    """Build exact column-major runs without materializing a full flat mask."""
    total = int(height) * int(width)
    if total == 0:
        return []
    if xs.size == 0:
        return [total]

    positions = np.sort(
        np.asarray(xs, dtype=np.int64) * int(height)
        + np.asarray(ys, dtype=np.int64)
    )
    split_after = np.flatnonzero(np.diff(positions) != 1)
    starts = positions[np.concatenate(([0], split_after + 1))]
    ends = positions[np.concatenate((split_after, [positions.size - 1]))]

    trailing = int(total - ends[-1] - 1)
    counts = np.empty(2 * len(starts) + int(trailing > 0), dtype=np.int64)
    counts[0] = starts[0]
    counts[1:2 * len(starts):2] = ends - starts + 1
    if len(starts) > 1:
        counts[2:2 * len(starts):2] = starts[1:] - ends[:-1] - 1
    if trailing:
        counts[-1] = trailing
    return counts.astype(int, copy=False).tolist()


def _geometry_from_coordinates(
    ys: np.ndarray,
    xs: np.ndarray,
    height: int,
    width: int,
) -> dict[str, Any] | None:
    if xs.size == 0:
        return None
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    rounded_x = min(max(int(round(centroid_x)), 0), width - 1)
    rounded_y = min(max(int(round(centroid_y)), 0), height - 1)
    occupied = bool(np.any((xs == rounded_x) & (ys == rounded_y)))
    return {
        'area_px': int(xs.size),
        'bbox_xyxy_px': [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        'centroid_px': [centroid_x, centroid_y],
        'centroid_inside_mask': occupied,
    }


def mask_to_compressed_coco_rle_with_geometry(
    mask: np.ndarray,
    bbox_xyxy_px: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Encode one mask and derive its geometry from the same sparse scan."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape={binary.shape}')
    height, width = binary.shape
    ys, xs = _mask_coordinates(binary, bbox_xyxy_px)
    counts = _rle_counts_from_coordinates(ys, xs, height, width)
    rle = {
        'size': [int(height), int(width)],
        'counts': compress_coco_rle_counts(counts),
    }
    return rle, _geometry_from_coordinates(ys, xs, height, width)


def mask_to_compressed_coco_rle(
    mask: np.ndarray,
    bbox_xyxy_px: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable compressed COCO RLE object."""
    rle, _ = mask_to_compressed_coco_rle_with_geometry(
        mask, bbox_xyxy_px
    )
    return rle


def mask_geometry(mask: np.ndarray) -> dict[str, Any] | None:
    """Calculate area, bounding box, and centroid from a binary mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape={binary.shape}')

    ys, xs = np.nonzero(binary)
    return _geometry_from_coordinates(
        ys, xs, int(binary.shape[0]), int(binary.shape[1])
    )


def build_instance_observation(
    *,
    model_class_index: int,
    confidence: float,
    detection_bbox_xyxy: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    """Build one version-2 2-D tool observation."""
    if not 0 <= model_class_index < len(CAM4_CLASS_NAMES):
        raise ValueError(f'unknown model class index: {model_class_index}')

    segmentation, geometry = mask_to_compressed_coco_rle_with_geometry(
        mask,
        tuple(float(value) for value in detection_bbox_xyxy),
    )
    if geometry is None:
        return None

    height, width = np.asarray(mask).shape
    centroid_x, centroid_y = geometry['centroid_px']
    bbox = [float(value) for value in detection_bbox_xyxy]
    return {
        'canonical_class_id': model_class_index + 1,
        'model_class_index': model_class_index,
        'class_name': CAM4_CLASS_NAMES[model_class_index],
        'class_confidence': float(confidence),
        'bbox_xyxy_px': bbox,
        'mask_bbox_xyxy_px': geometry['bbox_xyxy_px'],
        'segmentation': {
            'encoding': 'coco_rle_compressed',
            'size_hw': [height, width],
            'counts': segmentation['counts'],
        },
        'mask_area_px': geometry['area_px'],
        'mask_centroid_uv_px': [centroid_x, centroid_y],
        'mask_centroid_uv_norm': [centroid_x / width, centroid_y / height],
        'centroid_inside_mask': geometry['centroid_inside_mask'],
        'pose_mode': 'invalid_2d_only',
    }
