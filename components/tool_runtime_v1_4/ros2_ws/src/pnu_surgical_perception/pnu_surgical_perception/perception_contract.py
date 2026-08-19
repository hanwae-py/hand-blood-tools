"""Pure helpers for the surgical-tool observation JSON contract."""

from __future__ import annotations

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


def mask_to_compressed_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """Return a JSON-serializable compressed COCO RLE object."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape={binary.shape}')
    return {
        'size': [int(binary.shape[0]), int(binary.shape[1])],
        'counts': compress_coco_rle_counts(mask_to_rle_counts(binary)),
    }


def mask_geometry(mask: np.ndarray) -> dict[str, Any] | None:
    """Calculate area, bounding box, and centroid from a binary mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape={binary.shape}')

    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None

    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    rounded_x = min(max(int(round(centroid_x)), 0), binary.shape[1] - 1)
    rounded_y = min(max(int(round(centroid_y)), 0), binary.shape[0] - 1)
    return {
        'area_px': int(xs.size),
        'bbox_xyxy_px': [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        'centroid_px': [centroid_x, centroid_y],
        'centroid_inside_mask': bool(binary[rounded_y, rounded_x]),
    }


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

    geometry = mask_geometry(mask)
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
            'counts': mask_to_compressed_coco_rle(mask)['counts'],
        },
        'mask_area_px': geometry['area_px'],
        'mask_centroid_uv_px': [centroid_x, centroid_y],
        'mask_centroid_uv_norm': [centroid_x / width, centroid_y / height],
        'centroid_inside_mask': geometry['centroid_inside_mask'],
        'pose_mode': 'invalid_2d_only',
    }
