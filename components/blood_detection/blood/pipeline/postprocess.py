"""Connected-component filtering and per-region centroids."""

from __future__ import annotations

import cv2
import numpy as np


def filter_components(mask: np.ndarray, min_area: int = 400) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == i
    return keep


def region_centroids(mask: np.ndarray, min_area: int = 400) -> list[dict]:
    binary = filter_components(mask, min_area=min_area)
    if not binary.any():
        return []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        regions.append(
            {
                "centroid_xy": [float(cents[i][0]), float(cents[i][1])],
                "area": area,
                "bbox_xywh": [int(v) for v in stats[i, :4]],
                "label": i,
            }
        )
    return regions
