"""Dependency-free COCO uncompressed RLE utilities for binary masks."""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_uncompressed_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be HxW")
    flat = binary.reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for raw in flat:
        current = int(raw != 0)
        if current == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = current
    counts.append(run_length)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def decode_uncompressed_coco_rle(segmentation: dict[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in segmentation["size"])
    counts = segmentation["counts"]
    if not isinstance(counts, list):
        raise TypeError("Only uncompressed list-form COCO RLE is supported")
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    value = 0
    for count in counts:
        next_cursor = cursor + int(count)
        if next_cursor > len(flat):
            raise ValueError("RLE exceeds declared mask size")
        if value:
            flat[cursor:next_cursor] = 1
        cursor = next_cursor
        value = 1 - value
    if cursor != len(flat):
        raise ValueError("RLE does not fill declared mask size")
    return flat.reshape((height, width), order="F").astype(bool)

