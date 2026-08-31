"""Low-latency RF-DETR TensorRT prediction with lossless early filtering."""

from __future__ import annotations

from types import SimpleNamespace
import time
from typing import Any

import numpy as np


def thresholded_segmentation_postprocess(
    *,
    output_boxes: Any,
    output_logits: Any,
    output_masks: Any,
    original_sizes: list[tuple[int, int]],
    threshold: float,
    postprocessor: Any,
) -> list[SimpleNamespace]:
    """Match RF-DETR postprocessing while resizing only accepted masks.

    RF-DETR's stock segmentation postprocessor resizes all 300 top-K masks to
    source resolution and applies the confidence threshold afterwards. Mask
    interpolation is independent per query, so applying the same strict
    threshold before gather/resize is pixel-identical for retained detections.
    """
    import torch
    import torch.nn.functional as functional

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    batch_size = int(output_logits.shape[0])
    if len(original_sizes) != batch_size:
        raise ValueError("original_sizes must match the output batch")
    target_sizes = torch.as_tensor(
        original_sizes,
        device=output_logits.device,
        dtype=torch.int64,
    )
    scores, labels, topk_boxes = postprocessor._select_topk(output_logits)
    boxes = postprocessor._gather_and_scale_boxes(
        output_boxes,
        topk_boxes,
        target_sizes,
    )

    predictions: list[SimpleNamespace] = []
    mask_height = int(output_masks.shape[-2])
    mask_width = int(output_masks.shape[-1])
    for index, (height, width) in enumerate(original_sizes):
        keep = scores[index] > float(threshold)
        kept_scores = scores[index][keep]
        kept_labels = labels[index][keep]
        kept_boxes = boxes[index][keep]
        kept_queries = topk_boxes[index][keep]
        if int(kept_queries.numel()):
            selected_masks = torch.gather(
                output_masks[index],
                0,
                kept_queries[:, None, None].repeat(
                    1, mask_height, mask_width
                ),
            )
            resized_masks = functional.interpolate(
                selected_masks.unsqueeze(1),
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1) > 0.0
        else:
            resized_masks = torch.empty(
                (0, int(height), int(width)),
                device=output_masks.device,
                dtype=torch.bool,
            )
        predictions.append(SimpleNamespace(
            xyxy=kept_boxes.float().cpu().numpy(),
            class_id=kept_labels.cpu().numpy(),
            confidence=kept_scores.float().cpu().numpy(),
            mask=resized_masks.cpu().numpy(),
        ))
    return predictions


def predict_thresholded_masks(
    model: Any,
    engine_runner: Any,
    images: list[np.ndarray],
    threshold: float,
) -> tuple[Any, dict[str, float]]:
    """Run the deployed RF-DETR contract without resizing rejected masks."""
    import torch
    from torchvision.transforms import functional

    if not images:
        raise ValueError("at least one image is required")
    preprocess_started = time.perf_counter()
    preprocess_gpu_started = torch.cuda.Event(enable_timing=True)
    preprocess_gpu_completed = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    preprocess_gpu_started.record(current_stream)
    original_sizes: list[tuple[int, int]] = []
    processed = []
    for image in images:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("RF-DETR image must be uint8 HxWx3")
        tensor = functional.to_tensor(array)
        original_sizes.append((int(tensor.shape[1]), int(tensor.shape[2])))
        processed.append(tensor.to(model.model.device))
    resolution = int(model.model.resolution)
    batch_tensor = torch.stack([
        functional.resize(tensor, [resolution, resolution])
        for tensor in processed
    ])
    batch_tensor = functional.normalize(batch_tensor, model.means, model.stds)
    preprocess_gpu_completed.record(current_stream)
    preprocess_wall_ms = (
        time.perf_counter() - preprocess_started
    ) * 1000.0

    raw_outputs = engine_runner(batch_tensor)
    preprocess_gpu_ms = float(
        preprocess_gpu_started.elapsed_time(preprocess_gpu_completed)
    )
    if len(raw_outputs) != 3:
        raise RuntimeError(
            f"expected three RF-DETR outputs, received {len(raw_outputs)}"
        )

    postprocess_started = time.perf_counter()
    predictions = thresholded_segmentation_postprocess(
        output_boxes=raw_outputs[0],
        output_logits=raw_outputs[1],
        output_masks=raw_outputs[2],
        original_sizes=original_sizes,
        threshold=threshold,
        postprocessor=model.model.postprocess,
    )
    postprocess_wall_ms = (
        time.perf_counter() - postprocess_started
    ) * 1000.0
    diagnostics = {
        "preprocess_submit_wall_ms": preprocess_wall_ms,
        "preprocess_gpu_ms": preprocess_gpu_ms,
        "thresholded_postprocess_wall_ms": postprocess_wall_ms,
    }
    return (
        predictions[0] if len(predictions) == 1 else predictions,
        diagnostics,
    )
