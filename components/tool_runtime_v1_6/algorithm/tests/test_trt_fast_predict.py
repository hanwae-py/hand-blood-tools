from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from pnu_surgical_tool.trt_fast_predict import (  # noqa: E402
    thresholded_segmentation_postprocess,
)


class _Postprocessor:
    num_select = 7

    def _select_topk(self, logits):
        probability = logits.sigmoid()
        flattened = probability.view(logits.shape[0], -1)
        scores, indexes = torch.topk(flattened, self.num_select, dim=1)
        return scores, indexes % logits.shape[2], indexes // logits.shape[2]

    @staticmethod
    def _gather_and_scale_boxes(boxes, queries, target_sizes):
        cx, cy, width, height = boxes.unbind(-1)
        xyxy = torch.stack(
            (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
            dim=-1,
        )
        selected = torch.gather(xyxy, 1, queries[..., None].repeat(1, 1, 4))
        image_height, image_width = target_sizes.unbind(1)
        scale = torch.stack(
            (image_width, image_height, image_width, image_height), dim=1
        ).to(selected.dtype)
        return (selected * scale[:, None]).clamp_min(0).clamp(
            max=scale[:, None]
        )


def test_early_threshold_masks_match_interpolate_then_filter() -> None:
    functional = torch.nn.functional
    generator = torch.Generator().manual_seed(5020)
    batch, queries, classes = 2, 11, 4
    output_boxes = torch.rand(batch, queries, 4, generator=generator)
    output_logits = torch.randn(batch, queries, classes, generator=generator)
    output_masks = torch.randn(batch, queries, 9, 13, generator=generator)
    sizes = [(31, 47), (25, 39)]
    threshold = 0.58
    postprocessor = _Postprocessor()

    optimized = thresholded_segmentation_postprocess(
        output_boxes=output_boxes,
        output_logits=output_logits,
        output_masks=output_masks,
        original_sizes=sizes,
        threshold=threshold,
        postprocessor=postprocessor,
    )
    scores, labels, selected_queries = postprocessor._select_topk(output_logits)
    target_sizes = torch.tensor(sizes)
    boxes = postprocessor._gather_and_scale_boxes(
        output_boxes, selected_queries, target_sizes
    )
    for index, (height, width) in enumerate(sizes):
        gathered = torch.gather(
            output_masks[index],
            0,
            selected_queries[index, :, None, None].repeat(
                1, output_masks.shape[-2], output_masks.shape[-1]
            ),
        )
        all_resized = functional.interpolate(
            gathered[:, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1) > 0
        keep = scores[index] > threshold
        np.testing.assert_array_equal(
            optimized[index].mask,
            all_resized[keep].numpy(),
        )
        np.testing.assert_array_equal(
            optimized[index].xyxy,
            boxes[index][keep].float().numpy(),
        )
        np.testing.assert_array_equal(
            optimized[index].class_id,
            labels[index][keep].numpy(),
        )
        np.testing.assert_array_equal(
            optimized[index].confidence,
            scores[index][keep].float().numpy(),
        )
