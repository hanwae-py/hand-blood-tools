#!/usr/bin/env python3
"""Replay COCO mask centroids through the control-facing position filter.

This is a 2-D proxy test for RGB-only evaluation archives. It converts pixel
motion to metric motion using an explicitly supplied metres-per-pixel scale;
it does not claim to reproduce RGB-D pose or camera deprojection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def _package_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / 'ros2_ws'
        / 'src'
        / 'pnu_surgical_perception'
    )


sys.path.insert(0, str(_package_root()))

from pnu_surgical_perception.tool_position_stabilizer import (  # noqa: E402
    ToolPositionStabilizer,
)


def _mask_centroid(annotation: dict[str, Any]) -> tuple[float, float]:
    segmentation = dict(annotation['segmentation'])
    if isinstance(segmentation.get('counts'), str):
        segmentation['counts'] = segmentation['counts'].encode('ascii')
    mask = np.asarray(mask_utils.decode(segmentation), dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size:
        return float(xs.mean()), float(ys.mean())
    x, y, width, height = (float(value) for value in annotation['bbox'])
    return x + width / 2.0, y + height / 2.0


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _step_summary(values_m: list[float]) -> dict[str, float | int | None]:
    return {
        'count': len(values_m),
        'zero_count': sum(value <= 1e-12 for value in values_m),
        'median_mm': (
            _percentile(values_m, 50.0) * 1000.0 if values_m else None
        ),
        'p95_mm': (
            _percentile(values_m, 95.0) * 1000.0 if values_m else None
        ),
        'maximum_mm': max(values_m) * 1000.0 if values_m else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    source_bytes = args.predictions.read_bytes()
    payload = json.loads(source_bytes)
    category_names = {
        int(item['id']): str(item['name']) for item in payload['categories']
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload['annotations']:
        if float(annotation.get('confidence', 1.0)) >= args.confidence:
            annotations_by_image[int(annotation['image_id'])].append(annotation)

    stabilizer = ToolPositionStabilizer(
        enabled=True,
        deadband_m=args.deadband_m,
        smoothing_alpha=args.smoothing_alpha,
        max_jump_m=args.max_jump_m,
        relocation_confirmation_frames=args.relocation_confirmation_frames,
        relocation_consistency_m=args.relocation_consistency_m,
        max_missed_frames=args.max_missed_frames,
    )
    previous_raw: dict[str, tuple[int, tuple[float, float, float]]] = {}
    previous_filtered: dict[str, tuple[int, tuple[float, float, float]]] = {}
    raw_steps: list[float] = []
    filtered_steps: list[float] = []
    raw_moderate_steps: list[float] = []
    filtered_moderate_steps: list[float] = []
    raw_steady_steps: list[float] = []
    filtered_steady_steps: list[float] = []
    reason_counts: dict[str, int] = defaultdict(int)
    per_class_observations: dict[str, int] = defaultdict(int)
    observation_count = 0
    previous_active_keys: set[str] = set()

    for frame_index, image in enumerate(
        sorted(payload['images'], key=lambda item: int(item['id']))
    ):
        groups: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
        for annotation in annotations_by_image.get(int(image['id']), []):
            category_id = int(annotation['category_id'])
            class_name = category_names.get(category_id, f'class_{category_id}')
            groups[(category_id, class_name)].append(
                _mask_centroid(annotation)
            )

        rows: list[tuple[str, str, tuple[float, float, float]]] = []
        for (_category_id, class_name), centroids in groups.items():
            for ordinal, (u_px, v_px) in enumerate(
                sorted(centroids), start=1
            ):
                key = f'{class_name}#{ordinal}'
                position = (
                    u_px * args.meters_per_pixel,
                    v_px * args.meters_per_pixel,
                    0.0,
                )
                rows.append((key, class_name, position))

        active_keys = {key for key, _class_name, _position in rows}
        previous_by_class: dict[str, set[str]] = defaultdict(set)
        current_by_class: dict[str, set[str]] = defaultdict(set)
        for key in previous_active_keys:
            previous_by_class[key.rsplit('#', 1)[0]].add(key)
        for key in active_keys:
            current_by_class[key.rsplit('#', 1)[0]].add(key)
        for class_name in previous_by_class.keys() | current_by_class.keys():
            old_group = previous_by_class.get(class_name, set())
            new_group = current_by_class.get(class_name, set())
            if old_group != new_group:
                stabilizer.reset_keys(old_group | new_group)

        for key, class_name, position in rows:
            decision = stabilizer.update(key, position)
            reason_counts[decision.reason] += 1
            per_class_observations[class_name] += 1
            observation_count += 1

            raw_previous = previous_raw.get(key)
            filtered_previous = previous_filtered.get(key)
            if (
                raw_previous is not None
                and filtered_previous is not None
                and raw_previous[0] == frame_index - 1
                and filtered_previous[0] == frame_index - 1
            ):
                raw_step = math.dist(position, raw_previous[1])
                filtered_step = math.dist(
                    decision.position_m, filtered_previous[1]
                )
                raw_steps.append(raw_step)
                filtered_steps.append(filtered_step)
                if raw_step <= args.max_jump_m:
                    raw_moderate_steps.append(raw_step)
                    filtered_moderate_steps.append(filtered_step)
                    if decision.reason in (
                        'DEADBAND_HELD',
                        'EMA_SMOOTHED',
                    ):
                        raw_steady_steps.append(raw_step)
                        filtered_steady_steps.append(filtered_step)
            previous_raw[key] = (frame_index, position)
            previous_filtered[key] = (frame_index, decision.position_m)
        stabilizer.finish_frame(active_keys)
        previous_active_keys = active_keys

    raw_summary = _step_summary(raw_steps)
    filtered_summary = _step_summary(filtered_steps)
    raw_moderate_summary = _step_summary(raw_moderate_steps)
    filtered_moderate_summary = _step_summary(filtered_moderate_steps)
    raw_steady_summary = _step_summary(raw_steady_steps)
    filtered_steady_summary = _step_summary(filtered_steady_steps)
    raw_p95 = raw_moderate_summary['p95_mm']
    filtered_p95 = filtered_moderate_summary['p95_mm']
    p95_reduction = None
    if (
        isinstance(raw_p95, float)
        and isinstance(filtered_p95, float)
        and raw_p95 > 0.0
    ):
        p95_reduction = 100.0 * (1.0 - filtered_p95 / raw_p95)

    return {
        'schema': 'pnu.tool_position_stabilizer_rgb_proxy_evaluation.v1',
        'limitations': [
            'RGB-only archive: mask centroids are used instead of RGB-D pose',
            'meters_per_pixel is a control-team approximation, not calibration',
            'frozen predictions are not human ground truth',
        ],
        'source': {
            'predictions': str(args.predictions),
            'sha256': hashlib.sha256(source_bytes).hexdigest(),
            'frame_count': len(payload['images']),
            'annotation_count_at_threshold': observation_count,
            'confidence_threshold': args.confidence,
        },
        'proxy_conversion': {
            'meters_per_pixel': args.meters_per_pixel,
            'position': '[mask_centroid_u*m_per_px, mask_centroid_v*m_per_px, 0]',
            'association': 'class_then_left_to_right_ordinal_per_frame',
        },
        'filter': {
            'deadband_m': args.deadband_m,
            'smoothing_alpha': args.smoothing_alpha,
            'max_jump_m': args.max_jump_m,
            'relocation_confirmation_frames': (
                args.relocation_confirmation_frames
            ),
            'relocation_consistency_m': args.relocation_consistency_m,
            'max_missed_frames': args.max_missed_frames,
        },
        'results': {
            'all_consecutive_steps_raw': raw_summary,
            'all_consecutive_steps_filtered': filtered_summary,
            'steps_at_or_below_max_jump_raw': raw_moderate_summary,
            'steps_at_or_below_max_jump_filtered': filtered_moderate_summary,
            'steady_state_steps_raw': raw_steady_summary,
            'steady_state_steps_filtered': filtered_steady_summary,
            'moderate_step_p95_reduction_percent': p95_reduction,
            'decision_counts': dict(sorted(reason_counts.items())),
            'observations_by_class': dict(
                sorted(per_class_observations.items())
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confidence', type=float, default=0.3)
    parser.add_argument('--meters-per-pixel', type=float, default=0.0045)
    parser.add_argument('--deadband-m', type=float, default=0.0)
    parser.add_argument('--smoothing-alpha', type=float, default=0.20)
    parser.add_argument('--max-jump-m', type=float, default=0.04)
    parser.add_argument(
        '--relocation-confirmation-frames', type=int, default=2
    )
    parser.add_argument(
        '--relocation-consistency-m', type=float, default=0.015
    )
    parser.add_argument('--max-missed-frames', type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
