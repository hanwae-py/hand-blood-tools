#!/usr/bin/env python3
"""Generate frozen COCO mask predictions from an RF-DETR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from pnu_surgical_tool import DetectorConfig, SurgicalToolDetector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_mask(mask: np.ndarray) -> tuple[dict[str, Any], list[float], float]:
    encoded = mask_utils.encode(
        np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    )
    segmentation = {
        'size': [int(value) for value in encoded['size']],
        'counts': encoded['counts'].decode('ascii'),
    }
    bbox = [float(value) for value in mask_utils.toBbox(encoded)]
    area = float(mask_utils.area(encoded))
    return segmentation, bbox, area


def predict(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.source_coco.read_text(encoding='utf-8'))
    images = sorted(source['images'], key=lambda item: int(item['id']))
    if args.max_frames is not None:
        images = images[:args.max_frames]
    category_names = {
        int(item['id']): str(item['name']) for item in source['categories']
    }
    checkpoint_hash = _sha256(args.checkpoint)
    detector = SurgicalToolDetector(
        DetectorConfig(
            checkpoint_path=args.checkpoint,
            ontology_path=args.ontology,
            model_size=args.model_size,
            confidence_threshold=args.threshold,
            optimize=args.optimize,
        )
    )
    detector.load()

    annotations: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    started = time.perf_counter()
    for frame_index, image_record in enumerate(images):
        image_path = args.image_root / str(image_record['file_name'])
        encoded_image = image_path.read_bytes()
        expected_hash = str(image_record.get('source_sha256', ''))
        actual_hash = hashlib.sha256(encoded_image).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f'image hash mismatch: {image_path}')
        bgr = cv2.imdecode(
            np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None:
            raise RuntimeError(f'could not decode {image_path}')
        batch = detector.predict(bgr, color_order='BGR')
        if batch.inference_latency_ms is not None:
            latencies_ms.append(float(batch.inference_latency_ms))
        for instance in batch.instances:
            category_id = int(instance.model_class_index)
            expected_name = category_names.get(category_id)
            if expected_name != instance.class_name:
                raise RuntimeError(
                    f'category mismatch: {category_id} {instance.class_name}'
                )
            segmentation, bbox, area = _encode_mask(instance.mask)
            if area < 1.0:
                continue
            annotations.append(
                {
                    'id': len(annotations) + 1,
                    'image_id': int(image_record['id']),
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0,
                    'segmentation': segmentation,
                    'confidence': float(instance.class_confidence),
                    'proposal_source': args.proposal_source,
                    'checkpoint_sha256': checkpoint_hash,
                    'human_verified': False,
                }
            )
        completed = frame_index + 1
        if completed % 25 == 0 or completed == len(images):
            print(
                f'predicted {completed}/{len(images)} frames in '
                f'{time.perf_counter() - started:.1f}s',
                flush=True,
            )

    payload = {
        'info': {
            'description': 'Frozen RF-DETR segmentation predictions',
            'ground_truth_status': 'model_predictions_not_ground_truth',
            'checkpoint': str(args.checkpoint),
            'checkpoint_sha256': checkpoint_hash,
            'model_size': args.model_size,
            'checkpoint_color_order': detector.checkpoint_color_order,
            'confidence_threshold': args.threshold,
            'optimized': args.optimize,
            'frame_count': len(images),
            'annotation_count': len(annotations),
            'wall_duration_sec': time.perf_counter() - started,
            'inference_latency_ms_mean': (
                float(np.mean(latencies_ms)) if latencies_ms else None
            ),
            'inference_latency_ms_p95': (
                float(np.percentile(latencies_ms, 95.0))
                if latencies_ms
                else None
            ),
        },
        'licenses': source.get('licenses', []),
        'categories': source['categories'],
        'images': images,
        'annotations': annotations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    return payload['info']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ontology', type=Path, required=True)
    parser.add_argument('--model-size', required=True)
    parser.add_argument('--source-coco', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--proposal-source', default='rfdetr_frozen_evaluation')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--max-frames', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.checkpoint,
        args.ontology,
        args.source_coco,
        args.image_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    print(json.dumps(predict(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
