#!/usr/bin/env python3
"""Evaluate control-facing position stabilization with synchronized MCAP RGB-D.

The detector output is supplied as frozen COCO instance predictions.  Every
prediction frame is verified against the compressed RGB bytes in the MCAP,
then native depth is registered to RGB before the current constrained planar
pose estimator and position stabilizer are applied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from rosbags.highlevel import AnyReader
import yaml


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / 'ros2_ws'
    / 'src'
    / 'pnu_surgical_perception'
)
sys.path.insert(0, str(ALGORITHM_ROOT / 'src'))
sys.path.insert(0, str(ROS_PACKAGE_ROOT))

from pnu_surgical_perception.tool_position_stabilizer import (  # noqa: E402
    ToolPositionStabilizer,
)
from pnu_surgical_tool import (  # noqa: E402
    CameraCalibration,
    decode_compressed_depth_16uc1,
    DepthToColorRegistrar,
    DetectionBatch,
    DetectionInstance,
    DetectionPostprocessor,
    DetectionPostprocessorConfig,
    PlanarPoseConfig,
    PlanarPoseEstimator,
    RigidTransform,
    SupportPlane,
    TemporalClassConfig,
    WorkspaceRoiConfig,
)
from pnu_surgical_tool.visualization import (  # noqa: E402
    draw_detections_bgr,
)


SHORT_TOOL_NAMES = {
    'Scalpel': 'scalpel',
    'Allis Forceps': 'allis',
    'Mosquito': 'mosquito',
    'Adson Forceps': 'adson',
    'Bipolar Forceps': 'bipolar',
    'Bovie': 'bovie',
    'Army-Navy Retractor': 'army',
    'Thyroid Retractor': 'thyroid',
}

POSTPROCESS_TOTAL_KEYS = (
    'input_instances',
    'class_confidence_rejected_instances',
    'roi_rejected_instances',
    'output_instances',
    'tracks_created',
    'tracks_matched',
    'raw_class_transitions',
    'class_overrides',
    'class_switches',
)


def _parameters(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8'))['/**'][
        'ros__parameters'
    ]


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _calibration(message: Any, version: str) -> CameraCalibration:
    return CameraCalibration(
        width=int(message.width),
        height=int(message.height),
        k=np.asarray(message.k, dtype=np.float64).reshape(3, 3),
        distortion=np.asarray(message.d, dtype=np.float64),
        frame_name=str(message.header.frame_id),
        calibration_version=version,
    )


def _read_calibration(
    bag: Path, camera: str, version: str
) -> tuple[CameraCalibration, CameraCalibration, RigidTransform]:
    prefix = f'/synced/{camera}'
    names = {
        f'{prefix}/color/camera_info': 'color',
        f'{prefix}/depth/camera_info': 'depth',
        f'{prefix}/extrinsics/depth_to_color': 'extrinsics',
    }
    found: dict[str, Any] = {}
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic in names]
        for connection, _timestamp, raw in reader.messages(
            connections=connections
        ):
            key = names[connection.topic]
            if key not in found:
                found[key] = reader.deserialize(raw, connection.msgtype)
            if len(found) == 3:
                break
    missing = {'color', 'depth', 'extrinsics'} - found.keys()
    if missing:
        raise RuntimeError(
            f'MCAP calibration topics missing: {sorted(missing)}'
        )
    color = _calibration(found['color'], f'{version}:color')
    depth = _calibration(found['depth'], f'{version}:depth')
    extrinsics = found['extrinsics']
    # realsense2_camera_msgs/Extrinsics.rotation is column-major.
    rotation = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(
        3, 3, order='F'
    )
    transform = RigidTransform(
        rotation=rotation,
        translation_m=np.asarray(extrinsics.translation, dtype=np.float64),
        source_frame=depth.frame_name,
        target_frame=color.frame_name,
        calibration_version=f'{version}:depth_to_color_topic',
    )
    return color, depth, transform


def _decode_mask(annotation: dict[str, Any]) -> np.ndarray:
    segmentation = dict(annotation['segmentation'])
    if isinstance(segmentation.get('counts'), str):
        segmentation['counts'] = segmentation['counts'].encode('ascii')
    return np.asarray(mask_utils.decode(segmentation), dtype=bool)


def _postprocessor(parameters: dict[str, Any]) -> DetectionPostprocessor:
    return DetectionPostprocessor(
        DetectionPostprocessorConfig(
            workspace_roi=WorkspaceRoiConfig(
                enabled=bool(parameters['workspace_roi_enabled']),
                polygon_norm_xy=tuple(
                    float(value)
                    for value in parameters[
                        'workspace_roi_polygon_norm_xy'
                    ]
                ),
                minimum_mask_overlap=float(
                    parameters['workspace_roi_minimum_mask_overlap']
                ),
                require_mask_centroid_inside=bool(
                    parameters[
                        'workspace_roi_require_mask_centroid_inside'
                    ]
                ),
            ),
            temporal_class=TemporalClassConfig(
                enabled=True,
                history_size=7,
                minimum_switch_frames=3,
                switch_score_margin=0.2,
                minimum_mask_iou=0.1,
                minimum_bbox_iou=0.2,
                maximum_centroid_distance_norm=0.06,
                maximum_mask_area_ratio=3.0,
                max_missed_frames=3,
            ),
        )
    )


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _step_summary(values_m: list[float]) -> dict[str, float | int | None]:
    return {
        'count': len(values_m),
        'zero_count': sum(value <= 1e-12 for value in values_m),
        'median_mm': (
            1000.0 * _percentile(values_m, 50.0) if values_m else None
        ),
        'p95_mm': (
            1000.0 * _percentile(values_m, 95.0) if values_m else None
        ),
        'maximum_mm': 1000.0 * max(values_m) if values_m else None,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _project_position(
    position_m: tuple[float, float, float], camera: CameraCalibration
) -> tuple[int, int] | None:
    point = np.asarray(position_m, dtype=np.float64).reshape(1, 1, 3)
    if not np.all(np.isfinite(point)) or point[0, 0, 2] <= 0.0:
        return None
    projected, _jacobian = cv2.projectPoints(
        point,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera.k,
        camera.distortion,
    )
    u_px, v_px = projected.reshape(2)
    if not np.isfinite(u_px) or not np.isfinite(v_px):
        return None
    return int(round(float(u_px))), int(round(float(v_px)))


def _draw_position_panel(
    image: np.ndarray,
    title: str,
    rows: list[tuple[str, tuple[float, float, float], str]],
    trails: dict[str, deque[tuple[int, int]]],
    camera: CameraCalibration,
    color: tuple[int, int, int],
) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (18, 18, 18), -1)
    cv2.putText(
        output,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    for slot, position, _reason in rows:
        pixel = _project_position(position, camera)
        if pixel is None:
            continue
        trail = trails.setdefault(slot, deque(maxlen=15))
        trail.append(pixel)
        if len(trail) >= 2:
            cv2.polylines(
                output,
                [np.asarray(trail, dtype=np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.drawMarker(
            output,
            pixel,
            color,
            cv2.MARKER_CROSS,
            20,
            3,
            cv2.LINE_AA,
        )
    return output


def _load_predictions(
    path: Path, ontology_path: Path, confidence: float
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, tuple[int, str]],
    str,
    dict[str, Any],
]:
    source = path.read_bytes()
    payload = json.loads(source)
    ontology = json.loads(ontology_path.read_text(encoding='utf-8'))
    canonical_by_name = {
        str(item['canonical_name']): int(item['canonical_id'])
        for item in ontology['canonical_tool_classes']
    }
    categories = {
        int(item['id']): (
            canonical_by_name[str(item['name'])],
            str(item['name']),
        )
        for item in payload['categories']
    }
    images = {int(item['ros_stamp_ns']): item for item in payload['images']}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload['annotations']:
        if float(annotation.get('confidence', 1.0)) >= confidence:
            annotations[int(annotation['image_id'])].append(annotation)
    return (
        images,
        annotations,
        categories,
        hashlib.sha256(source).hexdigest(),
        dict(payload.get('info', {})),
    )


def _detection_batch(
    image: dict[str, Any],
    annotations: dict[int, list[dict[str, Any]]],
    categories: dict[int, tuple[int, str]],
) -> DetectionBatch:
    instances: list[DetectionInstance] = []
    for local_id, annotation in enumerate(
        annotations.get(int(image['id']), []), start=1
    ):
        category_id = int(annotation['category_id'])
        canonical_id, class_name = categories[category_id]
        x, y, width, height = (
            float(value) for value in annotation['bbox']
        )
        instances.append(
            DetectionInstance(
                frame_local_instance_id=local_id,
                canonical_class_id=canonical_id,
                model_class_index=category_id,
                class_name=class_name,
                class_confidence=float(annotation.get('confidence', 1.0)),
                bbox_xyxy_px=(x, y, x + width, y + height),
                mask=_decode_mask(annotation),
            )
        )
    checkpoint_hashes = {
        str(item.get('checkpoint_sha256', 'unknown'))
        for item in annotations.get(int(image['id']), [])
    }
    model_version = ','.join(sorted(checkpoint_hashes))
    return DetectionBatch(
        image_width=int(image['width']),
        image_height=int(image['height']),
        model_version=f'frozen_coco:{model_version}',
        ontology_version='pnu.cam4.tool_ontology.v1',
        instances=instances,
    )


def _slot_rows(instances: list[Any]) -> list[tuple[str, Any]]:
    groups: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for item in instances:
        groups[(int(item.canonical_class_id), item.class_name)].append(item)
    rows: list[tuple[str, Any]] = []
    for (_canonical_id, class_name), group in groups.items():
        ordered = sorted(
            group,
            key=lambda item: (
                (float(item.bbox_xyxy_px[0]) + float(item.bbox_xyxy_px[2]))
                / 2.0,
                int(item.frame_local_instance_id),
            ),
        )
        short_name = SHORT_TOOL_NAMES.get(
            class_name, class_name.lower().replace(' ', '_')
        )
        rows.extend(
            (f'mayo_{short_name}#{ordinal}', item)
            for ordinal, item in enumerate(ordered, start=1)
        )
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    pose_parameters = _parameters(args.pose_config)
    roi_parameters = _parameters(args.roi_profile)
    (
        images,
        annotations,
        categories,
        predictions_hash,
        predictions_info,
    ) = _load_predictions(args.predictions, args.ontology, args.confidence)
    calibration_version = str(pose_parameters['calibration_version'])
    color_camera, depth_camera, transform = _read_calibration(
        args.bag, args.camera, calibration_version
    )
    registrar = DepthToColorRegistrar(
        depth_camera, color_camera, transform, backend='numpy'
    )
    support_plane = SupportPlane(
        normal=np.asarray(
            pose_parameters['support_plane_normal'], dtype=np.float64
        ),
        offset_m=float(pose_parameters['support_plane_offset_m']),
        config_version=str(
            pose_parameters['support_plane_config_version']
        ),
        inlier_ratio=float(pose_parameters['support_plane_inlier_ratio']),
        residual_p95_m=float(
            pose_parameters['support_plane_residual_p95_m']
        ),
    )
    postprocessor = _postprocessor(roi_parameters)
    estimator = PlanarPoseEstimator(
        PlanarPoseConfig(
            positive_y_image_direction=str(
                pose_parameters['positive_y_image_direction']
            ),
            adson_face_on_width_enabled=bool(
                pose_parameters['adson_face_on_width_enabled']
            ),
        )
    )
    stabilizer = ToolPositionStabilizer(
        enabled=True,
        deadband_m=args.deadband_m,
        smoothing_alpha=args.smoothing_alpha,
        max_jump_m=args.max_jump_m,
        relocation_confirmation_frames=args.relocation_confirmation_frames,
        relocation_consistency_m=args.relocation_consistency_m,
        max_missed_frames=args.max_missed_frames,
    )

    prefix = f'/synced/{args.camera}'
    color_topic = f'{prefix}/color/image_raw/compressed'
    depth_topic = f'{prefix}/depth/image_rect_raw/compressedDepth'
    pending_color: dict[int, Any] = {}
    pending_depth: dict[int, Any] = {}
    previous_slots: set[str] = set()
    previous_raw: dict[str, tuple[int, tuple[float, float, float]]] = {}
    previous_filtered: dict[
        str, tuple[int, tuple[float, float, float]]
    ] = {}
    raw_steps: list[float] = []
    filtered_steps: list[float] = []
    raw_steady_steps: list[float] = []
    filtered_steady_steps: list[float] = []
    pair_deltas_ns: list[int] = []
    registration_ms: list[float] = []
    aligned_valid_ratios: list[float] = []
    pose_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    postprocess_totals: Counter[str] = Counter()
    verified_rgb_hashes = 0
    paired = 0
    started = time.perf_counter()
    video_writer = None
    raw_trails: dict[str, deque[tuple[int, int]]] = {}
    filtered_trails: dict[str, deque[tuple[int, int]]] = {}

    def process_pair(color_message: Any, depth_message: Any) -> None:
        nonlocal paired, previous_slots, verified_rgb_hashes, video_writer
        color_stamp = _stamp_ns(color_message)
        image_record = images.get(color_stamp)
        if image_record is None:
            return
        encoded_rgb = bytes(color_message.data)
        actual_hash = hashlib.sha256(encoded_rgb).hexdigest()
        expected_hash = str(image_record.get('source_sha256', ''))
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                f'RGB source hash mismatch at stamp {color_stamp}'
            )
        verified_rgb_hashes += int(bool(expected_hash))
        bgr = cv2.imdecode(
            np.frombuffer(encoded_rgb, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None:
            raise RuntimeError(f'could not decode RGB at stamp {color_stamp}')
        depth = decode_compressed_depth_16uc1(
            depth_message.data, str(depth_message.format)
        )
        registration_started = time.perf_counter()
        registration = registrar.register(
            depth,
            float(pose_parameters['depth_scale_m_per_unit']),
            minimum_depth_m=float(pose_parameters['minimum_depth_m']),
            maximum_depth_m=float(pose_parameters['maximum_depth_m']),
        )
        registration_ms.append(
            (time.perf_counter() - registration_started) * 1000.0
        )
        aligned_valid_ratios.append(registration.aligned_valid_ratio)
        detections = postprocessor.process(
            _detection_batch(
                image_record, annotations, categories
            )
        )
        postprocess_totals.update(
            {
                key: int(postprocessor.last_diagnostics.get(key, 0))
                for key in POSTPROCESS_TOTAL_KEYS
            }
        )
        result = estimator.estimate(
            detections,
            registration.aligned_depth_m,
            color_camera,
            support_plane,
            frame_key=paired,
            image_bgr=bgr,
        )
        rows = _slot_rows(result.instances)
        current_slots = {slot for slot, _item in rows}
        previous_by_prefix: dict[str, set[str]] = defaultdict(set)
        current_by_prefix: dict[str, set[str]] = defaultdict(set)
        for slot in previous_slots:
            previous_by_prefix[slot.rsplit('#', 1)[0]].add(slot)
        for slot in current_slots:
            current_by_prefix[slot.rsplit('#', 1)[0]].add(slot)
        slot_prefixes = previous_by_prefix.keys() | current_by_prefix.keys()
        for slot_prefix in slot_prefixes:
            old_group = previous_by_prefix.get(slot_prefix, set())
            new_group = current_by_prefix.get(slot_prefix, set())
            if old_group != new_group:
                stabilizer.reset_keys(old_group | new_group)
                for reset_slot in old_group | new_group:
                    raw_trails.pop(reset_slot, None)
                    filtered_trails.pop(reset_slot, None)

        valid_slots: set[str] = set()
        raw_visual_rows: list[
            tuple[str, tuple[float, float, float], str]
        ] = []
        filtered_visual_rows: list[
            tuple[str, tuple[float, float, float], str]
        ] = []
        for slot, item in rows:
            class_counts[item.class_name] += 1
            pose_counts[item.validity] += 1
            if item.invalid_reason:
                invalid_reasons[item.invalid_reason] += 1
            if item.position_m is None or item.orientation_xyzw is None:
                continue
            raw = tuple(float(value) for value in item.position_m)
            decision = stabilizer.update(slot, raw)
            decision_counts[decision.reason] += 1
            valid_slots.add(slot)
            raw_visual_rows.append((slot, raw, 'RAW'))
            filtered_visual_rows.append(
                (slot, decision.position_m, decision.reason)
            )
            raw_previous = previous_raw.get(slot)
            filtered_previous = previous_filtered.get(slot)
            if (
                decision.reason != 'INITIALIZED'
                and raw_previous is not None
                and filtered_previous is not None
                and raw_previous[0] == paired - 1
                and filtered_previous[0] == paired - 1
            ):
                raw_step = math.dist(raw, raw_previous[1])
                filtered_step = math.dist(
                    decision.position_m, filtered_previous[1]
                )
                raw_steps.append(raw_step)
                filtered_steps.append(filtered_step)
                if decision.reason in ('DEADBAND_HELD', 'EMA_SMOOTHED'):
                    raw_steady_steps.append(raw_step)
                    filtered_steady_steps.append(filtered_step)
            previous_raw[slot] = (paired, raw)
            previous_filtered[slot] = (paired, decision.position_m)
        stabilizer.finish_frame(valid_slots)
        previous_slots = current_slots
        if args.comparison_video is not None:
            detection_overlay = draw_detections_bgr(bgr, detections)
            before = _draw_position_panel(
                detection_overlay,
                'BEFORE: raw ToolPoseArray translation',
                raw_visual_rows,
                raw_trails,
                color_camera,
                (30, 80, 255),
            )
            after = _draw_position_panel(
                detection_overlay,
                'AFTER: stabilized control TF translation',
                filtered_visual_rows,
                filtered_trails,
                color_camera,
                (40, 235, 70),
            )
            comparison = np.hstack((before, after))
            if args.video_scale != 1.0:
                comparison = cv2.resize(
                    comparison,
                    None,
                    fx=args.video_scale,
                    fy=args.video_scale,
                    interpolation=cv2.INTER_AREA,
                )
            if video_writer is None:
                args.comparison_video.parent.mkdir(
                    parents=True, exist_ok=True
                )
                video_writer = cv2.VideoWriter(
                    str(args.comparison_video),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    args.fps,
                    (comparison.shape[1], comparison.shape[0]),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(
                        f'could not create {args.comparison_video}'
                    )
            video_writer.write(comparison)
        paired += 1
        if paired % 25 == 0:
            print(
                f'processed {paired} RGB-D pairs in '
                f'{time.perf_counter() - started:.1f}s',
                flush=True,
            )

    def try_pair(stamp: int, own_is_color: bool) -> None:
        own = pending_color if own_is_color else pending_depth
        other = pending_depth if own_is_color else pending_color
        if not other:
            return
        nearest = min(other, key=lambda value: abs(value - stamp))
        delta = abs(nearest - stamp)
        if delta > args.maximum_delta_ns:
            return
        own_message = own.pop(stamp)
        other_message = other.pop(nearest)
        pair_deltas_ns.append(delta)
        if own_is_color:
            process_pair(own_message, other_message)
        else:
            process_pair(other_message, own_message)

    try:
        with AnyReader([args.bag]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic in (color_topic, depth_topic)
            ]
            for connection, _timestamp, raw in reader.messages(
                connections=connections
            ):
                if args.max_pairs is not None and paired >= args.max_pairs:
                    break
                message = reader.deserialize(raw, connection.msgtype)
                stamp = _stamp_ns(message)
                if connection.topic == color_topic:
                    pending_color[stamp] = message
                    try_pair(stamp, True)
                else:
                    pending_depth[stamp] = message
                    try_pair(stamp, False)
    finally:
        registrar.close()
        if video_writer is not None:
            video_writer.release()
    if paired == 0:
        raise RuntimeError('no prediction-backed synchronized RGB-D pairs')

    raw_summary = _step_summary(raw_steps)
    filtered_summary = _step_summary(filtered_steps)
    steady_raw_summary = _step_summary(raw_steady_steps)
    steady_filtered_summary = _step_summary(filtered_steady_steps)
    raw_p95 = steady_raw_summary['p95_mm']
    filtered_p95 = steady_filtered_summary['p95_mm']
    p95_reduction = None
    if isinstance(raw_p95, float) and raw_p95 > 0.0:
        p95_reduction = 100.0 * (
            1.0 - float(filtered_p95) / raw_p95
        )
    requested_count = (
        min(len(images), args.max_pairs)
        if args.max_pairs is not None
        else len(images)
    )
    return {
        'schema': 'pnu.tool_position_stabilizer_mcap_rgbd_evaluation.v1',
        'interpretation': (
            'unlabeled replay stability evaluation; not absolute pose accuracy'
        ),
        'pose_contract': 'PLANAR_4DOF_WITH_NORMAL_PRIOR',
        'not_unconstrained_6d': True,
        'limitations': [
            'frozen segmentation predictions are not human ground truth',
            'support plane and 0.001 m depth scale remain provisional',
            'scene motion is not separated from measurement jitter',
        ],
        'source': {
            'bag': str(args.bag),
            'camera': args.camera,
            'predictions': str(args.predictions),
            'predictions_sha256': predictions_hash,
            'prediction_checkpoint': predictions_info.get('checkpoint'),
            'prediction_checkpoint_sha256': predictions_info.get(
                'checkpoint_sha256'
            ),
            'prediction_model_size': predictions_info.get('model_size'),
            'prediction_optimized': predictions_info.get('optimized'),
            'prediction_frame_count': len(images),
            'requested_frame_count': requested_count,
            'processed_pair_count': paired,
            'rgb_source_hashes_verified': verified_rgb_hashes,
            'unmatched_requested_prediction_frames': max(
                requested_count - paired, 0
            ),
            'confidence_threshold': args.confidence,
        },
        'rgb_depth': {
            'pair_delta_ns_max': max(pair_deltas_ns),
            'pair_delta_ns_mean': _mean(
                [float(value) for value in pair_deltas_ns]
            ),
            'depth_scale_m_per_unit': float(
                pose_parameters['depth_scale_m_per_unit']
            ),
            'depth_scale_verified': bool(
                pose_parameters['depth_scale_verified']
            ),
            'aligned_valid_ratio_mean': _mean(aligned_valid_ratios),
            'registration_ms_mean': _mean(registration_ms),
            'registration_ms_p95': _percentile(registration_ms, 95.0),
        },
        'calibration': {
            'version': calibration_version,
            'support_plane_config_version': support_plane.config_version,
            'support_plane_provisional': (
                'provisional' in support_plane.config_version.lower()
            ),
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
            'association': 'class_then_left_to_right_ordinal_per_frame',
            'output_scope': 'control-facing dynamic named TF translation only',
        },
        'pose': {
            'instance_counts_by_class': dict(sorted(class_counts.items())),
            'validity_counts': dict(sorted(pose_counts.items())),
            'invalid_reason_counts': dict(sorted(invalid_reasons.items())),
            'postprocessing_totals': dict(sorted(postprocess_totals.items())),
        },
        'results': {
            'all_consecutive_3d_steps_raw': raw_summary,
            'all_consecutive_3d_steps_filtered': filtered_summary,
            'steady_state_3d_steps_raw': steady_raw_summary,
            'steady_state_3d_steps_filtered': steady_filtered_summary,
            'steady_state_p95_reduction_percent': p95_reduction,
            'decision_counts': dict(sorted(decision_counts.items())),
            'association_reset_total': (
                stabilizer.association_reset_total
            ),
        },
        'artifacts': {
            'comparison_video': (
                str(args.comparison_video)
                if args.comparison_video is not None
                else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--ontology', type=Path, required=True)
    parser.add_argument('--pose-config', type=Path, required=True)
    parser.add_argument('--roi-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--comparison-video', type=Path)
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--video-scale', type=float, default=0.75)
    parser.add_argument('--camera', default='cam_4')
    parser.add_argument('--confidence', type=float, default=0.3)
    parser.add_argument('--maximum-delta-ns', type=int, default=1_000_000)
    parser.add_argument('--max-pairs', type=int)
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
    for path in (
        args.bag,
        args.predictions,
        args.ontology,
        args.pose_config,
        args.roi_profile,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    report = evaluate(args)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
