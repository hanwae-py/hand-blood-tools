"""Depth-backed semantic palm-facing estimation for fixed CAM4.

MediaPipe world landmarks are useful for finger articulation but are not in
the RealSense camera frame. Palm facing therefore uses registered metric depth
at the stable palm base (wrist and MCP landmarks) and compares an anatomical
palmar normal with the calibrated surgical-table up normal.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np


FACING_NAMES = ('PALM_UP', 'PALM_DOWN', 'EDGE', 'UNKNOWN')
ESTIMATOR_NAME = 'VIPLab CAM4 Depth Palm-Facing Estimator'
ESTIMATOR_VERSION = 'depth-palm-normal-v1'
PALM_LANDMARKS = (0, 5, 9, 13, 17)
REQUIRED_LANDMARKS = (0, 5, 17)


def _normalized(vector, *, minimum_norm=1e-9):
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm < minimum_norm:
        return None
    return values / norm


def _unknown(
    reason,
    *,
    valid_depth_points=0,
    calibration_version='',
    plane_residual_m=0.0,
    normal_cam=(0.0, 0.0, 0.0),
    support_height_m=0.0,
):
    return {
        'has_facing': False,
        'label': 'UNKNOWN',
        'palm_up_score': 0.0,
        'normal_cam': [float(value) for value in normal_cam],
        'plane_residual_m': float(plane_residual_m),
        'support_height_m': float(support_height_m),
        'valid_depth_points': int(valid_depth_points),
        'quality_valid': False,
        'rejection_reason': str(reason),
        'calibration_version': str(calibration_version),
    }


class PalmFacingEstimator:
    """Estimate PALM_UP/DOWN from metric palm-base landmarks.

    ``handedness_signs`` converts the topology normal into an anatomical
    palmar normal. CAM4 is non-mirrored while MediaPipe's handedness convention
    assumes mirrored selfie input, so these signs are explicit calibrated
    parameters rather than hidden assumptions.
    """

    def __init__(
        self,
        *,
        table_up_normal,
        support_plane_offset_m,
        handedness_signs,
        enter_cosine=0.75,
        max_plane_residual_m=0.012,
        min_palm_span_m=0.025,
        min_handedness_score=0.60,
        calibration_version='',
        handedness_mapping_version='',
        min_support_height_m=0.008,
        max_support_height_m=0.25,
    ):
        table_up = _normalized(table_up_normal)
        if table_up is None or np.asarray(table_up_normal).shape != (3,):
            raise ValueError('table_up_normal must be one finite non-zero 3-vector')
        signs = {}
        for label in ('Left', 'Right'):
            value = float(handedness_signs[label])
            if not np.isfinite(value) or abs(abs(value) - 1.0) > 1e-6:
                raise ValueError(f'handedness sign for {label} must be +1 or -1')
            signs[label] = value
        self.table_up_normal = table_up
        self.support_plane_offset_m = float(support_plane_offset_m)
        self.handedness_signs = signs
        self.enter_cosine = float(enter_cosine)
        self.max_plane_residual_m = float(max_plane_residual_m)
        self.min_palm_span_m = float(min_palm_span_m)
        self.min_handedness_score = float(min_handedness_score)
        self.calibration_version = str(calibration_version).strip()
        self.handedness_mapping_version = str(
            handedness_mapping_version).strip()
        self.min_support_height_m = float(min_support_height_m)
        self.max_support_height_m = float(max_support_height_m)
        if not np.isfinite(self.support_plane_offset_m):
            raise ValueError('support_plane_offset_m must be finite')
        if self.handedness_signs['Left'] == self.handedness_signs['Right']:
            raise ValueError('Left and Right handedness signs must be opposite')
        if not 0.0 < self.enter_cosine < 1.0:
            raise ValueError('enter_cosine must be between 0 and 1')
        if self.max_plane_residual_m <= 0.0:
            raise ValueError('max_plane_residual_m must be positive')
        if self.min_palm_span_m <= 0.0:
            raise ValueError('min_palm_span_m must be positive')
        if not 0.0 <= self.min_handedness_score <= 1.0:
            raise ValueError('min_handedness_score must be in [0, 1]')
        if not self.calibration_version:
            raise ValueError('calibration_version must be non-empty')
        if not self.handedness_mapping_version:
            raise ValueError('handedness_mapping_version must be non-empty')
        if not (
            0.0 <= self.min_support_height_m
            < self.max_support_height_m
        ):
            raise ValueError(
                'support height limits must satisfy 0 <= min < max')

        spec = (
            f'version={ESTIMATOR_VERSION};'
            f'table_up={self.table_up_normal.round(9).tolist()};'
            f'support_plane_offset_m={self.support_plane_offset_m:.9f};'
            f'signs={self.handedness_signs};'
            f'enter_cosine={self.enter_cosine:.4f};'
            f'max_plane_residual_m={self.max_plane_residual_m:.5f};'
            f'min_palm_span_m={self.min_palm_span_m:.5f};'
            f'min_handedness_score={self.min_handedness_score:.4f};'
            f'calibration={self.calibration_version};'
            f'handedness_mapping={self.handedness_mapping_version};'
            f'support_height_m={self.min_support_height_m:.4f},'
            f'{self.max_support_height_m:.4f}'
        )
        self.spec_sha256 = hashlib.sha256(spec.encode('utf-8')).hexdigest()

    def estimate(self, joints_3d, valid_depth, handedness):
        points = np.asarray(joints_3d, dtype=np.float64)
        valid = np.asarray(valid_depth, dtype=bool)
        if points.shape != (21, 3) or valid.shape != (21,):
            return _unknown(
                'invalid_metric_landmark_array',
                calibration_version=self.calibration_version,
            )
        finite = np.all(np.isfinite(points), axis=1)
        usable = valid & finite & (points[:, 2] > 0.0)
        valid_palm_points = int(sum(bool(usable[index]) for index in PALM_LANDMARKS))
        if valid_palm_points < 4 or not all(usable[index] for index in REQUIRED_LANDMARKS):
            return _unknown(
                'insufficient_palm_depth',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )

        if not handedness:
            return _unknown(
                'handedness_unavailable',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )
        label = str(handedness.get('label', '')).strip()
        score = float(handedness.get('score', 0.0))
        if label not in self.handedness_signs:
            return _unknown(
                'handedness_label_unsupported',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )
        if not np.isfinite(score) or score < self.min_handedness_score:
            return _unknown(
                'handedness_confidence_too_low',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )

        palm_indices = [index for index in PALM_LANDMARKS if usable[index]]
        palm_points = points[palm_indices]
        wrist = points[0]
        mcp_points = points[[index for index in (5, 9, 13, 17) if usable[index]]]
        longitudinal = _normalized(np.mean(mcp_points, axis=0) - wrist)
        transverse_raw = points[17] - points[5]
        if longitudinal is None:
            return _unknown(
                'degenerate_palm_longitudinal_axis',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )
        transverse = transverse_raw - longitudinal * float(
            np.dot(longitudinal, transverse_raw))
        transverse = _normalized(transverse)
        if transverse is None:
            return _unknown(
                'degenerate_palm_transverse_axis',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )
        if (
            float(np.linalg.norm(np.mean(mcp_points, axis=0) - wrist))
            < self.min_palm_span_m
            or float(np.linalg.norm(points[17] - points[5]))
            < self.min_palm_span_m
        ):
            return _unknown(
                'metric_palm_geometry_too_small',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )

        raw_normal = _normalized(np.cross(longitudinal, transverse))
        if raw_normal is None:
            return _unknown(
                'degenerate_palm_normal',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
            )
        palmar_normal = raw_normal * self.handedness_signs[label]
        centre = np.mean(palm_points, axis=0)
        residuals = np.abs((palm_points - centre) @ palmar_normal)
        plane_residual = float(np.sqrt(np.mean(residuals ** 2)))
        if not np.isfinite(plane_residual) or plane_residual > self.max_plane_residual_m:
            return _unknown(
                'palm_plane_residual_too_large',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
                plane_residual_m=plane_residual,
                normal_cam=palmar_normal,
            )

        support_height = float(np.median(
            palm_points @ self.table_up_normal
            + self.support_plane_offset_m
        ))
        if (
            not np.isfinite(support_height)
            or support_height < self.min_support_height_m
            or support_height > self.max_support_height_m
        ):
            return _unknown(
                'palm_support_height_out_of_range',
                valid_depth_points=valid_palm_points,
                calibration_version=self.calibration_version,
                plane_residual_m=plane_residual,
                normal_cam=palmar_normal,
                support_height_m=support_height,
            )

        palm_up_score = float(np.clip(
            np.dot(palmar_normal, self.table_up_normal), -1.0, 1.0))
        if palm_up_score >= self.enter_cosine:
            facing_label = 'PALM_UP'
        elif palm_up_score <= -self.enter_cosine:
            facing_label = 'PALM_DOWN'
        else:
            facing_label = 'EDGE'
        return {
            'has_facing': True,
            'label': facing_label,
            'palm_up_score': palm_up_score,
            'normal_cam': [float(value) for value in palmar_normal],
            'plane_residual_m': plane_residual,
            'support_height_m': support_height,
            'valid_depth_points': valid_palm_points,
            'quality_valid': True,
            'rejection_reason': '',
            'calibration_version': self.calibration_version,
        }


class PalmFacingTemporalFilter:
    """Short EMA + hysteresis keyed by handedness and image centroid."""

    def __init__(
        self,
        *,
        enter_cosine=0.75,
        hold_cosine=0.60,
        alpha=0.50,
        table_up_normal=(0.0, 0.0, -1.0),
        max_centroid_distance=0.25,
        track_ttl_sec=1.0,
    ):
        self.enter_cosine = float(enter_cosine)
        self.hold_cosine = float(hold_cosine)
        self.alpha = float(alpha)
        table_up = _normalized(table_up_normal)
        if table_up is None or np.asarray(table_up_normal).shape != (3,):
            raise ValueError('table_up_normal must be one finite non-zero 3-vector')
        self.table_up_normal = table_up
        self.max_centroid_distance = float(max_centroid_distance)
        self.track_ttl_sec = float(track_ttl_sec)
        if not 0.0 <= self.hold_cosine < self.enter_cosine < 1.0:
            raise ValueError('facing cosine thresholds must satisfy 0 <= hold < enter < 1')
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')
        if self.max_centroid_distance <= 0.0 or self.track_ttl_sec <= 0.0:
            raise ValueError('track distance and TTL must be positive')
        filter_spec = (
            f'version=temporal-normal-ema-v1;'
            f'enter={self.enter_cosine:.4f};hold={self.hold_cosine:.4f};'
            f'alpha={self.alpha:.4f};'
            f'max_centroid_distance={self.max_centroid_distance:.4f};'
            f'track_ttl_sec={self.track_ttl_sec:.4f};'
            f'table_up={self.table_up_normal.round(9).tolist()}'
        )
        self.spec_sha256 = hashlib.sha256(
            filter_spec.encode('utf-8')).hexdigest()
        self.reset()

    def reset(self):
        self._tracks = []

    def update(self, facing, *, centroid_norm, handedness):
        result = dict(facing)
        if not bool(result.get('has_facing', False)):
            return result
        now = time.monotonic()
        self._tracks = [
            track for track in self._tracks
            if now - track['last_seen'] <= self.track_ttl_sec
        ]
        side = str((handedness or {}).get('label', ''))
        centroid = np.asarray(centroid_norm, dtype=np.float64)
        candidates = [track for track in self._tracks if track['side'] == side]
        track = None
        if candidates:
            nearest = min(
                candidates,
                key=lambda item: float(np.linalg.norm(item['centroid'] - centroid)),
            )
            if float(np.linalg.norm(nearest['centroid'] - centroid)) <= self.max_centroid_distance:
                track = nearest

        raw_score = float(result['palm_up_score'])
        raw_normal = _normalized(result.get('normal_cam', (0.0, 0.0, 0.0)))
        if raw_normal is None:
            return _unknown(
                'temporal_filter_invalid_normal',
                valid_depth_points=result.get('valid_depth_points', 0),
                calibration_version=result.get('calibration_version', ''),
                plane_residual_m=result.get('plane_residual_m', 0.0),
                support_height_m=result.get('support_height_m', 0.0),
            )
        if track is None:
            track = {
                'side': side,
                'centroid': centroid,
                'score': float(np.clip(
                    np.dot(raw_normal, self.table_up_normal), -1.0, 1.0)),
                'normal': raw_normal,
                'label': 'EDGE',
                'last_seen': now,
            }
            self._tracks.append(track)
        else:
            filtered_normal = _normalized(
                self.alpha * raw_normal
                + (1.0 - self.alpha) * track['normal'])
            if filtered_normal is None:
                filtered_normal = raw_normal
            track['normal'] = filtered_normal
            track['score'] = float(np.clip(
                np.dot(filtered_normal, self.table_up_normal), -1.0, 1.0))
            track['centroid'] = centroid
            track['last_seen'] = now

        filtered_score = float(np.clip(track['score'], -1.0, 1.0))
        previous_label = str(track['label'])
        if filtered_score >= self.enter_cosine:
            label = 'PALM_UP'
        elif filtered_score <= -self.enter_cosine:
            label = 'PALM_DOWN'
        elif previous_label == 'PALM_UP' and filtered_score >= self.hold_cosine:
            label = 'PALM_UP'
        elif previous_label == 'PALM_DOWN' and filtered_score <= -self.hold_cosine:
            label = 'PALM_DOWN'
        else:
            label = 'EDGE'
        track['label'] = label
        result['palm_up_score_raw'] = raw_score
        result['palm_up_score'] = filtered_score
        result['normal_cam'] = [float(value) for value in track['normal']]
        result['label'] = label
        return result


def estimator_metadata(estimator, temporal_filter=None):
    """Return JSON-safe provenance for health and diagnostics."""
    if estimator is None:
        return {
            'name': ESTIMATOR_NAME,
            'version': ESTIMATOR_VERSION,
            'enabled': False,
        }
    combined_spec_sha256 = estimator.spec_sha256
    temporal_metadata = None
    if temporal_filter is not None:
        combined_spec_sha256 = hashlib.sha256(
            f'{estimator.spec_sha256}:{temporal_filter.spec_sha256}'.encode(
                'utf-8')).hexdigest()
        temporal_metadata = {
            'spec_sha256': temporal_filter.spec_sha256,
            'enter_cosine': temporal_filter.enter_cosine,
            'hold_cosine': temporal_filter.hold_cosine,
            'alpha': temporal_filter.alpha,
            'max_centroid_distance': (
                temporal_filter.max_centroid_distance),
            'track_ttl_sec': temporal_filter.track_ttl_sec,
        }
    metadata = {
        'name': ESTIMATOR_NAME,
        'version': ESTIMATOR_VERSION,
        'enabled': True,
        'spec_sha256': combined_spec_sha256,
        'estimator_spec_sha256': estimator.spec_sha256,
        'calibration_version': estimator.calibration_version,
        'handedness_mapping_version': (
            estimator.handedness_mapping_version),
        'table_up_normal': [
            round(float(value), 9) for value in estimator.table_up_normal],
        'support_plane_offset_m': estimator.support_plane_offset_m,
        'handedness_signs': dict(estimator.handedness_signs),
        'enter_cosine': estimator.enter_cosine,
        'max_plane_residual_m': estimator.max_plane_residual_m,
        'min_palm_span_m': estimator.min_palm_span_m,
        'min_handedness_score': estimator.min_handedness_score,
        'min_support_height_m': estimator.min_support_height_m,
        'max_support_height_m': estimator.max_support_height_m,
    }
    if temporal_metadata is not None:
        metadata['temporal_filter'] = temporal_metadata
    return metadata
