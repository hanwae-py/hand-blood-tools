"""Quality-gated view selection for synchronized hand observations.

The node deliberately selects one complete camera observation at a time.  It
does not splice frame-local hand indices from different optical frames.  The
selected HandKeypoints and HandGestureArray therefore retain a truthful source
Header while a separate status document records the selected camera/quality.
This is perception evidence only and never authorizes robot motion.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from hand_keypoint_interfaces.msg import (
    HandFacingArray,
    HandGestureArray,
    HandKeypoints,
)
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String


_PAIR_CACHE_SIZE = 32


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def status_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _geometry_quality(hand: Any, width: int, height: int) -> tuple[float, float, float]:
    points = np.asarray(
        [(float(point.u), float(point.v)) for point in hand.joints_2d],
        dtype=np.float64,
    )
    if points.shape != (21, 2) or not np.all(np.isfinite(points)):
        return 0.0, 0.0, 0.0
    in_frame = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < max(width, 1))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < max(height, 1))
    )
    valid_fraction = float(np.count_nonzero(in_frame) / 21.0)
    x0, y0 = np.min(points, axis=0)
    x1, y1 = np.max(points, axis=0)
    bbox_width = max(float(x1 - x0), 1.0)
    bbox_height = max(float(y1 - y0), 1.0)
    image_diagonal = max(math.hypot(width, height), 1.0)
    diagonal_ratio = math.hypot(bbox_width, bbox_height) / image_diagonal
    size_quality = _clamp01((diagonal_ratio - 0.025) / 0.12)

    # A palm seen edge-on collapses wrist/MCP support into a thin hull.  This is
    # a view-quality cue, not an Open/Closed classifier, so it only has a modest
    # weight in the final score.
    palm = points[[0, 5, 9, 13, 17]].astype(np.float32)
    palm_area = float(cv2.contourArea(cv2.convexHull(palm)))
    palm_ratio = palm_area / max(bbox_width * bbox_height, 1.0)
    palm_quality = _clamp01(palm_ratio / 0.30)
    return valid_fraction, size_quality, palm_quality


def observation_quality(
    keypoints: HandKeypoints,
    gestures: HandGestureArray,
    *,
    width: int,
    height: int,
) -> tuple[float, int]:
    """Return a bounded frame quality and detected-hand count.

    Positive Open/Closed evidence dominates.  Palm-plane visibility and image
    scale break ties when a view becomes edge-on or partially occluded.  Depth
    validity is only a bonus, so an auxiliary RGB-fallback view is not
    penalized while its metric-depth path is being commissioned.
    """
    gesture_by_index = {
        int(hand.hand_index): hand for hand in getattr(gestures, 'hands', ())
    }
    hands = list(getattr(keypoints, 'hands', ()))
    if not hands:
        return 0.0, 0

    scores: list[float] = []
    for hand in hands:
        visible, size_quality, palm_quality = _geometry_quality(hand, width, height)
        gesture = gesture_by_index.get(int(hand.hand_index))
        if (
            gesture is not None
            and bool(getattr(gesture, 'has_classification', False))
            and str(getattr(gesture, 'category_name', '')) in {'Open_Palm', 'Closed_Fist'}
        ):
            classification = 0.55 + 0.45 * _clamp01(
                float(getattr(gesture, 'score', 0.0)))
        else:
            classification = 0.12
        handedness = (
            _clamp01(float(getattr(hand, 'handedness_score', 0.0)))
            if bool(getattr(hand, 'has_handedness', False))
            else 0.0
        )
        valid_depth = list(getattr(hand, 'kp_valid_depth', ()))
        depth_fraction = (
            float(sum(bool(value) for value in valid_depth) / len(valid_depth))
            if valid_depth else 0.0
        )
        score = (
            0.55 * classification
            + 0.18 * palm_quality
            + 0.12 * size_quality
            + 0.10 * visible
            + 0.03 * handedness
            + 0.02 * depth_fraction
        )
        scores.append(_clamp01(score))
    return _clamp01(float(np.mean(scores))), len(hands)


@dataclass
class ViewObservation:
    camera: str
    keypoints: HandKeypoints
    gestures: HandGestureArray
    source_stamp_ns: int
    received_monotonic: float
    quality: float
    hand_count: int


def synchronized_cohort(
    observations: dict[str, ViewObservation], max_delta_ns: int
) -> dict[str, ViewObservation]:
    """Choose the largest source-time cohort, breaking ties by recency.

    Cameras arrive asynchronously even though their source stamps are aligned.
    Anchoring on the newest callback would temporarily reduce the comparison to
    one camera every frame and bypass hysteresis.  A largest contiguous stamp
    window keeps the prior synchronized cohort until its peers arrive.
    """
    if not observations:
        return {}
    ordered = sorted(
        observations.items(), key=lambda item: item[1].source_stamp_ns)
    best: list[tuple[str, ViewObservation]] = []
    best_key = (-1, -1)
    for left in range(len(ordered)):
        right = left
        while (
            right + 1 < len(ordered)
            and ordered[right + 1][1].source_stamp_ns
            - ordered[left][1].source_stamp_ns <= max_delta_ns
        ):
            right += 1
        candidate = ordered[left:right + 1]
        candidate_key = (
            len(candidate),
            max(item.source_stamp_ns for _, item in candidate),
        )
        if candidate_key > best_key:
            best = candidate
            best_key = candidate_key
    return dict(best)


def synchronized_history_cohort(
    histories: dict[str, list[ViewObservation]],
    max_delta_ns: int,
    *,
    preferred_camera: str | None = None,
) -> dict[str, ViewObservation]:
    """Select the newest largest source-aligned cohort from recent histories.

    Independent MediaPipe workers intentionally drop different RGB frames.
    Their latest outputs are therefore often one or more 15 Hz periods apart
    even though the source cameras are aligned.  Retaining recent completed
    observations lets the selector recover the matching source epoch instead
    of widening the source-time contract.
    """
    ordered = sorted(
        (observation for values in histories.values() for observation in values),
        key=lambda observation: observation.source_stamp_ns,
    )
    best: dict[str, ViewObservation] = {}
    best_key = (-1, -1, -1)
    for left, first in enumerate(ordered):
        chosen: dict[str, ViewObservation] = {}
        for observation in ordered[left:]:
            if (
                observation.source_stamp_ns - first.source_stamp_ns
                > max_delta_ns
            ):
                break
            previous = chosen.get(observation.camera)
            if (
                previous is None
                or observation.received_monotonic > previous.received_monotonic
            ):
                chosen[observation.camera] = observation
        key = (
            len(chosen),
            int(preferred_camera in chosen) if preferred_camera else 0,
            max(
                (item.source_stamp_ns for item in chosen.values()),
                default=-1,
            ),
        )
        if key > best_key:
            best = chosen
            best_key = key
    return best


class QualitySelector:
    """Sticky best-view selector with immediate recovery from total occlusion."""

    def __init__(self, *, switch_margin: float, switch_frames: int) -> None:
        self.switch_margin = max(0.0, float(switch_margin))
        self.switch_frames = max(1, int(switch_frames))
        self.current: str | None = None
        self._challenger: str | None = None
        self._challenger_frames = 0
        self.switch_count = 0

    def choose(self, observations: dict[str, ViewObservation]) -> str | None:
        if not observations:
            self._challenger = None
            self._challenger_frames = 0
            return None
        best = max(
            observations,
            key=lambda camera: (
                observations[camera].hand_count,
                observations[camera].quality,
                observations[camera].source_stamp_ns,
            ),
        )
        if self.current not in observations:
            previous = self.current
            self.current = best
            self._challenger = None
            self._challenger_frames = 0
            if previous is not None and previous != best:
                self.switch_count += 1
            return self.current
        if best == self.current:
            self._challenger = None
            self._challenger_frames = 0
            return self.current

        current = observations[self.current]
        challenger = observations[best]
        current_occluded = current.hand_count == 0 and challenger.hand_count > 0
        materially_better = (
            challenger.hand_count > current.hand_count
            or challenger.quality >= current.quality + self.switch_margin
        )
        if current_occluded:
            self.current = best
            self._challenger = None
            self._challenger_frames = 0
            self.switch_count += 1
        elif materially_better:
            if self._challenger == best:
                self._challenger_frames += 1
            else:
                self._challenger = best
                self._challenger_frames = 1
            if self._challenger_frames >= self.switch_frames:
                self.current = best
                self._challenger = None
                self._challenger_frames = 0
                self.switch_count += 1
        else:
            self._challenger = None
            self._challenger_frames = 0
        return self.current


class MultiviewHandFusion(Node):
    def __init__(self, *, context: Context | None = None) -> None:
        super().__init__('multiview_hand_fusion', context=context)
        self.declare_parameter('cameras', ['cam_1', 'cam_3', 'cam_4'])
        self.declare_parameter('max_observation_age_sec', 0.35)
        self.declare_parameter('max_source_delta_ms', 20.0)
        self.declare_parameter('comparison_settle_sec', 0.02)
        self.declare_parameter('cohort_hold_max_sec', 0.15)
        self.declare_parameter('switch_margin', 0.12)
        self.declare_parameter('switch_frames', 3)
        self.declare_parameter('keypoints_topic', '/perception/hand/fused/keypoints')
        self.declare_parameter('gestures_topic', '/perception/hand/fused/gestures')
        self.declare_parameter('facing_topic', '/perception/hand/fused/facing')
        self.declare_parameter('status_topic', '/perception/hand/fused/status')

        cameras = [str(value).strip() for value in self.get_parameter('cameras').value]
        if not cameras or any(
            camera not in {'cam_1', 'cam_2', 'cam_3', 'cam_4'} for camera in cameras
        ):
            raise ValueError(
                'cameras must contain cam_1, cam_2, cam_3, and/or cam_4')
        self._cameras = tuple(dict.fromkeys(cameras))
        self._max_age = max(
            0.05, float(self.get_parameter('max_observation_age_sec').value))
        self._max_source_delta_ns = max(
            0, int(float(self.get_parameter('max_source_delta_ms').value) * 1_000_000))
        self._comparison_settle_sec = max(
            0.0, float(self.get_parameter('comparison_settle_sec').value))
        self._cohort_hold_max_sec = max(
            0.02, float(self.get_parameter('cohort_hold_max_sec').value))
        self._selector = QualitySelector(
            switch_margin=float(self.get_parameter('switch_margin').value),
            switch_frames=int(self.get_parameter('switch_frames').value),
        )
        self._keypoint_cache = {
            camera: OrderedDict() for camera in self._cameras}
        self._gesture_cache = {
            camera: OrderedDict() for camera in self._cameras}
        self._facing_cache = {
            camera: OrderedDict() for camera in self._cameras}
        self._last_finalized_stamp: dict[str, int | None] = {
            camera: None for camera in self._cameras}
        self._observations: dict[str, ViewObservation] = {}
        self._observation_history = {
            camera: OrderedDict() for camera in self._cameras}
        self._image_sizes = {camera: (1280, 720) for camera in self._cameras}
        self._last_publish_signature: tuple[str, int] | None = None
        self._last_published_source_stamp_ns: int | None = None
        self._last_published_observation: ViewObservation | None = None
        self._last_selected: str | None = None
        self._last_observation_update: float | None = None
        self._last_selection_input_signature: tuple[tuple[str, int], ...] | None = None
        self._cohort_hold_started: float | None = None
        self._last_facing_publish_signature: tuple[str, int] | None = None
        self._last_facing_selected: str | None = None
        self._last_facing_valid_hands = 0
        self._last_facing_mapping_version = ''
        self._last_facing_published_at: float | None = None
        self._last_facing_message: HandFacingArray | None = None

        output_qos = reliable_qos()
        self._keypoints_pub = self.create_publisher(
            HandKeypoints, str(self.get_parameter('keypoints_topic').value), output_qos)
        self._gestures_pub = self.create_publisher(
            HandGestureArray, str(self.get_parameter('gestures_topic').value), output_qos)
        self._facing_pub = self.create_publisher(
            HandFacingArray, str(self.get_parameter('facing_topic').value), output_qos)
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), status_qos())
        self._fusion_subscriptions: list[Any] = []
        for camera in self._cameras:
            self._fusion_subscriptions.extend((
                self.create_subscription(
                    HandKeypoints,
                    f'/perception/{camera}/hand/keypoints',
                    lambda message, cam=camera: self._on_keypoints(cam, message),
                    output_qos,
                ),
                self.create_subscription(
                    HandGestureArray,
                    f'/perception/{camera}/hand/gestures',
                    lambda message, cam=camera: self._on_gestures(cam, message),
                    output_qos,
                ),
                self.create_subscription(
                    HandFacingArray,
                    f'/perception/{camera}/hand/facing',
                    lambda message, cam=camera: self._on_facing(cam, message),
                    output_qos,
                ),
                self.create_subscription(
                    CameraInfo,
                    f'/perception/ingress/{camera}/color/camera_info',
                    lambda message, cam=camera: self._on_camera_info(cam, message),
                    reliable_qos(20),
                ),
            ))
        self.create_timer(0.01, self._select_if_settled)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(
            'multiview hand fusion ready: cameras=' + ','.join(self._cameras))

    def _on_camera_info(self, camera: str, message: CameraInfo) -> None:
        if int(message.width) > 0 and int(message.height) > 0:
            self._image_sizes[camera] = (int(message.width), int(message.height))

    def _on_keypoints(self, camera: str, message: HandKeypoints) -> None:
        source_stamp = self._cache_message(self._keypoint_cache[camera], message)
        self._finalize(camera, source_stamp)

    def _on_gestures(self, camera: str, message: HandGestureArray) -> None:
        source_stamp = self._cache_message(self._gesture_cache[camera], message)
        self._finalize(camera, source_stamp)

    def _on_facing(self, camera: str, message: HandFacingArray) -> None:
        source_stamp = self._cache_message(self._facing_cache[camera], message)
        # Facing can arrive after keypoints+gesture selection.  Join that late
        # message only to the observation that was actually published; using a
        # newly computed cohort here could emit facing from a different source
        # frame than the fused keypoints and gesture arrays.
        observation = self._last_published_observation
        if (
            observation is not None
            and observation.camera == camera
            and observation.source_stamp_ns == source_stamp
        ):
            self._publish_facing({camera: observation}, camera)

    @staticmethod
    def _cache_message(cache: OrderedDict, message: Any) -> int:
        source_stamp = stamp_ns(message)
        cache[source_stamp] = message
        cache.move_to_end(source_stamp)
        while len(cache) > _PAIR_CACHE_SIZE:
            cache.popitem(last=False)
        return source_stamp

    def _finalize(self, camera: str, source_stamp: int) -> None:
        keypoints = self._keypoint_cache[camera].get(source_stamp)
        gestures = self._gesture_cache[camera].get(source_stamp)
        if keypoints is None or gestures is None:
            return
        last_stamp = self._last_finalized_stamp[camera]
        if last_stamp is not None and source_stamp <= last_stamp:
            self._keypoint_cache[camera].pop(source_stamp, None)
            self._gesture_cache[camera].pop(source_stamp, None)
            return
        self._keypoint_cache[camera].pop(source_stamp, None)
        self._gesture_cache[camera].pop(source_stamp, None)
        width, height = self._image_sizes[camera]
        quality, hand_count = observation_quality(
            keypoints, gestures, width=width, height=height)
        self._observations[camera] = ViewObservation(
            camera=camera,
            keypoints=keypoints,
            gestures=gestures,
            source_stamp_ns=source_stamp,
            received_monotonic=time.monotonic(),
            quality=quality,
            hand_count=hand_count,
        )
        self._observation_history[camera][source_stamp] = self._observations[camera]
        self._observation_history[camera].move_to_end(source_stamp)
        while len(self._observation_history[camera]) > _PAIR_CACHE_SIZE:
            self._observation_history[camera].popitem(last=False)
        self._last_finalized_stamp[camera] = source_stamp
        self._last_observation_update = time.monotonic()

    def _live_observations(self) -> dict[str, ViewObservation]:
        now = time.monotonic()
        return {
            camera: observation
            for camera, observation in self._observations.items()
            if now - observation.received_monotonic <= self._max_age
        }

    def _comparable_live(self) -> dict[str, ViewObservation]:
        now = time.monotonic()
        histories = {
            camera: [
                observation
                for observation in history.values()
                if now - observation.received_monotonic <= self._max_age
            ]
            for camera, history in self._observation_history.items()
        }
        if not any(histories.values()):
            return {}
        return synchronized_history_cohort(
            histories,
            self._max_source_delta_ns,
            preferred_camera=self._selector.current,
        )

    def _select_if_settled(self) -> None:
        if self._last_observation_update is None:
            return
        if time.monotonic() - self._last_observation_update < self._comparison_settle_sec:
            return
        self._select_and_publish()

    def _select_and_publish(self) -> None:
        live = self._live_observations()
        observations = self._comparable_live()
        if (
            self._selector.current in live
            and self._selector.current not in observations
        ):
            # A peer can finish inference slightly before the current view.
            # Hold the last published evidence until a matched cohort arrives
            # or the current view genuinely ages out.
            if self._cohort_hold_started is None:
                self._cohort_hold_started = time.monotonic()
            if (
                time.monotonic() - self._cohort_hold_started
                < self._cohort_hold_max_sec
            ):
                return
        else:
            self._cohort_hold_started = None
        input_signature = tuple(sorted(
            (camera, observation.source_stamp_ns)
            for camera, observation in observations.items()))
        if input_signature == self._last_selection_input_signature:
            return
        self._last_selection_input_signature = input_signature
        selected = self._selector.choose(observations)
        self._last_selected = selected
        if selected is None:
            return
        observation = observations[selected]
        signature = (selected, observation.source_stamp_ns)
        if signature == self._last_publish_signature:
            return
        if (
            self._last_published_source_stamp_ns is not None
            and observation.source_stamp_ns
            <= self._last_published_source_stamp_ns
        ):
            return
        self._keypoints_pub.publish(observation.keypoints)
        self._gestures_pub.publish(observation.gestures)
        self._last_publish_signature = signature
        self._last_published_source_stamp_ns = observation.source_stamp_ns
        self._last_published_observation = observation
        self._publish_facing(observations, selected)

    def _publish_facing(
        self, observations: dict[str, ViewObservation], selected: str
    ) -> None:
        observation = observations.get(selected)
        if observation is None:
            return
        message = self._facing_cache[selected].get(observation.source_stamp_ns)
        if message is None:
            return
        valid_hands = sum(
            bool(getattr(hand, 'has_facing', False))
            for hand in getattr(message, 'hands', ()))
        signature = (selected, observation.source_stamp_ns)
        if signature != self._last_publish_signature:
            return
        if signature == self._last_facing_publish_signature:
            return
        self._facing_pub.publish(message)
        self._last_facing_publish_signature = signature
        self._last_facing_selected = selected
        self._last_facing_valid_hands = valid_hands
        self._last_facing_mapping_version = str(
            getattr(message, 'handedness_mapping_version', ''))
        self._last_facing_published_at = time.monotonic()
        self._last_facing_message = message

    def _publish_status(self) -> None:
        now = time.monotonic()
        observations = self._comparable_live()
        selected_observation = self._last_published_observation
        selected_age = (
            None if selected_observation is None
            else now - selected_observation.received_monotonic)
        if selected_age is not None and selected_age > self._max_age:
            selected_observation = None
        selected = (
            None if selected_observation is None
            else selected_observation.camera)
        facing_age = (
            None if self._last_facing_published_at is None
            else now - self._last_facing_published_at)
        facing_live = facing_age is not None and facing_age <= self._max_age
        gesture_facing_joinable = bool(
            selected_observation is not None
            and facing_live
            and self._last_publish_signature is not None
            and self._last_publish_signature == self._last_facing_publish_signature
        )
        selected_gestures = []
        if selected_observation is not None:
            for hand in getattr(selected_observation.gestures, 'hands', ()):
                selected_gestures.append({
                    'hand_index': int(getattr(hand, 'hand_index', -1)),
                    'category': (
                        str(getattr(hand, 'category_name', ''))
                        if bool(getattr(hand, 'has_classification', False))
                        else 'None'),
                    'score': round(float(getattr(hand, 'score', 0.0)), 4),
                })
        selected_facings = []
        if gesture_facing_joinable and self._last_facing_message is not None:
            for hand in getattr(self._last_facing_message, 'hands', ()):
                selected_facings.append({
                    'hand_index': int(getattr(hand, 'hand_index', -1)),
                    'category': (
                        str(getattr(hand, 'facing_label', 'UNKNOWN'))
                        if bool(getattr(hand, 'has_facing', False))
                        else 'UNKNOWN'),
                    'score': round(float(getattr(hand, 'palm_up_score', 0.0)), 4),
                })
        payload = {
            'schema': 'pnu.perception.multiview_hand_fusion.v1',
            'node': self.get_name(),
            'selected_camera': selected if selected_observation is not None else None,
            'selected_quality': (
                None if selected_observation is None
                else round(selected_observation.quality, 4)
            ),
            'selected_source_stamp_ns': (
                None if selected_observation is None
                else selected_observation.source_stamp_ns
            ),
            'switch_count': self._selector.switch_count,
            'cohort_cameras': sorted(observations),
            'selector_current_camera': self._selector.current,
            'selected_facing_camera': (
                self._last_facing_selected if facing_live else None),
            'selected_facing_source_stamp_ns': (
                None if not facing_live or self._last_facing_publish_signature is None
                else self._last_facing_publish_signature[1]),
            'selected_facing_age_sec': (
                None if facing_age is None else round(max(0.0, facing_age), 3)),
            'selected_facing_valid_hands': (
                self._last_facing_valid_hands if facing_live else 0),
            'selected_facing_mapping_version': (
                self._last_facing_mapping_version if facing_live else ''),
            'selected_facing_provisional': (
                facing_live
                and 'pending' in self._last_facing_mapping_version.lower()),
            'gesture_facing_joinable': gesture_facing_joinable,
            'selected_gestures': selected_gestures,
            'selected_facings': selected_facings,
            'policy': {
                'max_observation_age_sec': self._max_age,
                'max_source_delta_ms': self._max_source_delta_ns / 1_000_000.0,
                'comparison_settle_sec': self._comparison_settle_sec,
                'cohort_hold_max_sec': self._cohort_hold_max_sec,
                'switch_margin': self._selector.switch_margin,
                'switch_frames': self._selector.switch_frames,
            },
            'views': {},
            'robot_authority': False,
        }
        for camera in self._cameras:
            observation = self._observations.get(camera)
            age = None if observation is None else now - observation.received_monotonic
            payload['views'][camera] = {
                'state': (
                    'missing' if observation is None
                    else ('stale' if age is None or age > self._max_age else 'live')
                ),
                'quality': None if observation is None else round(observation.quality, 4),
                'hands': 0 if observation is None else observation.hand_count,
                'age_sec': None if age is None else round(max(0.0, age), 3),
                'source_stamp_ns': (
                    None if observation is None else observation.source_stamp_ns),
            }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(',', ':'))))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MultiviewHandFusion()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
