"""Build safe, temporally stable constrained SE(3) tool TF transforms.

The native estimator observes RGB-D position and in-plane longitudinal
heading. It completes orientation with the calibrated support-plane normal.
The quaternion is therefore suitable for a full SE(3) TF transform, but it
does not turn the authoritative ToolPoseArray into an unconstrained 6-DoF
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

from geometry_msgs.msg import TransformStamped

from surgical_perception_msgs.msg import ToolPose

from pnu_surgical_perception.tool_axis_stabilizer import ToolAxisStabilizer
from pnu_surgical_perception.tool_position_stabilizer import (
    ToolPositionStabilizer,
)


CONSTRAINED_SE3_PROVENANCE = (
    'position_from_registered_depth; yaw_from_mask_longitudinal_axis; '
    'roll_pitch_completed_from_support_plane_normal'
)


@dataclass(frozen=True)
class ToolTfDecision:
    """One accepted transform or the precise reason it was not emitted."""

    transform: TransformStamped | None
    reason: str
    child_frame_id: str = ''


@dataclass(frozen=True)
class _ConstrainedPoseComponents:
    """Validated SE(3) components before assignment to a temporal track."""

    parent_frame_id: str
    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass
class _Track:
    """Latest geometry for one bounded, camera-local class track."""

    track_id: int
    position_m: tuple[float, float, float]
    last_source_stamp_ns: int


@dataclass
class _TrackGroup:
    """Track state for exactly one camera and canonical tool class."""

    next_track_id: int = 1
    last_source_stamp_ns: int | None = None
    tracks: dict[int, _Track] = field(default_factory=dict)


def source_stamp_nanoseconds(header: Any) -> int | None:
    """Return a positive ROS source timestamp in nanoseconds, if present."""
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return None
    seconds = int(getattr(stamp, 'sec', 0))
    nanoseconds = int(getattr(stamp, 'nanosec', 0))
    value = seconds * 1_000_000_000 + nanoseconds
    return value if value > 0 else None


def source_age_seconds(header: Any, now_nanoseconds: int) -> float | None:
    """Return ``now - source`` using the ROS-clock timebase, when available."""
    source_nanoseconds = source_stamp_nanoseconds(header)
    if source_nanoseconds is None:
        return None
    return (int(now_nanoseconds) - source_nanoseconds) / 1_000_000_000.0


def _frame_token(value: object, fallback: str) -> str:
    """Return an ASCII TF-safe token without silently accepting an empty ID."""
    token = re.sub(r'[^a-z0-9_]+', '_', str(value).strip().lower())
    token = token.strip('_')
    return token or fallback


_SHORT_TOOL_FRAME_NAMES = {
    'scalpel': 'scalpel',
    'allis_forceps': 'allis',
    'mosquito': 'mosquito',
    'adson_forceps': 'adson',
    'bipolar_forceps': 'bipolar',
    'bovie': 'bovie',
    'army_navy_retractor': 'army',
    'thyroid_retractor': 'thyroid',
}


def constrained_tool_child_frame(
    view: str, tool: ToolPose, track_id: int
) -> str:
    """Build a camera-qualified, human-readable child frame for one track."""
    if int(track_id) <= 0:
        raise ValueError('track_id must be positive')
    camera = _frame_token(view, 'camera')
    class_name = _frame_token(tool.class_name, 'tool')
    tool_name = _SHORT_TOOL_FRAME_NAMES.get(class_name, class_name)
    # The camera prefix keeps child frames globally unique. The short suffix is
    # intentionally operator-facing: ``allis#1``, ``mosquito#2``, ``army#5``.
    # The number is the stable per-camera/per-class temporal track ID.
    return f'{camera}_{tool_name}#{int(track_id)}'


def workspace_tool_child_frame(
    workspace_zone: str, tool: Any, ordinal: int
) -> str:
    """Build the operator-facing name for a current spatial selector.

    This name deliberately identifies a *slot in the current observation*,
    not a persistent physical instrument.  For example ``mayo_army#1`` means
    the left-most detected Army-Navy retractor on the Mayo view right now.
    """
    if int(ordinal) <= 0:
        raise ValueError('ordinal must be positive')
    zone = _frame_token(workspace_zone, 'workspace')
    class_name = _frame_token(getattr(tool, 'class_name', ''), 'tool')
    tool_name = _SHORT_TOOL_FRAME_NAMES.get(class_name, class_name)
    return f'{zone}_{tool_name}#{int(ordinal)}'


def selector_horizontal_u_px(
    tool: Any,
    horizontal_u_by_instance_id: Mapping[int, float] | None = None,
) -> float:
    """Return the best finite camera-image horizontal coordinate for a tool."""
    instance_id = int(getattr(tool, 'frame_local_instance_id', 0))
    if horizontal_u_by_instance_id is not None:
        override = horizontal_u_by_instance_id.get(instance_id)
        if override is not None and math.isfinite(float(override)):
            return float(override)

    bbox = tuple(getattr(tool, 'bbox_xyxy_px', ()))
    if len(bbox) == 4:
        x0, _y0, x1, _y1 = (float(value) for value in bbox)
        if math.isfinite(x0) and math.isfinite(x1):
            return (x0 + x1) / 2.0

    uv = tuple(getattr(tool, 'observation_point_uv_px', ()))
    if len(uv) >= 1 and math.isfinite(float(uv[0])):
        return float(uv[0])
    return math.inf


def spatial_tool_child_frames(
    workspace_zone: str,
    tools: Sequence[Any],
    *,
    horizontal_u_by_instance_id: Mapping[int, float] | None = None,
) -> list[str]:
    """Name tools by class and current left-to-right order in one camera.

    The returned list preserves input order.  Ordinals are independent for
    every canonical class.  Ties are deterministic, but the semantic contract
    remains a source-stamped spatial snapshot rather than durable identity.
    """
    labels = [''] * len(tools)
    groups: dict[tuple[int, str], list[int]] = {}
    for index, tool in enumerate(tools):
        key = (
            int(getattr(tool, 'canonical_class_id', 0)),
            _frame_token(getattr(tool, 'class_name', ''), 'tool'),
        )
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                selector_horizontal_u_px(
                    tools[index], horizontal_u_by_instance_id
                ),
                int(getattr(tools[index], 'frame_local_instance_id', 0)),
                index,
            ),
        )
        for ordinal, index in enumerate(ordered, start=1):
            labels[index] = workspace_tool_child_frame(
                workspace_zone, tools[index], ordinal
            )
    return labels


def _validated_components(
    header: Any, tool: ToolPose
) -> tuple[_ConstrainedPoseComponents | None, str]:
    """Validate that a measured planar pose is numerically representable in TF.

    ``validity``, ``position_valid``, ``orientation_valid`` and
    ``dof_observed`` are retained as quality evidence in ``ToolPoseArray``. They
    intentionally do not gate TF: a degraded estimator result can still carry
    the finite position and constrained quaternion shown by the pose overlay.
    Results with no computed pose map to a zero-norm quaternion and remain
    rejected by the structural checks below.
    """
    parent_frame = str(getattr(header, 'frame_id', '')).strip()
    if not parent_frame or parent_frame.startswith('/'):
        return None, 'PARENT_FRAME_INVALID'
    if source_stamp_nanoseconds(header) is None:
        return None, 'SOURCE_STAMP_MISSING'
    if tool.pose_mode != ToolPose.POSE_MODE_PLANAR_4DOF_WITH_NORMAL_PRIOR:
        return None, 'POSE_MODE_NOT_PLANAR_4DOF_WITH_NORMAL_PRIOR'

    position = tool.pose.position
    translation = (float(position.x), float(position.y), float(position.z))
    if not all(math.isfinite(value) for value in translation):
        return None, 'POSITION_NONFINITE'

    orientation = tool.pose.orientation
    quaternion = (
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
    if not all(math.isfinite(value) for value in quaternion):
        return None, 'ORIENTATION_NONFINITE'
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-8:
        return None, 'ORIENTATION_ZERO_NORM'
    normalized = tuple(value / norm for value in quaternion)
    return (
        _ConstrainedPoseComponents(parent_frame, translation, normalized),
        'PUBLISHED',
    )


def _transform_from_components(
    header: Any,
    child_frame_id: str,
    components: _ConstrainedPoseComponents,
) -> TransformStamped:
    """Construct a source-stamped TransformStamped from validated components."""
    transform = TransformStamped()
    transform.header.stamp = header.stamp
    transform.header.frame_id = components.parent_frame_id
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = components.translation[0]
    transform.transform.translation.y = components.translation[1]
    transform.transform.translation.z = components.translation[2]
    transform.transform.rotation.x = components.quaternion_xyzw[0]
    transform.transform.rotation.y = components.quaternion_xyzw[1]
    transform.transform.rotation.z = components.quaternion_xyzw[2]
    transform.transform.rotation.w = components.quaternion_xyzw[3]
    return transform


def constrained_transform_from_tool_pose(
    header: Any,
    tool: ToolPose,
    view: str,
    track_id: int,
) -> ToolTfDecision:
    """Convert a measured planar-with-normal-prior pose to a tracked TF frame."""
    components, reason = _validated_components(header, tool)
    if components is None:
        return ToolTfDecision(None, reason)
    try:
        child_frame_id = constrained_tool_child_frame(view, tool, track_id)
    except ValueError:
        return ToolTfDecision(None, 'TRACK_ID_INVALID')
    return ToolTfDecision(
        _transform_from_components(header, child_frame_id, components),
        'PUBLISHED',
        child_frame_id,
    )


class ToolSpatialTfSelector:
    """Publish current zone/class/left-to-right selectors as dynamic TFs.

    A selector such as ``mayo_army#2`` is intentionally ephemeral: it denotes
    the second Army-Navy retractor from the left in a source-stamped camera
    observation.  It does not claim that the same physical instrument keeps
    that name after removal, reordering, occlusion, or re-entry.
    """

    def __init__(
        self,
        *,
        max_tools_per_class: int,
        reset_stamp_jump_sec: float,
        position_stabilization_enabled: bool = False,
        position_deadband_m: float = 0.0,
        position_smoothing_alpha: float = 0.20,
        position_max_jump_m: float = 0.04,
        position_relocation_confirmation_frames: int = 2,
        position_relocation_consistency_m: float = 0.015,
        position_max_missed_frames: int = 3,
        axis_stabilization_enabled: bool = False,
        axis_flip_confirmation_frames: int = 3,
        axis_flip_dot_threshold: float = 0.0,
        axis_pending_consistency_dot: float = 0.85,
        axis_angular_deadband_rad: float = 0.0,
        axis_smoothing_alpha: float = 1.0,
        axis_lock_opposite: bool = False,
        axis_max_missed_frames: int = 3,
    ) -> None:
        if int(max_tools_per_class) < 1:
            raise ValueError('max_tools_per_class must be at least one')
        if (
            not math.isfinite(reset_stamp_jump_sec)
            or reset_stamp_jump_sec <= 0.0
        ):
            raise ValueError('reset_stamp_jump_sec must be positive')
        self._max_tools_per_class = int(max_tools_per_class)
        self._reset_stamp_jump_ns = int(
            round(float(reset_stamp_jump_sec) * 1_000_000_000.0)
        )
        self._last_source_stamp_ns: int | None = None
        self._position_stabilizer = ToolPositionStabilizer(
            enabled=position_stabilization_enabled,
            deadband_m=position_deadband_m,
            smoothing_alpha=position_smoothing_alpha,
            max_jump_m=position_max_jump_m,
            relocation_confirmation_frames=(
                position_relocation_confirmation_frames
            ),
            relocation_consistency_m=position_relocation_consistency_m,
            max_missed_frames=position_max_missed_frames,
        )
        self._axis_stabilizer = ToolAxisStabilizer(
            enabled=axis_stabilization_enabled,
            flip_confirmation_frames=axis_flip_confirmation_frames,
            flip_dot_threshold=axis_flip_dot_threshold,
            pending_consistency_dot=axis_pending_consistency_dot,
            angular_deadband_rad=axis_angular_deadband_rad,
            smoothing_alpha=axis_smoothing_alpha,
            lock_opposite_axis=axis_lock_opposite,
            max_missed_frames=axis_max_missed_frames,
        )
        self._active_slots: set[str] = set()
        self._seen_slots: set[str] = set()
        self._created_total = 0
        self._expired_total = 0
        self._rejected_total = 0
        self._reset_total = 0

    @property
    def active_track_count(self) -> int:
        """Compatibility metric: number of selector slots in the last frame."""
        return len(self._active_slots)

    @property
    def created_total(self) -> int:
        """Return the number of distinct spatial selector slots ever observed."""
        return self._created_total

    @property
    def expired_total(self) -> int:
        """Return slot disappearances caused by a lower current cardinality."""
        return self._expired_total

    @property
    def rejected_total(self) -> int:
        """Return tools rejected by validation, ordering, or capacity rules."""
        return self._rejected_total

    @property
    def reset_total(self) -> int:
        """Return explicit gate or large source-clock reset count."""
        return self._reset_total

    @property
    def position_filter_active_count(self) -> int:
        return self._position_stabilizer.active_count

    @property
    def position_filter_held_total(self) -> int:
        return self._position_stabilizer.held_total

    @property
    def position_filter_smoothed_total(self) -> int:
        return self._position_stabilizer.smoothed_total

    @property
    def position_filter_outlier_held_total(self) -> int:
        return self._position_stabilizer.outlier_held_total

    @property
    def position_filter_relocation_total(self) -> int:
        return self._position_stabilizer.relocation_total

    @property
    def position_filter_association_reset_total(self) -> int:
        return self._position_stabilizer.association_reset_total

    @property
    def axis_filter_active_count(self) -> int:
        return self._axis_stabilizer.active_count

    @property
    def axis_filter_flip_held_total(self) -> int:
        return self._axis_stabilizer.flip_held_total

    @property
    def axis_filter_flip_confirmed_total(self) -> int:
        return self._axis_stabilizer.flip_confirmed_total

    @property
    def axis_filter_angular_held_total(self) -> int:
        return self._axis_stabilizer.angular_held_total

    @property
    def axis_filter_angular_smoothed_total(self) -> int:
        return self._axis_stabilizer.angular_smoothed_total

    @property
    def axis_filter_association_reset_total(self) -> int:
        return self._axis_stabilizer.association_reset_total

    @property
    def axis_filter_relocation_reset_total(self) -> int:
        return self._axis_stabilizer.relocation_reset_total

    def reset(self) -> None:
        """Forget the current snapshot without reusing persistent identity."""
        self._active_slots.clear()
        self._last_source_stamp_ns = None
        self._position_stabilizer.reset()
        self._axis_stabilizer.reset()
        self._reset_total += 1

    def assign(
        self,
        header: Any,
        tools: list[ToolPose],
        workspace_zone: str,
        *,
        horizontal_u_by_instance_id: Mapping[int, float] | None = None,
    ) -> list[ToolTfDecision]:
        """Build one decision per tool using current horizontal order."""
        source_stamp_ns = source_stamp_nanoseconds(header)
        if source_stamp_ns is None:
            self._rejected_total += len(tools)
            return [
                ToolTfDecision(None, 'SOURCE_STAMP_MISSING') for _ in tools
            ]

        if (
            self._last_source_stamp_ns is not None
            and source_stamp_ns < self._last_source_stamp_ns
        ):
            backwards_ns = self._last_source_stamp_ns - source_stamp_ns
            if backwards_ns >= self._reset_stamp_jump_ns:
                self._active_slots.clear()
                self._last_source_stamp_ns = None
                self._position_stabilizer.reset()
                self._axis_stabilizer.reset()
                self._reset_total += 1
            else:
                self._rejected_total += len(tools)
                return [
                    ToolTfDecision(
                        None, 'SELECTOR_SOURCE_STAMP_OUT_OF_ORDER'
                    )
                    for _ in tools
                ]

        labels = spatial_tool_child_frames(
            workspace_zone,
            tools,
            horizontal_u_by_instance_id=horizontal_u_by_instance_id,
        )
        current_slots = set(labels)
        previous_by_prefix: dict[str, set[str]] = {}
        current_by_prefix: dict[str, set[str]] = {}
        for slot in self._active_slots:
            previous_by_prefix.setdefault(slot.rsplit('#', 1)[0], set()).add(
                slot
            )
        for slot in current_slots:
            current_by_prefix.setdefault(slot.rsplit('#', 1)[0], set()).add(
                slot
            )
        for prefix in previous_by_prefix.keys() | current_by_prefix.keys():
            previous_group = previous_by_prefix.get(prefix, set())
            current_group = current_by_prefix.get(prefix, set())
            if previous_group != current_group:
                changed_slots = previous_group | current_group
                self._position_stabilizer.reset_keys(changed_slots)
                self._axis_stabilizer.reset_keys(changed_slots)

        decisions: list[ToolTfDecision] = []
        valid_position_slots: set[str] = set()
        for tool, child_frame_id in zip(tools, labels, strict=True):
            ordinal = int(child_frame_id.rsplit('#', 1)[1])
            if ordinal > self._max_tools_per_class:
                decisions.append(
                    ToolTfDecision(
                        None, 'SELECTOR_CLASS_CAPACITY_REACHED', child_frame_id
                    )
                )
                self._rejected_total += 1
                continue

            components, reason = _validated_components(header, tool)
            if components is None:
                decisions.append(
                    ToolTfDecision(None, reason, child_frame_id)
                )
                self._rejected_total += 1
                continue
            stabilized = self._position_stabilizer.update(
                child_frame_id, components.translation
            )
            if stabilized.reason == 'RELOCATION_CONFIRMED':
                self._axis_stabilizer.reset_for_relocation(child_frame_id)
            stabilized_axis = self._axis_stabilizer.update(
                child_frame_id,
                components.quaternion_xyzw,
                allow_flip_confirmation=bool(tool.orientation_valid),
            )
            components = _ConstrainedPoseComponents(
                components.parent_frame_id,
                stabilized.position_m,
                stabilized_axis.quaternion_xyzw,
            )
            valid_position_slots.add(child_frame_id)
            decisions.append(
                ToolTfDecision(
                    _transform_from_components(
                        header, child_frame_id, components
                    ),
                    'PUBLISHED',
                    child_frame_id,
                )
            )

        self._position_stabilizer.finish_frame(valid_position_slots)
        self._axis_stabilizer.finish_frame(valid_position_slots)

        new_slots = current_slots - self._seen_slots
        self._seen_slots.update(new_slots)
        self._created_total += len(new_slots)
        self._expired_total += len(self._active_slots - current_slots)
        self._active_slots = current_slots
        self._last_source_stamp_ns = source_stamp_ns
        return decisions


class ToolTfTracker:
    """Bounded nearest-neighbour assignment for dynamic camera-frame TFs.

    The detector's ``frame_local_instance_id`` is deliberately never used in a
    TF child name. It can change due to NMS ordering, so each camera/class has
    a short-lived track pool instead. Association uses only current metric
    camera-frame positions, with a maximum jump bound and a source-stamp TTL.
    Unmatched objects are rejected while active tracks of the same class still
    occupy the expected cardinality; this prefers a temporary missing TF to a
    false identity or unbounded frame churn.
    """

    def __init__(
        self,
        *,
        max_displacement_m: float,
        ttl_sec: float,
        max_tracks_per_class: int,
        reset_stamp_jump_sec: float,
    ) -> None:
        if not math.isfinite(max_displacement_m) or max_displacement_m <= 0.0:
            raise ValueError('max_displacement_m must be positive')
        if not math.isfinite(ttl_sec) or ttl_sec <= 0.0:
            raise ValueError('ttl_sec must be positive')
        if int(max_tracks_per_class) < 1:
            raise ValueError('max_tracks_per_class must be at least one')
        if (
            not math.isfinite(reset_stamp_jump_sec)
            or reset_stamp_jump_sec <= 0.0
        ):
            raise ValueError('reset_stamp_jump_sec must be positive')
        self._max_displacement_m = float(max_displacement_m)
        self._ttl_ns = int(round(float(ttl_sec) * 1_000_000_000.0))
        self._max_tracks_per_class = int(max_tracks_per_class)
        self._reset_stamp_jump_ns = int(
            round(float(reset_stamp_jump_sec) * 1_000_000_000.0)
        )
        self._groups: dict[tuple[str, int, str], _TrackGroup] = {}
        self._created_total = 0
        self._expired_total = 0
        self._rejected_total = 0
        self._reset_total = 0

    @property
    def active_track_count(self) -> int:
        """Return the currently allocated, not-yet-expired track count."""
        return sum(len(group.tracks) for group in self._groups.values())

    @property
    def created_total(self) -> int:
        """Return tracks allocated since the node process started."""
        return self._created_total

    @property
    def expired_total(self) -> int:
        """Return expired tracks removed since the node process started."""
        return self._expired_total

    @property
    def rejected_total(self) -> int:
        """Return candidates rejected by temporal identity safety rules."""
        return self._rejected_total

    @property
    def reset_total(self) -> int:
        """Return explicit gate or clock-reset tracker clears."""
        return self._reset_total

    def reset(self) -> None:
        """Clear active associations while preserving monotonic child IDs."""
        for group in self._groups.values():
            group.tracks.clear()
            group.last_source_stamp_ns = None
        self._reset_total += 1

    def assign(
        self, header: Any, tools: list[ToolPose], view: str
    ) -> list[ToolTfDecision]:
        """Return at most one source-stamped TF decision per supplied tool."""
        source_stamp_ns = source_stamp_nanoseconds(header)
        if source_stamp_ns is None:
            return [
                ToolTfDecision(None, 'SOURCE_STAMP_MISSING') for _ in tools
            ]

        decisions: list[ToolTfDecision | None] = [None] * len(tools)
        candidates_by_group: dict[
            tuple[str, int, str],
            list[tuple[int, ToolPose, _ConstrainedPoseComponents]],
        ] = {}
        camera = _frame_token(view, 'camera')
        for index, tool in enumerate(tools):
            components, reason = _validated_components(header, tool)
            if components is None:
                decisions[index] = ToolTfDecision(None, reason)
                continue
            group_key = (
                camera,
                int(tool.canonical_class_id),
                _frame_token(tool.class_name, 'tool'),
            )
            candidates_by_group.setdefault(group_key, []).append(
                (index, tool, components)
            )

        for group_key, candidates in candidates_by_group.items():
            self._assign_group(
                header,
                source_stamp_ns,
                group_key,
                candidates,
                decisions,
            )

        return [
            decision
            if decision is not None
            else ToolTfDecision(None, 'TRACK_ASSIGNMENT_INCOMPLETE')
            for decision in decisions
        ]

    def _assign_group(
        self,
        header: Any,
        source_stamp_ns: int,
        group_key: tuple[str, int, str],
        candidates: list[tuple[int, ToolPose, _ConstrainedPoseComponents]],
        decisions: list[ToolTfDecision | None],
    ) -> None:
        """Associate one same-camera/class candidate set to its active tracks."""
        group = self._groups.setdefault(group_key, _TrackGroup())
        previous_stamp_ns = group.last_source_stamp_ns
        if previous_stamp_ns is not None and source_stamp_ns < previous_stamp_ns:
            backwards_ns = previous_stamp_ns - source_stamp_ns
            if backwards_ns >= self._reset_stamp_jump_ns:
                group.tracks.clear()
                group.last_source_stamp_ns = None
                self._reset_total += 1
            else:
                for index, _tool, _components in candidates:
                    decisions[index] = ToolTfDecision(
                        None, 'TRACK_SOURCE_STAMP_OUT_OF_ORDER'
                    )
                self._rejected_total += len(candidates)
                return

        expired_track_ids = [
            track_id
            for track_id, track in group.tracks.items()
            if source_stamp_ns - track.last_source_stamp_ns > self._ttl_ns
        ]
        for track_id in expired_track_ids:
            del group.tracks[track_id]
        self._expired_total += len(expired_track_ids)

        active_track_ids = sorted(group.tracks)
        edges: list[tuple[float, int, int]] = []
        for candidate_index, (_index, _tool, components) in enumerate(candidates):
            for track_id in active_track_ids:
                distance = math.dist(
                    components.translation, group.tracks[track_id].position_m
                )
                if distance <= self._max_displacement_m:
                    edges.append((distance, track_id, candidate_index))
        edges.sort()

        matched_track_ids: set[int] = set()
        matched_candidate_indices: set[int] = set()
        for _distance, track_id, candidate_index in edges:
            if (
                track_id in matched_track_ids
                or candidate_index in matched_candidate_indices
            ):
                continue
            matched_track_ids.add(track_id)
            matched_candidate_indices.add(candidate_index)
            index, tool, components = candidates[candidate_index]
            group.tracks[track_id].position_m = components.translation
            group.tracks[track_id].last_source_stamp_ns = source_stamp_ns
            decisions[index] = self._published_decision(
                header, tool, group_key[0], track_id, components
            )

        unmatched_candidate_indices = [
            index
            for index in range(len(candidates))
            if index not in matched_candidate_indices
        ]
        allocation_count = max(0, len(candidates) - len(active_track_ids))
        capacity = self._max_tracks_per_class - len(group.tracks)
        allocated_candidate_indices = sorted(
            unmatched_candidate_indices,
            key=lambda index: (
                -float(candidates[index][1].class_confidence),
                candidates[index][0],
            ),
        )[: min(allocation_count, capacity)]
        for candidate_index in allocated_candidate_indices:
            index, tool, components = candidates[candidate_index]
            track_id = group.next_track_id
            group.next_track_id += 1
            group.tracks[track_id] = _Track(
                track_id, components.translation, source_stamp_ns
            )
            self._created_total += 1
            decisions[index] = self._published_decision(
                header, tool, group_key[0], track_id, components
            )

        allocated_set = set(allocated_candidate_indices)
        for candidate_index in unmatched_candidate_indices:
            if candidate_index in allocated_set:
                continue
            index, _tool, _components = candidates[candidate_index]
            reason = (
                'TRACK_CAPACITY_REACHED'
                if capacity <= 0
                else 'TRACK_DISPLACEMENT_EXCEEDED'
            )
            decisions[index] = ToolTfDecision(None, reason)
            self._rejected_total += 1
        group.last_source_stamp_ns = source_stamp_ns

    @staticmethod
    def _published_decision(
        header: Any,
        tool: ToolPose,
        view: str,
        track_id: int,
        components: _ConstrainedPoseComponents,
    ) -> ToolTfDecision:
        """Build a transform after the caller chose a stable temporal track."""
        child_frame_id = constrained_tool_child_frame(view, tool, track_id)
        return ToolTfDecision(
            _transform_from_components(header, child_frame_id, components),
            'PUBLISHED',
            child_frame_id,
        )
