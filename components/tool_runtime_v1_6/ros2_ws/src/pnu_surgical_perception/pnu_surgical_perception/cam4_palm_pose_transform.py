"""Publish source-stamped CAM4 palm poses without entering robot control.

``hand_keypoint_ros`` already performs the expensive part of this pipeline:
MediaPipe landmarking, registered RGB-D sampling, and construction of a
metric ``T_cam4_palm``. This node only validates that typed result, publishes
the camera-frame pose for observability, and composes it with the image-time
TF lookup into the current ``humanoid`` anchor.

The two outputs are deliberately perception-only ``PoseStamped`` topics. No
TF is broadcast and this node never calls a motion, handover, or controller
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from geometry_msgs.msg import PoseStamped, TransformStamped
from hand_keypoint_interfaces.msg import HandKeypoints

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


POSE_SCHEMA = "pnu.cam4_palm_pose.v1"


def result_qos() -> QoSProfile:
    """Keep only the newest typed observation; pose history is unsafe here."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def status_qos() -> QoSProfile:
    """Retain the latest explicit reason for a late-joining operator UI."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


@dataclass(frozen=True)
class PalmPoseComponents:
    """A finite, normalized metric palm pose with frame-local identity."""

    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    hand_index: int
    handedness: str


def source_stamp_nanoseconds(header: Any) -> int | None:
    """Return a positive source timestamp in nanoseconds, if available."""
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", 0))
    value = seconds * 1_000_000_000 + nanoseconds
    return value if value > 0 else None


def normalized_quaternion_values(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Return finite normalized quaternion values or reject the input."""
    if not all(math.isfinite(component) for component in values):
        return None
    norm = math.sqrt(sum(component * component for component in values))
    if norm <= 1e-9:
        return None
    return tuple(component / norm for component in values)


def normalized_quaternion_xyzw(
    value: Any,
) -> tuple[float, float, float, float] | None:
    """Read a finite unit quaternion from a ROS message-like object."""
    try:
        values = tuple(
            float(getattr(value, name)) for name in ("x", "y", "z", "w")
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return normalized_quaternion_values(values)


def finite_translation(value: Any) -> tuple[float, float, float] | None:
    """Return finite metric XYZ without silently filling bad values."""
    try:
        result = tuple(float(getattr(value, name)) for name in ("x", "y", "z"))
    except (AttributeError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def select_camera_palm(
    message: HandKeypoints,
    *,
    preferred_handedness: str = "Right",
    min_handedness_score: float = 0.0,
) -> tuple[PalmPoseComponents | None, str]:
    """Select exactly one real-depth hand palm or return an explicit reason.

    The transformed ``PoseStamped`` has no hand-index field. Requiring one
    valid selected palm prevents a 3-D pose from being attached to another
    hand in the same image.
    """
    if str(getattr(message, "depth_source", "")).strip().lower() != "real":
        return None, "DEPTH_SOURCE_NOT_REAL"
    header = getattr(message, "header", None)
    source_frame = str(getattr(header, "frame_id", "")).strip()
    if not source_frame or source_frame.startswith("/"):
        return None, "SOURCE_FRAME_INVALID"
    if source_stamp_nanoseconds(header) is None:
        return None, "SOURCE_STAMP_MISSING"

    preferred = str(preferred_handedness).strip().lower()
    if preferred not in {"", "left", "right"}:
        return None, "PREFERRED_HANDEDNESS_INVALID"
    minimum_score = max(0.0, min(1.0, float(min_handedness_score)))
    candidates: list[PalmPoseComponents] = []
    for hand in getattr(message, "hands", ()):
        if not bool(getattr(hand, "has_palm_6d", False)):
            continue
        has_handedness = bool(getattr(hand, "has_handedness", False))
        handedness = str(getattr(hand, "handedness_label", "")).strip()
        if preferred:
            if not has_handedness or handedness.lower() != preferred:
                continue
            try:
                handedness_score = float(
                    getattr(hand, "handedness_score", math.nan)
                )
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(handedness_score)
                or handedness_score < minimum_score
            ):
                continue
        translation = finite_translation(
            getattr(hand.palm_6d, "translation", None)
        )
        quaternion = normalized_quaternion_xyzw(
            getattr(hand.palm_6d, "orientation", None)
        )
        if translation is None or quaternion is None:
            continue
        try:
            hand_index = int(getattr(hand, "hand_index"))
        except (AttributeError, TypeError, ValueError):
            continue
        if hand_index < 0:
            continue
        candidates.append(PalmPoseComponents(
            translation=translation,
            quaternion_xyzw=quaternion,
            hand_index=hand_index,
            handedness=handedness if has_handedness else "Unknown",
        ))
    if not candidates:
        return None, "NO_VALID_SELECTED_PALM"
    if len(candidates) != 1:
        return None, "AMBIGUOUS_SELECTED_PALMS"
    return candidates[0], "SELECTED"


def quaternion_multiply_xyzw(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product for ROS's ``xyzw`` quaternion ordering."""
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector_xyzw(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector using a normalized quaternion without a TF broadcaster."""
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def compose_target_palm(
    target_from_source: TransformStamped,
    camera_palm: PalmPoseComponents,
) -> tuple[PalmPoseComponents | None, str]:
    """Compose ``T_target_source @ T_source_palm`` with finite-only geometry."""
    transform = getattr(target_from_source, "transform", None)
    transform_translation = finite_translation(
        getattr(transform, "translation", None)
    )
    transform_quaternion = normalized_quaternion_xyzw(
        getattr(transform, "rotation", None)
    )
    if transform_translation is None or transform_quaternion is None:
        return None, "TF_NONFINITE_OR_ZERO_QUATERNION"
    rotated = rotate_vector_xyzw(transform_quaternion, camera_palm.translation)
    translation = tuple(
        transform_translation[index] + rotated[index] for index in range(3)
    )
    quaternion = normalized_quaternion_values(quaternion_multiply_xyzw(
        transform_quaternion, camera_palm.quaternion_xyzw
    ))
    if quaternion is None or not all(math.isfinite(component) for component in translation):
        return None, "COMPOSED_POSE_INVALID"
    return PalmPoseComponents(
        translation=translation,
        quaternion_xyzw=quaternion,
        hand_index=camera_palm.hand_index,
        handedness=camera_palm.handedness,
    ), "COMPOSED"


def pose_stamped_from_components(
    source_header: Any,
    frame_id: str,
    components: PalmPoseComponents,
) -> PoseStamped:
    """Build a fresh pose message retaining the exact RGB source timestamp."""
    message = PoseStamped()
    message.header.stamp.sec = int(source_header.stamp.sec)
    message.header.stamp.nanosec = int(source_header.stamp.nanosec)
    message.header.frame_id = str(frame_id)
    message.pose.position.x, message.pose.position.y, message.pose.position.z = (
        components.translation
    )
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = components.quaternion_xyzw
    return message


class Cam4PalmPoseTransform(Node):
    """Transform typed CAM4 palm poses into the current humanoid anchor."""

    def __init__(self) -> None:
        super().__init__("cam4_palm_pose_transform")
        self.declare_parameter("input_topic", "/perception/cam_4/hand/keypoints")
        self.declare_parameter(
            "camera_pose_topic", "/perception/cam_4/hand/palm_pose_camera"
        )
        self.declare_parameter(
            "humanoid_pose_topic", "/perception/cam_4/hand/palm_pose_humanoid"
        )
        self.declare_parameter(
            "status_topic", "/perception/cam_4/hand/palm_pose_humanoid/status"
        )
        self.declare_parameter("target_frame", "humanoid")
        self.declare_parameter("preferred_handedness", "Right")
        self.declare_parameter("min_handedness_score", 0.0)
        self.declare_parameter("tf_timeout_sec", 0.05)

        self._input_topic = self._absolute_topic("input_topic")
        self._camera_pose_topic = self._absolute_topic("camera_pose_topic")
        self._humanoid_pose_topic = self._absolute_topic("humanoid_pose_topic")
        self._status_topic = self._absolute_topic("status_topic")
        self._target_frame = str(self.get_parameter("target_frame").value).strip()
        if not self._target_frame or self._target_frame.startswith("/"):
            raise ValueError("target_frame must be a non-empty relative TF frame")
        self._preferred_handedness = str(
            self.get_parameter("preferred_handedness").value
        ).strip()
        self._min_handedness_score = max(
            0.0,
            min(1.0, float(self.get_parameter("min_handedness_score").value)),
        )
        self._tf_timeout_sec = max(
            0.0,
            min(0.5, float(self.get_parameter("tf_timeout_sec").value)),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._camera_pose_publisher = self.create_publisher(
            PoseStamped, self._camera_pose_topic, result_qos()
        )
        self._humanoid_pose_publisher = self.create_publisher(
            PoseStamped, self._humanoid_pose_topic, result_qos()
        )
        self._status_publisher = self.create_publisher(
            String, self._status_topic, status_qos()
        )
        self._subscription = self.create_subscription(
            HandKeypoints, self._input_topic, self._on_keypoints, result_qos()
        )
        self.get_logger().info(
            "perception-only CAM4 palm transform: "
            f"input={self._input_topic}; camera_pose={self._camera_pose_topic}; "
            f"humanoid_pose={self._humanoid_pose_topic}; target_frame={self._target_frame}; "
            f"handedness={self._preferred_handedness or 'any'}"
        )

    def _absolute_topic(self, parameter_name: str) -> str:
        value = str(self.get_parameter(parameter_name).value).strip()
        if not value.startswith("/"):
            raise ValueError(f"{parameter_name} must be an absolute ROS topic")
        return value

    def _on_keypoints(self, message: HandKeypoints) -> None:
        camera_palm, reason = select_camera_palm(
            message,
            preferred_handedness=self._preferred_handedness,
            min_handedness_score=self._min_handedness_score,
        )
        if camera_palm is None:
            self._publish_status(
                message,
                ready=False,
                reason=reason,
                camera_published=False,
                humanoid_published=False,
            )
            return

        source_header = message.header
        source_frame = str(source_header.frame_id).strip()
        self._camera_pose_publisher.publish(pose_stamped_from_components(
            source_header, source_frame, camera_palm
        ))
        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame,
                source_frame,
                Time.from_msg(source_header.stamp),
                timeout=Duration(seconds=self._tf_timeout_sec),
            )
        except TransformException as exc:
            self._publish_status(
                message,
                ready=False,
                reason=f"TF_LOOKUP_FAILED:{type(exc).__name__}",
                camera_published=True,
                humanoid_published=False,
                components=camera_palm,
            )
            return
        if str(transform.header.frame_id).strip() != self._target_frame:
            self._publish_status(
                message,
                ready=False,
                reason="TF_TARGET_FRAME_MISMATCH",
                camera_published=True,
                humanoid_published=False,
                components=camera_palm,
            )
            return
        humanoid_palm, compose_reason = compose_target_palm(transform, camera_palm)
        if humanoid_palm is None:
            self._publish_status(
                message,
                ready=False,
                reason=compose_reason,
                camera_published=True,
                humanoid_published=False,
                components=camera_palm,
            )
            return
        self._humanoid_pose_publisher.publish(pose_stamped_from_components(
            source_header, self._target_frame, humanoid_palm
        ))
        self._publish_status(
            message,
            ready=True,
            reason="PUBLISHED",
            camera_published=True,
            humanoid_published=True,
            components=humanoid_palm,
        )

    def _publish_status(
        self,
        source: HandKeypoints,
        *,
        ready: bool,
        reason: str,
        camera_published: bool,
        humanoid_published: bool,
        components: PalmPoseComponents | None = None,
    ) -> None:
        header = getattr(source, "header", None)
        stamp = getattr(header, "stamp", None)
        payload: dict[str, Any] = {
            "schema": POSE_SCHEMA,
            "perception_only": True,
            "ready": bool(ready),
            "reason": str(reason),
            "source_frame": str(getattr(header, "frame_id", "")).strip(),
            "target_frame": self._target_frame,
            "source_stamp": {
                "sec": int(getattr(stamp, "sec", 0)),
                "nanosec": int(getattr(stamp, "nanosec", 0)),
            },
            "depth_source": str(getattr(source, "depth_source", "")).strip(),
            "camera_pose_published": bool(camera_published),
            "humanoid_pose_published": bool(humanoid_published),
            "hand_index": components.hand_index if components is not None else None,
            "handedness": components.handedness if components is not None else None,
        }
        status = String()
        status.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._status_publisher.publish(status)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: Cam4PalmPoseTransform | None = None
    try:
        node = Cam4PalmPoseTransform()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
