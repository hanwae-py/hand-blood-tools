"""ROS2 (Jazzy) hand-detection node — "section 2" of the camera -> AI
detection -> TF -> robot flow (see the SurgBlood ROS2 diagram this
package was scoped from). Subscribes to camera color/depth/camera_info
topics, runs MediaPipe + (real or monocular) depth backprojection +
palm-6D pose (the exact same math as scripts/run_hand_keypoints.py, via
the shared hand_keypoint_ros.core module), and publishes results.
Coordinate-frame transforms (TF) and robot control are OUT OF SCOPE here
— that's sections 3-5, someone else's node.

This is a MANAGED (lifecycle) node. Three perception algorithms — tool
detection, this one, and blood detection — share one RTX 3060, and they
must take turns rather than run together. A plain "enabled" flag would
stop this node processing frames but would leave MediaPipe and
Depth-Anything V2 resident in VRAM, which is the resource that actually
runs out. The lifecycle states map onto that directly:

  unconfigured  no model in VRAM. on_configure() loads them.
  inactive      models loaded, camera frames ignored. Re-activation is
                instant; costs VRAM.
  active        processing frames and publishing.
  on_cleanup()  releases the models and empties the CUDA cache.

surgical_task_coordinator drives those transitions. Run standalone with
autostart:=true (the default) and the node configures and activates
itself on startup, so single-node testing is unchanged.

Published topics (ARPA-H interface contract naming, all overridable by
parameter):
  /perception/cam_4/hand/keypoints (hand_keypoint_interfaces/HandKeypoints)
    — this frame's detected hand(s), typed: per hand, hand_index,
    handedness, joints_3d (metres, camera optical frame), joints_2d
    (pixels), kp_valid_depth, palm_6d. Carries both the 2D and 3D result
    together (see hand_keypoint_interfaces/msg/HandKeypoints.msg) — no
    separate 2D->3D conversion node is needed downstream.
  /perception/cam_4/hand/gestures (hand_keypoint_interfaces/HandGestureArray)
    — VIPLab top-view landmark classification from the same inference result
    as the landmarks: ``Closed_Fist`` and ``Open_Palm`` plus the fail-closed
    rejection category ``None``. Image and MediaPipe world landmarks are used;
    this remains RGB-only perception evidence and does not authorize a
    handover or robot action.
  /perception/cam_4/hand/facing (hand_keypoint_interfaces/HandFacingArray)
    — registered-depth palm-surface orientation relative to the calibrated
    surgical-table normal: ``PALM_UP``, ``PALM_DOWN``, ``EDGE``, or the
    fail-closed ``UNKNOWN`` state. This stays separate from HandKeypoints so
    existing subscribers retain their original DDS type hash.
  /perception/cam_4/hand/overlay/compressed (sensor_msgs/CompressedImage)
    — annotated debug visualization (skeleton + palm gizmo drawn on the
    input frame). CompressedImage, not raw Image, to match every other
    image topic in the contract. Disable with publish_overlay:=false.
  /perception/cam_4/hand/target_pose (geometry_msgs/PoseStamped)
    — the palm 6D pose the robot should hand a tool to, ready for a
    downstream TF/robot node to consume without JSON parsing. When
    robot_position is set this is that single handoff hand; otherwise it
    is the first detected hand that has a valid palm_6d.
  /perception/cam_4/hand/health (std_msgs/String, JSON, 1 Hz)
  /perception/cam_4/hand/diagnostics (std_msgs/String, 1 Hz)
    — the health/diagnostics pair every perception node in the contract
    publishes.

Depth input is accepted either as a raw ``sensor_msgs/Image`` or directly as
the synchronized camera provider's ``compressedDepth``
``sensor_msgs/CompressedImage``.  The latter is decoded in this node; no
republisher is required.  Decoding a 16UC1 PNG does not prove color alignment,
so metric 3D output remains invalid until ``depth_alignment_validated`` is
explicitly enabled after the RGB/depth frame and calibration contract is
approved. Native depth that is not already RGB-sized is registered into the
color camera with the same depth-to-color extrinsics as Tool; it is never
sampled by clipping RGB UVs into the native depth image.

Gesture publication is intentionally not gated by that synchronization. The
color subscriber feeds an RGB-only callback first; its one GestureRecognizer
result is cached and reused by the later depth/keypoint callback. Thus a depth
stall can stop 3-D keypoints without suppressing valid 2-D gesture evidence or
running MediaPipe twice for the same source frame.
"""
from collections import OrderedDict

import cv2
import json
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
import message_filters
from realsense2_camera_msgs.msg import Extrinsics
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3
from std_msgs.msg import String
from cv_bridge import CvBridge

from hand_keypoint_interfaces.msg import (
    Hand,
    HandFacing,
    HandFacingArray,
    HandGesture,
    HandGestureArray,
    HandKeypoints,
    PalmPose6D,
    Point2D,
)

from hand_keypoint_ros.core import (
    DEFAULT_DEPTH_MODEL,
    DEFAULT_GESTURE_RECOGNIZER_MODEL,
    GESTURE_RECOGNIZER_MODEL_SHA256,
    GESTURE_RECOGNIZER_MODEL_VERSION,
    gesture_rows_from_result, recognize_frame, robot_position_target_px,
    load_gesture_recognizer, load_mono_depth_model, run_mono_depth,
    process_frame,
)
from hand_keypoint_ros.topview_gesture import (
    GESTURE_PROFILES,
    classifier_metadata as gesture_classifier_metadata,
)
from hand_keypoint_ros.palm_facing import (
    ESTIMATOR_NAME as PALM_FACING_ESTIMATOR_NAME,
    ESTIMATOR_VERSION as PALM_FACING_ESTIMATOR_VERSION,
    PalmFacingEstimator,
    PalmFacingTemporalFilter,
    estimator_metadata,
)

_AUTO_DEPTH_DETECT_TIMEOUT_S = 3.0
_AUTO_DEPTH_DETECT_POLL_S = 0.2
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_RECOGNITION_CACHE_SIZE = 16
_MESSAGE_DELIVERY_KEY_CACHE_SIZE = 32
_PUBLISHED_GESTURE_STAMP_CACHE_SIZE = 32


def image_reader_qos() -> QoSProfile:
    """Latest frame only; matches local ingress image fan-out exactly."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )


def camera_info_qos() -> QoSProfile:
    """Reliable latest CameraInfo; never queue stale per-frame calibration."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )


def depth_to_color_extrinsics_qos() -> QoSProfile:
    """Match the latched RealSense depth-to-color calibration QoS."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


def camera_infos_share_pixel_grid(
    color_info, depth_info, native_shape, rgb_height, rgb_width,
):
    """Return True only when depth is proven to use the RGB pixel grid.

    Equal HxW alone is insufficient: CAM4's native color and depth images are
    both 1280x720 but have different optical frames and intrinsics.
    """
    if color_info is None or depth_info is None or len(native_shape) != 2:
        return False
    expected_shape = (int(rgb_height), int(rgb_width))
    if tuple(native_shape) != expected_shape:
        return False
    if (
        (int(color_info.height), int(color_info.width)) != expected_shape
        or (int(depth_info.height), int(depth_info.width)) != expected_shape
    ):
        return False
    color_frame = str(color_info.header.frame_id)
    depth_frame = str(depth_info.header.frame_id)
    if not color_frame or color_frame != depth_frame:
        return False
    if str(color_info.distortion_model) != str(depth_info.distortion_model):
        return False
    for field in ('k', 'd', 'r', 'p'):
        color_values = np.asarray(getattr(color_info, field), dtype=np.float64)
        depth_values = np.asarray(getattr(depth_info, field), dtype=np.float64)
        if color_values.shape != depth_values.shape:
            return False
        if not np.allclose(
            color_values, depth_values, rtol=0.0, atol=1e-9,
        ):
            return False
    return True


def decode_compressed_depth(message: CompressedImage) -> np.ndarray:
    """Decode a ROS compressedDepth 16UC1 PNG without changing its header.

    ``compressed_depth_image_transport`` prepends a small transport header to
    the PNG.  Searching for the PNG signature is compatible with both the
    current 12-byte header and recordings that omit it.
    """
    format_text = str(message.format or '')
    if 'compressedDepth' not in format_text or '16UC1' not in format_text:
        raise ValueError(
            'expected 16UC1 compressedDepth; received '
            f'{format_text!r}'
        )
    payload = bytes(message.data)
    png_offset = payload.find(_PNG_SIGNATURE)
    if png_offset < 0:
        raise ValueError('compressedDepth payload has no PNG signature')
    decoded = cv2.imdecode(
        np.frombuffer(payload[png_offset:], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if decoded is None:
        raise ValueError('OpenCV failed to decode compressedDepth PNG')
    if decoded.ndim != 2 or decoded.dtype != np.uint16:
        raise ValueError(
            'compressedDepth must decode to a 2-D uint16 image; '
            f'got shape={decoded.shape}, dtype={decoded.dtype}'
        )
    return np.ascontiguousarray(decoded)


def _row_hand_to_msg(hand):
    """Convert one process_frame() row_hands dict into a typed Hand message."""
    msg = Hand()
    msg.hand_index = int(hand['hand_index'])

    hd = hand['handedness']
    msg.has_handedness = hd is not None
    if hd is not None:
        msg.handedness_label = hd['label']
        msg.handedness_score = float(hd['score'])

    msg.joints_2d = [Point2D(u=float(u), v=float(v)) for u, v in hand['joints_2d']]
    msg.joints_3d = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in hand['joints_3d']]
    msg.kp_scores = [float(s) for s in hand['kp_scores']]
    msg.kp_valid_depth = [bool(v) for v in hand['kp_valid_depth']]

    palm = hand['palm_6d']
    msg.has_palm_6d = palm is not None
    if palm is not None:
        p6 = PalmPose6D()
        tx, ty, tz = palm['translation']
        p6.translation = Point(x=float(tx), y=float(ty), z=float(tz))
        qw, qx, qy, qz = palm['rotation_quat_wxyz']
        p6.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        p6.rotation_matrix = [float(v) for row in palm['rotation_matrix'] for v in row]
        msg.palm_6d = p6

    return msg


def _row_facing_to_msg(hand):
    """Convert one process_frame() row into a depth-backed facing result."""
    msg = HandFacing()
    msg.hand_index = int(hand['hand_index'])
    handedness = hand['handedness']
    msg.has_handedness = handedness is not None
    if handedness is not None:
        msg.handedness_label = handedness['label']
        msg.handedness_score = float(handedness['score'])

    facing = hand.get('palm_facing') or {}
    try:
        normal = np.asarray(
            facing.get('normal_cam', (0.0, 0.0, 0.0)), dtype=np.float64)
    except (TypeError, ValueError):
        normal = np.zeros(3, dtype=np.float64)
        normal_valid = False
    else:
        normal_valid = (
            normal.shape == (3,) and bool(np.all(np.isfinite(normal))))
    if not normal_valid:
        normal = np.zeros(3, dtype=np.float64)
    try:
        score = float(facing.get('palm_up_score', 0.0))
        residual = float(facing.get('plane_residual_m', 0.0))
        support_height = float(facing.get('support_height_m', 0.0))
    except (TypeError, ValueError):
        score = residual = support_height = 0.0
        scalar_values_valid = False
    else:
        scalar_values_valid = bool(np.all(np.isfinite(
            (score, residual, support_height))))
    if not scalar_values_valid:
        score = residual = support_height = 0.0
    msg.has_facing = bool(
        facing.get('has_facing', False)
        and normal_valid
        and scalar_values_valid
    )
    msg.facing_label = (
        str(facing.get('label', 'UNKNOWN'))
        if msg.has_facing else 'UNKNOWN'
    )
    msg.palm_up_score = score if msg.has_facing else 0.0
    msg.palm_normal_cam = Vector3(
        x=float(normal[0]), y=float(normal[1]), z=float(normal[2]))
    msg.palm_plane_residual_m = residual
    msg.support_height_m = support_height
    try:
        valid_depth_points = int(facing.get('valid_depth_points', 0))
    except (TypeError, ValueError):
        valid_depth_points = 0
    msg.valid_depth_points = max(0, min(5, valid_depth_points))
    rejection_reason = str(facing.get('rejection_reason', ''))
    if not msg.has_facing and not rejection_reason:
        rejection_reason = 'serialization_input_invalid'
    msg.rejection_reason = rejection_reason
    return msg


def _row_gesture_to_msg(hand):
    """Convert one process_frame() row into a typed gesture observation."""
    msg = HandGesture()
    msg.hand_index = int(hand['hand_index'])

    handedness = hand['handedness']
    msg.has_handedness = handedness is not None
    if handedness is not None:
        msg.handedness_label = handedness['label']
        msg.handedness_score = float(handedness['score'])

    gesture = hand.get('gesture')
    msg.has_classification = bool(gesture and gesture['has_gesture'])
    if msg.has_classification:
        msg.category_name = gesture['category_name']
        msg.score = float(gesture['score'])
    return msg


class HandDetectionNode(LifecycleNode):
    def __init__(self):
        super().__init__('hand_detection_node')

        # ---- parameters -----------------------------------------------
        self.declare_parameter('camera', 'cam_4')
        camera = str(self.get_parameter('camera').value).strip() or 'cam_4'
        synced = f'/synced/{camera}'
        out = f'/perception/{camera}/hand'
        self.declare_parameter(
            'color_topic', f'{synced}/color/image_raw/compressed')
        self.declare_parameter('color_transport', 'compressed')  # compressed | raw
        self.declare_parameter(
            'depth_topic',
            f'{synced}/depth/image_rect_raw/compressedDepth')
        self.declare_parameter(
            'depth_transport', 'compressed_depth')  # compressed_depth | raw
        self.declare_parameter('depth_alignment_validated', False)
        self.declare_parameter('depth_registration_backend', 'numpy')
        self.declare_parameter(
            'depth_registration_allow_sticky_numpy_fallback', False)
        self.declare_parameter(
            'camera_info_topic', f'{synced}/color/camera_info')
        self.declare_parameter(
            'depth_camera_info_topic', f'{synced}/depth/camera_info')
        self.declare_parameter(
            'extrinsics_topic', f'{synced}/extrinsics/depth_to_color')
        self.declare_parameter('require_extrinsics_topic', True)
        self.declare_parameter('depth_scale_m_per_unit', 0.001)
        self.declare_parameter('depth_to_color_rotation', [float('nan')] * 9)
        self.declare_parameter('depth_to_color_translation_m', [float('nan')] * 3)
        self.declare_parameter('minimum_depth_to_color_baseline_m', 0.02)
        self.declare_parameter('maximum_depth_to_color_baseline_m', 0.12)
        self.declare_parameter(
            'expected_depth_to_color_translation_direction',
            [-1.0, 0.0, 0.0],
        )
        self.declare_parameter(
            'minimum_depth_to_color_direction_cosine', 0.95)
        self.declare_parameter(
            'depth_to_color_rotation_orthonormal_tolerance', 1e-4)
        self.declare_parameter(
            'depth_to_color_rotation_determinant_tolerance', 1e-4)
        self.declare_parameter(
            'expected_color_frame', f'{camera}_color_optical_frame')
        self.declare_parameter(
            'expected_depth_frame', f'{camera}_depth_optical_frame')
        self.declare_parameter('calibration_version', '')
        # Additive capability: fixed-camera deployments must opt in with a
        # calibrated table normal, handedness signs, and provenance string.
        self.declare_parameter('palm_facing_enabled', False)
        self.declare_parameter(
            'palm_table_up_normal', [0.0, 0.0, -1.0])
        self.declare_parameter('palm_support_plane_offset_m', float('nan'))
        self.declare_parameter('palm_normal_sign_left', -1.0)
        self.declare_parameter('palm_normal_sign_right', 1.0)
        self.declare_parameter(
            'palm_facing_expected_flip_handedness', False)
        self.declare_parameter('palm_facing_enter_cosine', 0.75)
        self.declare_parameter('palm_facing_hold_cosine', 0.60)
        self.declare_parameter('palm_facing_filter_alpha', 0.50)
        self.declare_parameter('palm_facing_max_plane_residual_m', 0.012)
        self.declare_parameter('palm_facing_min_span_m', 0.025)
        self.declare_parameter('palm_facing_min_handedness_score', 0.60)
        self.declare_parameter('palm_facing_calibration_version', '')
        self.declare_parameter('palm_facing_handedness_mapping_version', '')
        self.declare_parameter('palm_facing_mapping_verified', False)
        self.declare_parameter('palm_facing_min_support_height_m', 0.008)
        self.declare_parameter('palm_facing_max_support_height_m', 0.25)

        # Output topic names follow /perception/<camera>/<task>/<detail>.
        self.declare_parameter('keypoints_topic', f'{out}/keypoints')
        self.declare_parameter('gesture_topic', f'{out}/gestures')
        self.declare_parameter('facing_topic', f'{out}/facing')
        self.declare_parameter('overlay_topic', f'{out}/overlay/compressed')
        self.declare_parameter('target_pose_topic', f'{out}/target_pose')
        self.declare_parameter('health_topic', f'{out}/health')
        self.declare_parameter('diagnostics_topic', f'{out}/diagnostics')

        self.declare_parameter(
            'depth_source', 'auto')  # auto | real | mono | rgb_only
        self.declare_parameter('rgb_fallback_when_real_depth_missing', False)
        self.declare_parameter('real_depth_fallback_timeout_sec', 0.5)
        self.declare_parameter('depth_model', DEFAULT_DEPTH_MODEL)
        self.declare_parameter(
            'gesture_model', DEFAULT_GESTURE_RECOGNIZER_MODEL)
        self.declare_parameter('gesture_profile', 'topview')
        self.declare_parameter('max_hands', 4)
        self.declare_parameter('cpu_only', False)
        self.declare_parameter('flip_handedness', False)
        self.declare_parameter('forced_handedness_label', '')
        self.declare_parameter('region_x_min', 0.0)
        self.declare_parameter('region_x_max', 1.0)
        self.declare_parameter('region_y_min', 0.0)
        self.declare_parameter('region_y_max', 1.0)
        self.declare_parameter('robot_position', '')  # '' = disabled (default: keep every hand)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('publish_target_pose', True)
        self.declare_parameter('sync_slop_sec', 0.05)
        self.declare_parameter('sync_queue_size', 2)
        self.declare_parameter('overlay_jpeg_quality', 80)
        self.declare_parameter('input_stale_timeout_sec', 2.0)
        # true: configure+activate ourselves at startup, so running this node
        # on its own behaves exactly as it did before it became a lifecycle
        # node. Set false when surgical_task_coordinator owns the turn-taking.
        self.declare_parameter('autostart', True)

        self.bridge = CvBridge()

        # Everything below is created in on_configure() and released in
        # on_cleanup(); nothing here holds GPU memory while unconfigured.
        self.mp = self.hand_det = None
        self.torch = self.depth_processor = self.depth_model = self.device = self.dtype = None
        self.sync = None
        self.rgb_fallback_sync = None
        self._subs = []
        self._raw_subs = []
        self._active = False
        self._depth_info = None
        self._depth_info_key = None
        self._registrar = None
        self._registrar_key = None
        self._registration_helpers = 'unloaded'
        self._extrinsics_validator = 'unloaded'
        self._depth_to_color_extrinsics = None
        self._reference_extrinsics = None
        self._received_extrinsics = 0
        self._rejected_extrinsics = 0
        self._last_extrinsics_error = ''
        self._metric_depth_ready = False
        self._last_metric_depth_at = None
        self._aligned_depth_valid_fraction = 0.0
        self._depth_registration_mode = 'disabled'
        self._last_depth_registration_ms = 0.0
        self._last_depth_registration_gpu_ms = 0.0
        self._last_rgb_depth_delta_ns = None
        self.depth_registration_backend = 'numpy'
        self.depth_registration_allow_sticky_numpy_fallback = False
        self.palm_facing_estimator = None
        self.palm_facing_filter = None
        self.palm_facing_mapping_verified = False
        self.forced_handedness_label = ''
        self.gesture_profile = 'topview'
        self.gesture_classifier_metadata = gesture_classifier_metadata(
            self.gesture_profile)

        self.use_real_depth = False
        self.use_mono_depth = False
        self.rgb_only = False
        self.rgb_fallback_when_real_depth_missing = False
        self.real_depth_fallback_timeout_sec = 0.5
        self.depth_source_label = 'unconfigured'

        # ---- diagnostics counters ---------------------------------------
        self._frames = 0
        self._hands_last = 0
        self._gestures_last = 0
        self._palm_facing_valid_last = 0
        self._palm_facing_errors = 0
        self._palm_facing_rejections_last = {}
        self._gesture_frames = 0
        self._gesture_errors = 0
        self._last_process_ms = 0.0
        self._last_gesture_process_ms = 0.0
        self._last_frame_at = None
        self._last_gesture_frame_at = None
        self._errors = 0
        self._rate_sample_at = time.monotonic()
        self._rate_sample_frames = 0
        self._processed_hz = 0.0
        self._gesture_rate_sample_at = time.monotonic()
        self._gesture_rate_sample_frames = 0
        self._gesture_processed_hz = 0.0
        self._last_gesture_error_code = ''
        self._last_gesture_error_message = ''
        self.input_stale_timeout_sec = float(
            self.get_parameter('input_stale_timeout_sec').value)
        self.depth_transport = str(
            self.get_parameter('depth_transport').value).strip().lower()
        self.depth_alignment_validated = bool(
            self.get_parameter('depth_alignment_validated').value)
        self.extrinsics_topic = str(
            self.get_parameter('extrinsics_topic').value).strip()
        self.require_extrinsics_topic = bool(
            self.get_parameter('require_extrinsics_topic').value)
        self.publish_target_pose = bool(
            self.get_parameter('publish_target_pose').value)
        self._rgb_publisher_seen = False
        self._depth_publisher_seen = False
        self._last_real_depth_at = None
        self._last_depth_frame_id = ''
        self._last_source_stamp_sec = 0
        self._last_source_stamp_nanosec = 0
        self._last_source_frame_id = ''
        # MediaPipe VIDEO mode rejects timestamps that go backwards. Live
        # cameras normally provide monotonic header stamps, while rosbag
        # --loop deliberately starts its recorded stamps from the beginning.
        self._last_mp_ts_ms = None
        self._last_source_ts_ms = None
        self._mp_timestamp_offset_ms = 0
        self._mp_frame_interval_ms = 67  # CAM4 recording is about 15 Hz
        self._last_error_code = 'MODEL_NOT_CONFIGURED'
        self._last_error_message = 'hand detector is not configured'
        self._last_gesture_error_code = 'MODEL_NOT_CONFIGURED'
        self._last_gesture_error_message = 'gesture recognizer is not configured'
        self._recognition_cache = OrderedDict()
        self._message_delivery_keys = OrderedDict()
        self._published_gesture_stamps = OrderedDict()
        self._color_delivery_sequence = 0
        self._sync_cached_inference_ms = 0.0

        # Lifecycle publishers: rclpy silently drops publishes on these while
        # the node is not ACTIVE, which is exactly the gating we want.
        g = self.get_parameter
        reliable_output = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        overlay_output = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        status_output = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        diagnostics_output = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub_keypoints = self.create_lifecycle_publisher(
            HandKeypoints, g('keypoints_topic').value, reliable_output)
        self.pub_gestures = self.create_lifecycle_publisher(
            HandGestureArray, g('gesture_topic').value, reliable_output)
        self.pub_facing = self.create_lifecycle_publisher(
            HandFacingArray, g('facing_topic').value, reliable_output)
        self.pub_overlay = self.create_lifecycle_publisher(
            CompressedImage, g('overlay_topic').value, overlay_output)
        self.pub_target_pose = self.create_lifecycle_publisher(
            PoseStamped, g('target_pose_topic').value, reliable_output)

        # Health and diagnostics are NOT lifecycle publishers on purpose:
        # a node that has been deactivated to free VRAM still needs to be
        # able to say so. They are plain publishers and keep reporting in
        # every state.
        self.pub_health = self.create_publisher(
            String, g('health_topic').value, status_output)
        self.pub_diagnostics = self.create_publisher(
            String, g('diagnostics_topic').value, diagnostics_output)
        self.create_timer(1.0, self._publish_health)

        self.get_logger().info(
            f'hand_detection_node created (unconfigured). '
            f'keypoints={g("keypoints_topic").value} '
            f'gestures={g("gesture_topic").value} '
            f'facing={g("facing_topic").value}')

    # ---------------------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Load the models and wire up the camera subscriptions."""
        g = self.get_parameter
        self.color_topic = g('color_topic').value
        self.color_transport = str(g('color_transport').value).strip().lower()
        if self.color_transport not in ('compressed', 'raw'):
            self.get_logger().error(
                'color_transport must be "compressed" or "raw"; received '
                f'{self.color_transport!r}')
            return TransitionCallbackReturn.FAILURE
        self.depth_topic = g('depth_topic').value
        self.depth_transport = str(g('depth_transport').value).strip().lower()
        if self.depth_transport not in ('compressed_depth', 'raw'):
            self.get_logger().error(
                'depth_transport must be "compressed_depth" or "raw"; received '
                f'{self.depth_transport!r}')
            return TransitionCallbackReturn.FAILURE
        self.depth_alignment_validated = bool(
            g('depth_alignment_validated').value)
        self.depth_registration_backend = str(
            g('depth_registration_backend').value).strip().lower()
        if self.depth_registration_backend not in ('numpy', 'cuda'):
            self.get_logger().error(
                'depth_registration_backend must be "numpy" or "cuda"')
            return TransitionCallbackReturn.FAILURE
        self.depth_registration_allow_sticky_numpy_fallback = bool(
            g('depth_registration_allow_sticky_numpy_fallback').value)
        self.camera_info_topic = g('camera_info_topic').value
        self.depth_camera_info_topic = str(g('depth_camera_info_topic').value)
        self.extrinsics_topic = str(g('extrinsics_topic').value).strip()
        self.require_extrinsics_topic = bool(
            g('require_extrinsics_topic').value)
        self.depth_scale_m_per_unit = float(g('depth_scale_m_per_unit').value)
        self.minimum_extrinsics_baseline_m = float(
            g('minimum_depth_to_color_baseline_m').value)
        self.maximum_extrinsics_baseline_m = float(
            g('maximum_depth_to_color_baseline_m').value)
        self.expected_extrinsics_direction = list(
            g('expected_depth_to_color_translation_direction').value)
        self.minimum_extrinsics_direction_cosine = float(
            g('minimum_depth_to_color_direction_cosine').value)
        self.extrinsics_orthonormal_tolerance = float(
            g('depth_to_color_rotation_orthonormal_tolerance').value)
        self.extrinsics_determinant_tolerance = float(
            g('depth_to_color_rotation_determinant_tolerance').value)
        self.expected_color_frame = str(g('expected_color_frame').value)
        self.expected_depth_frame = str(g('expected_depth_frame').value)
        self.calibration_version = str(
            g('calibration_version').value).strip()
        self.palm_facing_enabled = bool(
            g('palm_facing_enabled').value)
        self.palm_table_up_normal = list(
            g('palm_table_up_normal').value)
        self.palm_support_plane_offset_m = float(
            g('palm_support_plane_offset_m').value)
        self.palm_normal_signs = {
            'Left': float(g('palm_normal_sign_left').value),
            'Right': float(g('palm_normal_sign_right').value),
        }
        self.palm_facing_expected_flip_handedness = bool(
            g('palm_facing_expected_flip_handedness').value)
        self.palm_facing_enter_cosine = float(
            g('palm_facing_enter_cosine').value)
        self.palm_facing_hold_cosine = float(
            g('palm_facing_hold_cosine').value)
        self.palm_facing_filter_alpha = float(
            g('palm_facing_filter_alpha').value)
        self.palm_facing_max_plane_residual_m = float(
            g('palm_facing_max_plane_residual_m').value)
        self.palm_facing_min_span_m = float(
            g('palm_facing_min_span_m').value)
        self.palm_facing_min_handedness_score = float(
            g('palm_facing_min_handedness_score').value)
        self.palm_facing_calibration_version = str(
            g('palm_facing_calibration_version').value).strip()
        self.palm_facing_handedness_mapping_version = str(
            g('palm_facing_handedness_mapping_version').value).strip()
        self.palm_facing_mapping_verified = bool(
            g('palm_facing_mapping_verified').value)
        self.palm_facing_min_support_height_m = float(
            g('palm_facing_min_support_height_m').value)
        self.palm_facing_max_support_height_m = float(
            g('palm_facing_max_support_height_m').value)
        depth_source_param = g('depth_source').value
        self.rgb_fallback_when_real_depth_missing = bool(
            g('rgb_fallback_when_real_depth_missing').value)
        self.real_depth_fallback_timeout_sec = float(
            g('real_depth_fallback_timeout_sec').value)
        if self.real_depth_fallback_timeout_sec <= 0.0:
            self.get_logger().error(
                'real_depth_fallback_timeout_sec must be > 0')
            return TransitionCallbackReturn.FAILURE
        self.depth_model_name = g('depth_model').value
        self.gesture_model_path = os.path.expanduser(
            str(g('gesture_model').value))
        self.gesture_profile = str(
            g('gesture_profile').value).strip().lower()
        if self.gesture_profile not in GESTURE_PROFILES:
            self.get_logger().error(
                'gesture_profile must be one of '
                f'{GESTURE_PROFILES}; received {self.gesture_profile!r}')
            return TransitionCallbackReturn.FAILURE
        self.gesture_classifier_metadata = gesture_classifier_metadata(
            self.gesture_profile)
        self.max_hands = g('max_hands').value
        self.cpu_only = g('cpu_only').value
        self.flip_handedness = g('flip_handedness').value
        self.forced_handedness_label = str(
            g('forced_handedness_label').value).strip()
        if self.forced_handedness_label not in ('', 'Left', 'Right'):
            self.get_logger().error(
                'forced_handedness_label must be empty, Left, or Right')
            return TransitionCallbackReturn.FAILURE
        if self.forced_handedness_label and bool(self.flip_handedness):
            self.get_logger().error(
                'forced_handedness_label and flip_handedness cannot both '
                'be enabled')
            return TransitionCallbackReturn.FAILURE
        if self.forced_handedness_label:
            self.get_logger().warn(
                'camera-view handedness constraint active: every detected '
                f'hand is published as {self.forced_handedness_label}; '
                'raw MediaPipe handedness is ignored')
        if (
            self.palm_facing_enabled
            and bool(self.flip_handedness)
            != self.palm_facing_expected_flip_handedness
        ):
            self.get_logger().error(
                'flip_handedness does not match the palm-facing handedness '
                'mapping profile')
            return TransitionCallbackReturn.FAILURE
        self.region = (g('region_x_min').value, g('region_x_max').value,
                       g('region_y_min').value, g('region_y_max').value)
        self.robot_position = g('robot_position').value or None
        self.publish_overlay = g('publish_overlay').value
        self.publish_target_pose = bool(g('publish_target_pose').value)
        self.jpeg_quality = int(g('overlay_jpeg_quality').value)
        self.input_stale_timeout_sec = float(g('input_stale_timeout_sec').value)
        if self.input_stale_timeout_sec <= 0.0:
            self.get_logger().error('input_stale_timeout_sec must be > 0')
            return TransitionCallbackReturn.FAILURE
        sync_slop = float(g('sync_slop_sec').value)
        sync_queue_size = int(g('sync_queue_size').value)
        if sync_slop < 0.0:
            self.get_logger().error('sync_slop_sec must be >= 0')
            return TransitionCallbackReturn.FAILURE
        if not 1 <= sync_queue_size <= 8:
            self.get_logger().error('sync_queue_size must be in [1, 8]')
            return TransitionCallbackReturn.FAILURE
        self.sync_queue_size = sync_queue_size

        if self.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.get_logger().info(
                'cpu_only:=true -- forcing CPU for MediaPipe and any mono-depth model')

        # ---- depth source resolution ------------------------------------
        # 'auto' mirrors run_hand_keypoints.py's find_depth_h5(): prefer
        # real depth if it's actually available, else fall back to mono.
        # Here "available" means a publisher for depth_topic is visible in
        # the ROS graph -- poll briefly since node startup order isn't
        # guaranteed (the camera driver may still be coming up).
        if depth_source_param == 'real':
            self.use_real_depth = True
            self.use_mono_depth = False
            self.rgb_only = False
        elif depth_source_param == 'mono':
            self.use_real_depth = False
            self.use_mono_depth = True
            self.rgb_only = False
        elif depth_source_param == 'rgb_only':
            self.use_real_depth = False
            self.use_mono_depth = False
            self.rgb_only = True
        else:
            self.use_real_depth = self._wait_for_publisher(self.depth_topic)
            self.use_mono_depth = not self.use_real_depth
            self.rgb_only = False
            if not self.use_real_depth:
                self.get_logger().warn(
                    f'depth_source=auto: no publisher seen for "{self.depth_topic}" within '
                    f'{_AUTO_DEPTH_DETECT_TIMEOUT_S}s -- falling back to monocular depth '
                    f'(Depth-Anything V2). If the depth publisher just starts later than this '
                    f'node, reconfigure this node, or set depth_source:=real explicitly.')
        self._depth_to_color_extrinsics = None
        self._reference_extrinsics = None
        self._last_extrinsics_error = ''
        self._metric_depth_ready = False
        self._last_metric_depth_at = None
        self._aligned_depth_valid_fraction = 0.0
        self._depth_registration_mode = 'disabled'
        self._last_depth_registration_ms = 0.0
        self._last_depth_registration_gpu_ms = 0.0
        self._last_rgb_depth_delta_ns = None
        if self.use_real_depth and self.depth_alignment_validated:
            if (
                not np.isfinite(self.depth_scale_m_per_unit)
                or self.depth_scale_m_per_unit <= 0.0
            ):
                self.get_logger().error(
                    'depth_scale_m_per_unit must be finite and > 0')
                return TransitionCallbackReturn.FAILURE
            if not self.calibration_version:
                self.get_logger().error(
                    'calibration_version is required for metric hand depth')
                return TransitionCallbackReturn.FAILURE
            if self.require_extrinsics_topic and not self.extrinsics_topic:
                self.get_logger().error(
                    'extrinsics_topic is required for native metric hand depth')
                return TransitionCallbackReturn.FAILURE
            if not self.require_extrinsics_topic:
                try:
                    self._reference_extrinsics = (
                        self._validate_depth_to_color_extrinsics(
                            g('depth_to_color_rotation').value,
                            g('depth_to_color_translation_m').value,
                        )
                    )
                except (ImportError, ValueError) as exc:
                    self.get_logger().error(
                        f'configured depth-to-color reference is invalid: {exc}')
                    return TransitionCallbackReturn.FAILURE
        if self.use_real_depth and self.depth_alignment_validated:
            self.depth_source_label = 'REAL DEPTH (REGISTERED TO RGB)'
        elif self.use_real_depth:
            self.depth_source_label = 'REAL DEPTH (ALIGNMENT PENDING)'
        elif self.use_mono_depth:
            self.depth_source_label = 'MONO DEPTH (Depth-Anything V2)'
        else:
            self.depth_source_label = 'RGB ONLY (NO DEPTH CLAIMS)'
        self.get_logger().info(f'depth source: {self.depth_source_label}')

        self.palm_facing_estimator = None
        self.palm_facing_filter = None
        if self.palm_facing_enabled:
            if not (self.use_real_depth and self.depth_alignment_validated):
                self.get_logger().warn(
                    'palm-facing requested but registered real depth is not '
                    'enabled; the facing topic will remain silent')
            else:
                try:
                    self.palm_facing_estimator = PalmFacingEstimator(
                        table_up_normal=self.palm_table_up_normal,
                        support_plane_offset_m=(
                            self.palm_support_plane_offset_m),
                        handedness_signs=self.palm_normal_signs,
                        enter_cosine=self.palm_facing_enter_cosine,
                        max_plane_residual_m=(
                            self.palm_facing_max_plane_residual_m),
                        min_palm_span_m=self.palm_facing_min_span_m,
                        min_handedness_score=(
                            self.palm_facing_min_handedness_score),
                        calibration_version=(
                            self.palm_facing_calibration_version),
                        handedness_mapping_version=(
                            self.palm_facing_handedness_mapping_version),
                        min_support_height_m=(
                            self.palm_facing_min_support_height_m),
                        max_support_height_m=(
                            self.palm_facing_max_support_height_m),
                    )
                    self.palm_facing_filter = PalmFacingTemporalFilter(
                        enter_cosine=self.palm_facing_enter_cosine,
                        hold_cosine=self.palm_facing_hold_cosine,
                        alpha=self.palm_facing_filter_alpha,
                        table_up_normal=self.palm_table_up_normal,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self.get_logger().error(
                        f'palm-facing configuration is invalid: {exc}')
                    return TransitionCallbackReturn.FAILURE
                self.get_logger().info(
                    'palm-facing ready: '
                    f'calibration={self.palm_facing_calibration_version} '
                    f'table_up={self.palm_table_up_normal} '
                    f'signs={self.palm_normal_signs}')
                if not self.palm_facing_mapping_verified:
                    self.get_logger().warn(
                        'palm-facing observations are enabled for overlay '
                        'validation, but the physical left/right sign mapping '
                        'is not yet verified')

        # ---- models (this is the part that costs VRAM) --------------------
        try:
            self.mp, self.hand_det = load_gesture_recognizer(
                self.max_hands,
                model_path=self.gesture_model_path,
                cpu_only=self.cpu_only,
            )
            if self.use_mono_depth:
                (self.torch, self.depth_processor, self.depth_model,
                 self.device, self.dtype) = load_mono_depth_model(
                     self.depth_model_name, cpu_only=self.cpu_only)
        except Exception:
            import traceback
            self.get_logger().error('model loading failed:\n' + traceback.format_exc())
            self._release_models()
            return TransitionCallbackReturn.FAILURE

        # ---- subscribers (time-synchronized) ---------------------------
        rgb_qos = image_reader_qos()
        info_qos = camera_info_qos()
        depth_qos = image_reader_qos()
        color_type = CompressedImage if self.color_transport == 'compressed' else Image
        color_sub = message_filters.Subscriber(
            self, color_type, self.color_topic, qos_profile=rgb_qos)
        # Reuse the same DDS subscription for an immediate RGB-only gesture
        # callback and for the existing RGB/depth/CameraInfo synchronizer.
        # registerCallback() adds a SimpleFilter consumer; it does not create
        # a second ROS subscription or duplicate network traffic.
        color_sub.registerCallback(self._on_color_for_gesture)
        info_sub = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=info_qos)
        self._subs = [color_sub, info_sub]
        self._raw_subs = []
        if self.use_real_depth:
            depth_type = (
                CompressedImage
                if self.depth_transport == 'compressed_depth'
                else Image
            )
            depth_sub = message_filters.Subscriber(
                self, depth_type, self.depth_topic, qos_profile=depth_qos)
            self._subs.append(depth_sub)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [color_sub, depth_sub, info_sub],
                queue_size=sync_queue_size,
                slop=sync_slop,
            )
            self.sync.registerCallback(self._on_synced_real)
            if self.rgb_fallback_when_real_depth_missing:
                self.rgb_fallback_sync = (
                    message_filters.ApproximateTimeSynchronizer(
                        [color_sub, info_sub],
                        queue_size=sync_queue_size,
                        slop=sync_slop,
                    ))
                self.rgb_fallback_sync.registerCallback(
                    self._on_synced_rgb_fallback)
            depth_info_sub = self.create_subscription(
                CameraInfo,
                self.depth_camera_info_topic,
                self._on_depth_info,
                info_qos)
            self._raw_subs.append(depth_info_sub)
            if self.extrinsics_topic:
                extrinsics_sub = self.create_subscription(
                    Extrinsics,
                    self.extrinsics_topic,
                    self._on_depth_to_color_extrinsics,
                    depth_to_color_extrinsics_qos(),
                )
                self._raw_subs.append(extrinsics_sub)
        else:
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [color_sub, info_sub],
                queue_size=sync_queue_size,
                slop=sync_slop,
            )
            self.sync.registerCallback(self._on_synced_mono)

        self.get_logger().info(
            f'configured. color={self.color_topic} ({self.color_transport}) '
            + (
                f'depth={self.depth_topic} ({self.depth_transport}, '
                f'alignment_validated={self.depth_alignment_validated}) '
                if self.use_real_depth else (
                    '(monocular depth) ' if self.use_mono_depth
                    else '(RGB-only; 2D keypoints) '
                )
            )
            + f'camera_info={self.camera_info_topic}'
            + (
                f' depth_camera_info={self.depth_camera_info_topic}'
                if self.use_real_depth else ''
            )
            + (
                f' extrinsics={self.extrinsics_topic}'
                if self.use_real_depth and self.extrinsics_topic else ''
            )
            + (f' | robot_position={self.robot_position}' if self.robot_position else ''))
        self._last_error_code = ''
        self._last_error_message = ''
        # A newly configured MediaPipe instance starts its own VIDEO timeline.
        self._last_mp_ts_ms = None
        self._last_source_ts_ms = None
        self._mp_timestamp_offset_ms = 0
        self._recognition_cache.clear()
        self._message_delivery_keys.clear()
        self._published_gesture_stamps.clear()
        if self.palm_facing_filter is not None:
            self.palm_facing_filter.reset()
        self._color_delivery_sequence = 0
        self._last_gesture_frame_at = None
        self._last_gesture_error_code = ''
        self._last_gesture_error_message = ''
        self._last_real_depth_at = None
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Our turn: start processing the frames we are already subscribed to."""
        self._active = True
        self._frames = 0
        self._gestures_last = 0
        self._palm_facing_valid_last = 0
        self._gesture_frames = 0
        self._rate_sample_at = time.monotonic()
        self._rate_sample_frames = 0
        self._processed_hz = 0.0
        self._gesture_rate_sample_at = time.monotonic()
        self._gesture_rate_sample_frames = 0
        self._gesture_processed_hz = 0.0
        self._last_gesture_frame_at = None
        self._last_gesture_error_code = ''
        self._last_gesture_error_message = ''
        self._last_real_depth_at = None
        self.get_logger().info('ACTIVE: processing camera frames')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Turn is over. Frames are ignored, but the models stay in VRAM so
        the next activation is instant."""
        self._active = False
        self._recognition_cache.clear()
        self._message_delivery_keys.clear()
        self._published_gesture_stamps.clear()
        if self.palm_facing_filter is not None:
            self.palm_facing_filter.reset()
        self.get_logger().info(
            f'INACTIVE: stopped processing after {self._frames} frames '
            f'(models still loaded)')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Release the models so their VRAM goes back to the pool."""
        self._active = False
        self._teardown_subscriptions()
        self._release_models()
        self.palm_facing_estimator = None
        self.palm_facing_filter = None
        self.depth_source_label = 'unconfigured'
        self._last_error_code = 'MODEL_NOT_CONFIGURED'
        self._last_error_message = 'hand detector was cleaned up'
        self._last_gesture_frame_at = None
        self._last_gesture_error_code = 'MODEL_NOT_CONFIGURED'
        self._last_gesture_error_message = 'gesture recognizer was cleaned up'
        self.get_logger().info('cleaned up: models released, VRAM freed')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._teardown_subscriptions()
        self._release_models()
        self.palm_facing_estimator = None
        self.palm_facing_filter = None
        self._last_gesture_frame_at = None
        self._last_gesture_error_code = 'MODEL_NOT_CONFIGURED'
        self._last_gesture_error_message = 'gesture recognizer was shut down'
        return TransitionCallbackReturn.SUCCESS

    def _teardown_subscriptions(self):
        self.sync = None
        self.rgb_fallback_sync = None
        for sub in self._subs:
            # message_filters.Subscriber wraps the real rclpy subscription.
            inner = getattr(sub, 'sub', None)
            if inner is not None:
                try:
                    self.destroy_subscription(inner)
                except Exception:                          # noqa: BLE001
                    self.get_logger().debug('subscription teardown raised; ignoring')
        for sub in self._raw_subs:
            try:
                self.destroy_subscription(sub)
            except Exception:                              # noqa: BLE001
                self.get_logger().debug('subscription teardown raised; ignoring')
        self._subs = []
        self._raw_subs = []
        self._depth_info = None
        self._depth_info_key = None
        self._discard_depth_registrar()
        self._registration_helpers = 'unloaded'

    def _release_models(self):
        """Drop every reference that could be pinning GPU memory."""
        if self.hand_det is not None:
            try:
                self.hand_det.close()
            except Exception:                              # noqa: BLE001
                self.get_logger().debug('MediaPipe close() raised; ignoring')
        self.mp = self.hand_det = None
        self._recognition_cache.clear()
        self._message_delivery_keys.clear()
        self._published_gesture_stamps.clear()

        torch = self.torch
        self.depth_processor = self.depth_model = None
        self.device = self.dtype = None
        self.torch = None
        if torch is not None and torch.cuda.is_available():
            # Without this the freed tensors stay in torch's caching
            # allocator and nvidia-smi still shows them as used.
            torch.cuda.empty_cache()

    def _wait_for_publisher(self, topic_name):
        """True once topic_name has at least one live PUBLISHER.

        count_publishers(), not "is the name in the graph". On the live
        integration LAN a contract monitor may already subscribe to the
        configured endpoint even when its provider is offline, so the name
        can be present in get_topic_names_and_types() with zero publishers.
        Treating that as "real depth is available" would make
        depth_source=auto subscribe to a silent topic and prevent the
        synchronizer from producing frames.
        """
        deadline = time.monotonic() + _AUTO_DEPTH_DETECT_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.count_publishers(topic_name) > 0:
                return True
            time.sleep(_AUTO_DEPTH_DETECT_POLL_S)
        return self.count_publishers(topic_name) > 0

    @staticmethod
    def _source_key(message):
        return (
            int(message.header.stamp.sec),
            int(message.header.stamp.nanosec),
            str(message.header.frame_id),
        )

    def _decode_color(self, color_msg):
        if isinstance(color_msg, CompressedImage):
            return self.bridge.compressed_imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8')
        return self.bridge.imgmsg_to_cv2(
            color_msg, desired_encoding='bgr8')

    def _recognize_color(self, color_msg, *, direct_delivery=False):
        """Return one result for this delivered color message.

        The direct RGB callback allocates a monotonically increasing delivery
        key, so repeated Header stamps from a short rosbag loop are never
        mistaken for an already-published frame. The later synchronizer sees
        the same Python message object and reuses that result.
        """
        source_key = self._source_key(color_msg)
        message_id = id(color_msg)
        if not direct_delivery:
            mapped = self._message_delivery_keys.get(message_id)
            if mapped is not None and mapped[0] == source_key:
                delivery_key = mapped[1]
                cached = self._recognition_cache.get(delivery_key)
                if cached is not None:
                    self._recognition_cache.move_to_end(delivery_key)
                    frame_bgr, result, inference_ms = cached
                    return frame_bgr, result, inference_ms, True, delivery_key
                raise RuntimeError(
                    'MediaPipe result cache was evicted before the '
                    'RGB/depth synchronizer consumed this frame')

        self._color_delivery_sequence += 1
        delivery_key = (self._color_delivery_sequence,) + source_key
        self._message_delivery_keys[message_id] = (source_key, delivery_key)
        self._message_delivery_keys.move_to_end(message_id)
        while (
            len(self._message_delivery_keys)
            > _MESSAGE_DELIVERY_KEY_CACHE_SIZE
        ):
            self._message_delivery_keys.popitem(last=False)

        frame_bgr = self._decode_color(color_msg)
        source_ts_ms = (
            color_msg.header.stamp.sec * 1000
            + color_msg.header.stamp.nanosec // 1_000_000
        )
        ts_ms = self._mediapipe_timestamp_ms(source_ts_ms)
        started = time.monotonic()
        result = recognize_frame(
            frame_bgr, self.hand_det, self.mp, ts_ms)
        inference_ms = (time.monotonic() - started) * 1000.0
        self._recognition_cache[delivery_key] = (
            frame_bgr, result, inference_ms)
        while len(self._recognition_cache) > _RECOGNITION_CACHE_SIZE:
            self._recognition_cache.popitem(last=False)
        return frame_bgr, result, inference_ms, False, delivery_key

    def _publish_gestures_for_result(
        self, color_msg, frame_bgr, result, delivery_key,
    ):
        """Publish at most once for a source frame, without depth input."""
        if delivery_key in self._published_gesture_stamps:
            self._published_gesture_stamps.move_to_end(delivery_key)
            return False

        height, width = frame_bgr.shape[:2]
        target_px = None
        if self.robot_position:
            target_px, _ = robot_position_target_px(
                self.robot_position, width, height)
        gesture_rows = gesture_rows_from_result(
            result,
            width,
            height,
            region=self.region,
            target_px=target_px,
            flip_handedness=self.flip_handedness,
            forced_handedness_label=self.forced_handedness_label,
            gesture_profile=self.gesture_profile,
        )

        message = HandGestureArray()
        message.header = color_msg.header
        metadata = self.gesture_classifier_metadata
        message.model_name = metadata['name']
        message.model_version = metadata['version']
        message.model_asset_sha256 = metadata['sha256']
        message.supported_gestures = list(metadata['supported_gestures'])
        message.rejection_category = 'None'
        message.hands = [_row_gesture_to_msg(hand) for hand in gesture_rows]
        self.pub_gestures.publish(message)

        self._published_gesture_stamps[delivery_key] = None
        while (
            len(self._published_gesture_stamps)
            > _PUBLISHED_GESTURE_STAMP_CACHE_SIZE
        ):
            self._published_gesture_stamps.popitem(last=False)
        self._gesture_frames += 1
        self._gestures_last = sum(
            1 for hand in message.hands if hand.has_classification)
        self._last_gesture_frame_at = time.monotonic()
        self._last_gesture_error_code = ''
        self._last_gesture_error_message = ''
        self.get_logger().info(
            'published top-view gesture observations: '
            f'{len(message.hands)} hands, '
            f'{self._gestures_last} classifications',
            throttle_duration_sec=1.0,
        )
        return True

    def _on_color_for_gesture(self, color_msg):
        """RGB-only fast path; depth loss must not stop gesture evidence."""
        if not self._active:
            return
        started = time.monotonic()
        try:
            frame_bgr, result, _, _, delivery_key = self._recognize_color(
                color_msg, direct_delivery=True)
            self._publish_gestures_for_result(
                color_msg, frame_bgr, result, delivery_key)
        except Exception:
            self._gesture_errors += 1
            import traceback
            self._last_gesture_error_code = 'GESTURE_PROCESSING_ERROR'
            self._last_gesture_error_message = (
                traceback.format_exc().splitlines()[-1])
            self.get_logger().error(
                'exception in RGB-only gesture path:\n'
                + traceback.format_exc())
        finally:
            self._last_gesture_process_ms = (
                time.monotonic() - started) * 1000.0

    # ---------------------------------------------------------------------
    # health / diagnostics -- published in every lifecycle state
    # ---------------------------------------------------------------------

    def _lifecycle_state_name(self):
        try:
            return self._state_machine.current_state[1]
        except Exception:                                  # noqa: BLE001
            return 'unknown'

    def _publish_health(self):
        state = self._lifecycle_state_name()
        camera = str(self.get_parameter('camera').value).strip() or 'camera'
        camera_code = camera.upper()
        now = time.monotonic()
        elapsed = now - self._rate_sample_at
        if elapsed > 0.0:
            self._processed_hz = (
                (self._frames - self._rate_sample_frames) / elapsed)
            self._rate_sample_at = now
            self._rate_sample_frames = self._frames
        gesture_elapsed = now - self._gesture_rate_sample_at
        if gesture_elapsed > 0.0:
            self._gesture_processed_hz = (
                (self._gesture_frames - self._gesture_rate_sample_frames)
                / gesture_elapsed
            )
            self._gesture_rate_sample_at = now
            self._gesture_rate_sample_frames = self._gesture_frames
        stale_s = (None if self._last_frame_at is None
                   else round(now - self._last_frame_at, 2))
        gesture_stale_s = (
            None if self._last_gesture_frame_at is None
            else round(now - self._last_gesture_frame_at, 2)
        )
        # count_publishers(), not "name present in the graph" -- see the note
        # in _wait_for_publisher(): a subscriber-only topic name is visible in
        # get_topic_names_and_types() and would be read as a live source.
        depth_topic = self.get_parameter('depth_topic').value
        self._rgb_publisher_seen = self.count_publishers(
            self.get_parameter('color_topic').value) > 0
        self._depth_publisher_seen = self.count_publishers(depth_topic) > 0
        rgb_ready = (
            self._rgb_publisher_seen
            and stale_s is not None
            and stale_s <= self.input_stale_timeout_sec
        )
        gesture_rgb_ready = (
            self._rgb_publisher_seen
            and gesture_stale_s is not None
            and gesture_stale_s <= self.input_stale_timeout_sec
        )
        active_extrinsics, extrinsics_source = (
            self._active_depth_to_color_extrinsics())
        extrinsics_ready = active_extrinsics is not None
        metric_depth_stale_s = (
            None if self._last_metric_depth_at is None
            else round(now - self._last_metric_depth_at, 2)
        )
        registration_ready = bool(
            self._metric_depth_ready
            and metric_depth_stale_s is not None
            and metric_depth_stale_s <= self.input_stale_timeout_sec
        )
        palm_facing_meta = estimator_metadata(
            self.palm_facing_estimator, self.palm_facing_filter)
        handedness_policy = (
            'forced_camera_constraint'
            if self.forced_handedness_label
            else ('mediapipe_flipped' if self.flip_handedness else 'mediapipe')
        )
        palm_facing_observation_ready = bool(
            self._active
            and self.palm_facing_estimator is not None
            and registration_ready
        )
        palm_facing_ready = bool(
            palm_facing_observation_ready
            and self.palm_facing_mapping_verified
        )
        if self.use_real_depth:
            depth_stream_ready = bool(
                self._depth_publisher_seen
                and self._last_real_depth_at is not None
                and now - self._last_real_depth_at <= self.input_stale_timeout_sec
            )
            depth_ready = bool(
                depth_stream_ready
                and self.depth_alignment_validated
                and registration_ready
            )
        elif self.use_mono_depth:
            # Monocular fallback: depth is inferred by Depth-Anything V2 from
            # the RGB frame we just processed, so there is no second stream
            # whose freshness could be checked. Demanding one reported every
            # mono-depth run as degraded even while keypoints were flowing.
            depth_stream_ready = self.depth_model is not None and rgb_ready
            depth_ready = depth_stream_ready
        else:
            # RGB-only auxiliary views intentionally make no depth claim. They
            # remain useful for 2-D hand shape and cross-view occlusion recovery.
            depth_stream_ready = False
            depth_ready = False
        model_ready = self.hand_det is not None
        rgb_fallback_active = bool(
            self.use_real_depth
            and self.rgb_fallback_when_real_depth_missing
            and rgb_ready
            and not depth_ready
        )
        inference_ready = bool(
            self._active
            and rgb_ready
            and model_ready
            and (depth_ready or self.rgb_only or rgb_fallback_active)
        )
        # Top-view gesture recognition is RGB/landmark-only. Keep its
        # readiness independent of the still-unapproved metric depth path.
        gesture_inference_ready = bool(
            self._active and gesture_rgb_ready and model_ready)
        status = 'ok' if inference_ready else 'degraded'
        error_code = self._last_error_code
        error_message = self._last_error_message
        if not error_code:
            if not model_ready:
                error_code = 'MODEL_NOT_CONFIGURED'
                error_message = 'hand detector model is not configured'
            elif not rgb_ready:
                error_code = f'{camera_code}_RGB_STALE_OR_MISSING'
                error_message = f'fresh {camera} RGB has not been processed'
            elif not (self.rgb_only or rgb_fallback_active) and not depth_ready:
                if self.use_real_depth and not self._depth_publisher_seen:
                    # Distinguishable from "the stream stalled": the aligned
                    # -depth provider does not exist on this LAN yet.
                    error_code = f'{camera_code}_ALIGNED_DEPTH_NOT_PROVIDED'
                    error_message = f'no publisher on {depth_topic}'
                elif self.use_real_depth and depth_stream_ready and not self.depth_alignment_validated:
                    error_code = f'{camera_code}_DEPTH_ALIGNMENT_NOT_VALIDATED'
                    error_message = (
                        'compressed depth is received but RGB/depth alignment '
                        'and calibration are not approved')
                elif (
                    self.use_real_depth
                    and self.depth_alignment_validated
                    and self._last_extrinsics_error
                ):
                    error_code = f'{camera_code}_DEPTH_TO_COLOR_EXTRINSICS_INVALID'
                    error_message = self._last_extrinsics_error
                elif (
                    self.use_real_depth
                    and self.depth_alignment_validated
                    and self.require_extrinsics_topic
                    and not extrinsics_ready
                ):
                    error_code = f'{camera_code}_DEPTH_TO_COLOR_EXTRINSICS_MISSING'
                    error_message = (
                        f'no valid latched extrinsics on {self.extrinsics_topic}')
                elif self.use_real_depth and self.depth_alignment_validated:
                    error_code = f'{camera_code}_DEPTH_REGISTRATION_NOT_READY'
                    error_message = (
                        'native depth has not produced a fresh RGB-grid map')
                else:
                    error_code = f'{camera_code}_DEPTH_STALE_OR_MISSING'
                    error_message = 'real or monocular depth is not ready'
        gesture_error_code = self._last_gesture_error_code
        gesture_error_message = self._last_gesture_error_message
        if not gesture_error_code:
            if not model_ready:
                gesture_error_code = 'MODEL_NOT_CONFIGURED'
                gesture_error_message = 'gesture recognizer is not configured'
            elif not self._active:
                gesture_error_code = 'GESTURE_NODE_NOT_ACTIVE'
                gesture_error_message = 'gesture recognizer is not active'
            elif not gesture_rgb_ready:
                gesture_error_code = f'{camera_code}_GESTURE_RGB_STALE_OR_MISSING'
                gesture_error_message = (
                    f'fresh {camera} RGB has not reached the gesture path')
        registrar = self._registrar
        registration_backend_active = str(
            getattr(registrar, 'backend_name', 'uninitialized'))
        registration_backend_version = str(
            getattr(registrar, 'backend_version', ''))
        registration_fallback_active = bool(
            getattr(registrar, 'fallback_active', False))
        registration_fallback_count = int(
            getattr(registrar, 'fallback_count', 0))
        registration_backend_error = str(
            getattr(registrar, 'last_backend_error', ''))
        if registration_fallback_active:
            status = 'degraded'
            if not error_code:
                error_code = f'{camera_code}_DEPTH_REGISTRATION_CUDA_FALLBACK'
                error_message = (
                    'CUDA depth registration fell back to NumPy: '
                    + registration_backend_error)
        self.pub_health.publish(String(data=json.dumps({
            'schema': 'pnu.hand_keypoint_health.v1',
            'node': self.get_name(),
            'lifecycle_state': state,
            'status': status,
            'ready': inference_ready,
            'rgb_ready': rgb_ready,
            'depth_stream_ready': depth_stream_ready,
            'depth_ready': depth_ready,
            'rgb_fallback_active': rgb_fallback_active,
            'depth_alignment_validated': self.depth_alignment_validated,
            'depth_registration_ready': registration_ready,
            'depth_registration_mode': self._depth_registration_mode,
            'depth_registration_backend_requested': (
                self.depth_registration_backend),
            'depth_registration_backend_active': (
                registration_backend_active),
            'depth_registration_backend_version': (
                registration_backend_version),
            'depth_registration_degraded': registration_fallback_active,
            'depth_registration_fallback_active': (
                registration_fallback_active),
            'depth_registration_fallback_count': (
                registration_fallback_count),
            'depth_registration_last_backend_error': (
                registration_backend_error),
            'depth_registration_latency_ms': round(
                self._last_depth_registration_ms, 3),
            'depth_registration_gpu_ms': round(
                self._last_depth_registration_gpu_ms, 3),
            'rgb_depth_timestamp_delta_ns': self._last_rgb_depth_delta_ns,
            'depth_to_color_extrinsics_required': (
                self.require_extrinsics_topic),
            'depth_to_color_extrinsics_ready': extrinsics_ready,
            'depth_to_color_extrinsics_source': extrinsics_source,
            'seconds_since_last_metric_depth': metric_depth_stale_s,
            'aligned_depth_valid_fraction': round(
                self._aligned_depth_valid_fraction, 4),
            'model_ready': model_ready,
            'hand_inference_ready': inference_ready,
            'gesture_model_ready': model_ready,
            'gesture_inference_ready': gesture_inference_ready,
            'gesture_status': (
                'ok' if gesture_inference_ready else 'degraded'),
            'gesture_rgb_ready': gesture_rgb_ready,
            'handedness_policy': handedness_policy,
            'forced_handedness_label': self.forced_handedness_label,
            'palm_facing_ready': palm_facing_ready,
            'palm_facing_observation_ready': palm_facing_observation_ready,
            'palm_facing_mapping_verified': (
                self.palm_facing_mapping_verified),
            'palm_facing_valid_hands_last_frame': (
                self._palm_facing_valid_last),
            'palm_facing_estimator': palm_facing_meta,
            'palm_facing_topic': self.get_parameter('facing_topic').value,
            'palm_facing_errors': self._palm_facing_errors,
            'palm_facing_rejections_last_frame': (
                self._palm_facing_rejections_last),
            'gesture_topic': self.get_parameter('gesture_topic').value,
            'gesture_profile': self.gesture_profile,
            'gesture_model_version': (
                self.gesture_classifier_metadata['version']),
            'gesture_model_asset_sha256': (
                self.gesture_classifier_metadata['sha256']),
            'gesture_classifier': (
                self.gesture_classifier_metadata['name']),
            'mediapipe_task_model_version': (
                GESTURE_RECOGNIZER_MODEL_VERSION),
            'mediapipe_task_asset_sha256': (
                GESTURE_RECOGNIZER_MODEL_SHA256),
            'depth_source': self.depth_source_label,
            'effective_depth_source': (
                'rgb_only_fallback' if rgb_fallback_active
                else ('real' if self.use_real_depth and depth_ready
                      else ('mono' if self.use_mono_depth else 'rgb_only'))
            ),
            'rgb_only': self.rgb_only,
            'seconds_since_last_frame': stale_s,
            'seconds_since_last_gesture_frame': gesture_stale_s,
            'input_stale_timeout_sec': self.input_stale_timeout_sec,
            'last_error_code': error_code,
            'last_error_message': error_message,
            'last_gesture_error_code': gesture_error_code,
            'last_gesture_error_message': gesture_error_message,
        })))
        self.pub_diagnostics.publish(String(data=json.dumps({
            'node': self.get_name(),
            'lifecycle_state': state,
            'frames_processed': self._frames,
            'hands_last_frame': self._hands_last,
            'gesture_frames_processed': self._gesture_frames,
            'gestures_last_frame': self._gestures_last,
            'last_process_ms': round(self._last_process_ms, 1),
            'last_gesture_process_ms': round(
                self._last_gesture_process_ms, 1),
            'processed_hz_1s': round(self._processed_hz, 2),
            'gesture_processed_hz_1s': round(
                self._gesture_processed_hz, 2),
            'errors': self._errors,
            'gesture_errors': self._gesture_errors,
            'depth_source': self.depth_source_label,
            'rgb_only': self.rgb_only,
            'rgb_fallback_active': rgb_fallback_active,
            'real_depth_fallback_timeout_sec': (
                self.real_depth_fallback_timeout_sec),
            'color_topic': self.get_parameter('color_topic').value,
            'color_transport': self.get_parameter('color_transport').value,
            'camera_info_topic': self.get_parameter('camera_info_topic').value,
            'depth_topic': depth_topic,
            'depth_transport': self.get_parameter('depth_transport').value,
            'depth_alignment_validated': self.depth_alignment_validated,
            'depth_registration_ready': registration_ready,
            'depth_registration_mode': self._depth_registration_mode,
            'depth_registration_backend_requested': (
                self.depth_registration_backend),
            'depth_registration_backend_active': (
                registration_backend_active),
            'depth_registration_backend_version': (
                registration_backend_version),
            'depth_registration_degraded': registration_fallback_active,
            'depth_registration_fallback_active': (
                registration_fallback_active),
            'depth_registration_fallback_count': (
                registration_fallback_count),
            'depth_registration_last_backend_error': (
                registration_backend_error),
            'depth_registration_latency_ms': round(
                self._last_depth_registration_ms, 3),
            'depth_registration_gpu_ms': round(
                self._last_depth_registration_gpu_ms, 3),
            'rgb_depth_timestamp_delta_ns': self._last_rgb_depth_delta_ns,
            'depth_to_color_extrinsics_topic': self.extrinsics_topic,
            'depth_to_color_extrinsics_required': (
                self.require_extrinsics_topic),
            'depth_to_color_extrinsics_ready': extrinsics_ready,
            'depth_to_color_extrinsics_source': extrinsics_source,
            'depth_to_color_extrinsics_received': self._received_extrinsics,
            'depth_to_color_extrinsics_rejected': self._rejected_extrinsics,
            'depth_to_color_extrinsics_error': self._last_extrinsics_error,
            'seconds_since_last_metric_depth': metric_depth_stale_s,
            'aligned_depth_valid_fraction': round(
                self._aligned_depth_valid_fraction, 4),
            'depth_frame_id': self._last_depth_frame_id,
            'max_hands': self.get_parameter('max_hands').value,
            'cpu_only': self.get_parameter('cpu_only').value,
            'gesture_topic': self.get_parameter('gesture_topic').value,
            'gesture_profile': self.gesture_profile,
            'gesture_model_version': (
                self.gesture_classifier_metadata['version']),
            'gesture_model_asset_sha256': (
                self.gesture_classifier_metadata['sha256']),
            'gesture_classifier': (
                self.gesture_classifier_metadata['name']),
            'mediapipe_task_model_version': (
                GESTURE_RECOGNIZER_MODEL_VERSION),
            'mediapipe_task_asset_sha256': (
                GESTURE_RECOGNIZER_MODEL_SHA256),
            'gesture_inference_ready': gesture_inference_ready,
            'gesture_rgb_ready': gesture_rgb_ready,
            'handedness_policy': handedness_policy,
            'forced_handedness_label': self.forced_handedness_label,
            'palm_facing_ready': palm_facing_ready,
            'palm_facing_observation_ready': palm_facing_observation_ready,
            'palm_facing_mapping_verified': (
                self.palm_facing_mapping_verified),
            'palm_facing_valid_hands_last_frame': (
                self._palm_facing_valid_last),
            'palm_facing_estimator': palm_facing_meta,
            'palm_facing_topic': self.get_parameter('facing_topic').value,
            'palm_facing_errors': self._palm_facing_errors,
            'palm_facing_rejections_last_frame': (
                self._palm_facing_rejections_last),
            'publish_target_pose': self.publish_target_pose,
            'source_stamp_sec': self._last_source_stamp_sec,
            'source_stamp_nanosec': self._last_source_stamp_nanosec,
            'frame_id': self._last_source_frame_id,
            'error_code': error_code,
            'error_message': error_message,
            'gesture_error_code': gesture_error_code,
            'gesture_error_message': gesture_error_message,
        })))

    # ---------------------------------------------------------------------

    def _on_synced_mono(self, color_msg, info_msg):
        self._process(color_msg, info_msg, depth_msg=None)

    def _on_synced_real(self, color_msg, depth_msg, info_msg):
        self._process(color_msg, info_msg, depth_msg=depth_msg)

    def _on_synced_rgb_fallback(self, color_msg, info_msg):
        """Keep 2-D evidence live until real depth successfully pairs."""
        if not self.rgb_fallback_when_real_depth_missing:
            return
        now = time.monotonic()
        if (
            self._last_real_depth_at is not None
            and now - self._last_real_depth_at
            <= self.real_depth_fallback_timeout_sec
        ):
            return
        self._process(color_msg, info_msg, depth_msg=None)

    @staticmethod
    def _depth_camera_info_key(message):
        """Calibration identity excluding the per-frame header timestamp."""

        return (
            int(message.width),
            int(message.height),
            str(message.distortion_model),
            tuple(float(value) for value in message.k),
            tuple(float(value) for value in message.d),
            str(message.header.frame_id),
        )

    def _on_depth_info(self, message):
        key = self._depth_camera_info_key(message)
        self._depth_info = message
        if key == self._depth_info_key:
            return
        self._depth_info_key = key
        self._discard_depth_registrar()

    def _discard_depth_registrar(self):
        registrar = self._registrar
        self._registrar = None
        self._registrar_key = None
        close = getattr(registrar, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                self.get_logger().warn(
                    'Hand depth registrar cleanup raised; continuing')

    def _validate_depth_to_color_extrinsics(self, rotation, translation):
        if self._extrinsics_validator == 'unloaded':
            from pnu_surgical_perception.depth_to_color_extrinsics import (
                validate_depth_to_color_extrinsics,
            )
            self._extrinsics_validator = validate_depth_to_color_extrinsics
        return self._extrinsics_validator(
            rotation,
            translation,
            minimum_baseline_m=self.minimum_extrinsics_baseline_m,
            maximum_baseline_m=self.maximum_extrinsics_baseline_m,
            expected_translation_direction=self.expected_extrinsics_direction,
            minimum_direction_cosine=(
                self.minimum_extrinsics_direction_cosine),
            orthonormal_tolerance=self.extrinsics_orthonormal_tolerance,
            determinant_tolerance=self.extrinsics_determinant_tolerance,
        )

    def _on_depth_to_color_extrinsics(self, message):
        """Accept only a physically plausible latched RealSense transform."""
        self._received_extrinsics += 1
        try:
            extrinsics = self._validate_depth_to_color_extrinsics(
                message.rotation, message.translation)
        except (ImportError, TypeError, ValueError) as exc:
            self._rejected_extrinsics += 1
            self._depth_to_color_extrinsics = None
            self._discard_depth_registrar()
            self._metric_depth_ready = False
            self._last_extrinsics_error = str(exc)
            self.get_logger().error(
                f'rejected depth-to-color extrinsics: {exc}')
            return
        self._depth_to_color_extrinsics = extrinsics
        self._discard_depth_registrar()
        self._metric_depth_ready = False
        self._last_extrinsics_error = ''
        self.get_logger().info(
            'accepted depth-to-color extrinsics '
            f'(baseline={extrinsics.baseline_m:.6f} m)')

    def _active_depth_to_color_extrinsics(self):
        if self._depth_to_color_extrinsics is not None:
            return self._depth_to_color_extrinsics, 'live'
        if not self.require_extrinsics_topic and self._reference_extrinsics is not None:
            return self._reference_extrinsics, 'reference'
        return None, ''

    def _registration_api(self):
        if self._registration_helpers == 'unloaded':
            try:
                from pnu_surgical_tool.depth_registration import (
                    finite_vector_or_none,
                    metric_depth_in_rgb_frame,
                    registrar_from_camera_messages,
                )
                self._registration_helpers = (
                    finite_vector_or_none,
                    metric_depth_in_rgb_frame,
                    registrar_from_camera_messages,
                )
            except ImportError as exc:
                self.get_logger().warn(
                    'Hand depth-to-color registration unavailable '
                    f'(set PYTHONPATH to tool algorithm/src): {exc}')
                self._registration_helpers = None
        return self._registration_helpers

    def _depth_to_color_registrar(self, color_info, native_shape, rgb_height, rgb_width):
        helpers = self._registration_api()
        if helpers is None or len(native_shape) != 2:
            return None
        _, _, registrar_from_camera_messages = helpers
        depth_info = self._depth_info
        extrinsics, extrinsics_source = (
            self._active_depth_to_color_extrinsics())
        version = self.calibration_version
        if (
            color_info is None
            or depth_info is None
            or extrinsics is None
            or not version
        ):
            return None
        if int(color_info.width) != rgb_width or int(color_info.height) != rgb_height:
            return None
        if (int(depth_info.height), int(depth_info.width)) != tuple(native_shape):
            return None
        color_frame = str(color_info.header.frame_id)
        depth_frame = str(depth_info.header.frame_id)
        if self.expected_color_frame and color_frame != self.expected_color_frame:
            self.get_logger().warn(
                f'Hand color frame {color_frame!r} != expected '
                f'{self.expected_color_frame!r}',
                throttle_duration_sec=5.0)
            return None
        if self.expected_depth_frame and depth_frame != self.expected_depth_frame:
            self.get_logger().warn(
                f'Hand depth frame {depth_frame!r} != expected '
                f'{self.expected_depth_frame!r}',
                throttle_duration_sec=5.0)
            return None
        rotation = extrinsics.rotation
        translation = extrinsics.translation_m
        key = (
            int(color_info.width),
            int(color_info.height),
            tuple(np.asarray(color_info.k, dtype=np.float64).ravel()),
            tuple(np.asarray(color_info.d, dtype=np.float64).ravel()),
            str(color_info.header.frame_id),
            int(depth_info.width),
            int(depth_info.height),
            tuple(np.asarray(depth_info.k, dtype=np.float64).ravel()),
            tuple(np.asarray(depth_info.d, dtype=np.float64).ravel()),
            str(depth_info.header.frame_id),
            tuple(rotation.ravel()),
            tuple(translation.ravel()),
            version,
            extrinsics_source,
            self.depth_registration_backend,
            self.depth_registration_allow_sticky_numpy_fallback,
        )
        if self._registrar is not None and self._registrar_key == key:
            return self._registrar
        try:
            registrar = registrar_from_camera_messages(
                color_info,
                depth_info,
                rotation,
                translation,
                f'{version}:{extrinsics_source}',
                backend=self.depth_registration_backend,
                allow_sticky_numpy_fallback=(
                    self.depth_registration_allow_sticky_numpy_fallback),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.get_logger().warn(
                f'Hand depth-to-color registrar skipped: {exc}',
                throttle_duration_sec=5.0)
            return None
        self._registrar = registrar
        self._registrar_key = key
        registration_message = (
            'Hand depth registration initialized '
            f'(requested={registrar.requested_backend}, '
            f'active={registrar.backend_name}, '
            f'version={registrar.backend_version}, '
            f'fallback={registrar.fallback_active})')
        if registrar.fallback_active:
            self.get_logger().warn(
                registration_message
                + f'; error={registrar.last_backend_error}')
        else:
            self.get_logger().info(registration_message)
        return registrar

    def _rgb_sized_depth_map(self, native, height, width, color_info):
        """Map native depth into the RGB frame. Never clip RGB UVs into native HxW."""
        rgb_nan = np.full((height, width), np.nan, dtype=np.float32)
        self._metric_depth_ready = False
        self._aligned_depth_valid_fraction = 0.0
        self._depth_registration_mode = 'unavailable'
        self._last_depth_registration_ms = 0.0
        self._last_depth_registration_gpu_ms = 0.0
        if not self.depth_alignment_validated:
            return rgb_nan
        helpers = self._registration_api()
        metric_depth_in_rgb_frame = None if helpers is None else helpers[1]
        grid_is_aligned = camera_infos_share_pixel_grid(
            color_info,
            self._depth_info,
            native.shape,
            height,
            width,
        )
        registrar = None
        if not grid_is_aligned:
            registrar = self._depth_to_color_registrar(
                color_info, native.shape, height, width)
            if registrar is None:
                self.get_logger().warn(
                    'Hand native depth is not proven to use the RGB pixel '
                    'grid and no valid depth-to-color registrar is ready; '
                    '3D keypoints stay invalid',
                    throttle_duration_sec=5.0)
                return rgb_nan
        registration_started = time.perf_counter()
        try:
            if metric_depth_in_rgb_frame is None:
                if grid_is_aligned:
                    scale = float(self.depth_scale_m_per_unit)
                    depth_m = native.astype(np.float32) * scale
                    depth_m[native == 0] = 0.0
                    aligned = depth_m
                else:
                    return rgb_nan
            else:
                aligned = metric_depth_in_rgb_frame(
                    native,
                    height,
                    width,
                    float(self.depth_scale_m_per_unit),
                    registrar,
                )
        finally:
            self._last_depth_registration_ms = (
                time.perf_counter() - registration_started) * 1000.0
            self._last_depth_registration_gpu_ms = float(
                getattr(registrar, 'last_gpu_ms', 0.0))
        if aligned is None:
            self.get_logger().warn(
                'Hand native depth could not be mapped into the RGB grid; '
                '3D keypoints stay invalid',
                throttle_duration_sec=5.0)
            return rgb_nan
        valid = np.isfinite(aligned) & (aligned > 0.0)
        self._aligned_depth_valid_fraction = float(
            np.count_nonzero(valid) / max(int(aligned.size), 1))
        if np.any(valid):
            self._metric_depth_ready = True
            self._last_metric_depth_at = time.monotonic()
            self._depth_registration_mode = (
                'aligned-color-grid'
                if grid_is_aligned else 'native-depth-to-color'
            )
        return aligned

    def _process(self, color_msg, info_msg, depth_msg):
        # The subscriptions stay alive while INACTIVE so that reactivating is
        # instant; this is where a non-active node throws the frame away
        # before doing any GPU work.
        if not self._active:
            return
        self._last_source_stamp_sec = int(color_msg.header.stamp.sec)
        self._last_source_stamp_nanosec = int(color_msg.header.stamp.nanosec)
        self._last_source_frame_id = str(color_msg.header.frame_id)
        if depth_msg is not None:
            self._last_depth_frame_id = str(depth_msg.header.frame_id)
            color_stamp_ns = (
                int(color_msg.header.stamp.sec) * 1_000_000_000
                + int(color_msg.header.stamp.nanosec))
            depth_stamp_ns = (
                int(depth_msg.header.stamp.sec) * 1_000_000_000
                + int(depth_msg.header.stamp.nanosec))
            self._last_rgb_depth_delta_ns = abs(
                color_stamp_ns - depth_stamp_ns)
        else:
            self._last_rgb_depth_delta_ns = None
        started = time.monotonic()
        self._sync_cached_inference_ms = 0.0
        succeeded = False
        try:
            self._process_inner(color_msg, info_msg, depth_msg)
            succeeded = True
            self._last_error_code = ''
            self._last_error_message = ''
        except Exception:
            self._errors += 1
            import traceback
            self._last_error_code = 'PROCESSING_ERROR'
            self._last_error_message = traceback.format_exc().splitlines()[-1]
            self.get_logger().error('exception in _process:\n' + traceback.format_exc())
        finally:
            self._last_process_ms = (
                (time.monotonic() - started) * 1000.0
                + self._sync_cached_inference_ms
            )
            if succeeded:
                self._last_frame_at = time.monotonic()
                if depth_msg is not None:
                    self._last_real_depth_at = self._last_frame_at
            self._frames += 1

    def _mediapipe_timestamp_ms(self, source_ts_ms):
        """Keep MediaPipe VIDEO timestamps increasing across rosbag --loop."""
        if self._last_source_ts_ms is not None:
            source_delta = source_ts_ms - self._last_source_ts_ms
            if source_delta > 0:
                # Preserve the source timing, with a conservative guard
                # against a corrupt or very large timestamp jump.
                self._mp_frame_interval_ms = min(max(source_delta, 1), 1000)
            else:
                # The recorded Header.stamp rewound. Continue a synthetic
                # MediaPipe-only timeline at the normal frame interval; ROS
                # messages retain their original header stamps unchanged.
                if self.palm_facing_filter is not None:
                    self.palm_facing_filter.reset()
                self._mp_timestamp_offset_ms = (
                    self._last_mp_ts_ms + self._mp_frame_interval_ms - source_ts_ms)
                self.get_logger().info(
                    'source timestamp rewound (rosbag loop); continuing '
                    'MediaPipe VIDEO time monotonically')

        ts_ms = source_ts_ms + self._mp_timestamp_offset_ms
        if self._last_mp_ts_ms is not None and ts_ms <= self._last_mp_ts_ms:
            ts_ms = self._last_mp_ts_ms + max(self._mp_frame_interval_ms, 1)
            self._mp_timestamp_offset_ms = ts_ms - source_ts_ms

        self._last_source_ts_ms = source_ts_ms
        self._last_mp_ts_ms = ts_ms
        return int(ts_ms)

    def _process_inner(self, color_msg, info_msg, depth_msg):
        (frame_bgr, recognition_result, inference_ms, cache_hit,
         delivery_key) = (
            self._recognize_color(color_msg))
        if cache_hit:
            # _process() times the synchronized depth/keypoint part. Include
            # the earlier RGB-only recognizer time so existing latency
            # diagnostics remain comparable to the old single callback.
            self._sync_cached_inference_ms = inference_ms
        H, W = frame_bgr.shape[:2]

        # Normally the direct color callback has already published this exact
        # result. The idempotent fallback preserves gesture output if callback
        # ordering changes or its first publish attempt raised.
        gesture_fallback_started = time.monotonic()
        try:
            gesture_was_published = self._publish_gestures_for_result(
                color_msg, frame_bgr, recognition_result, delivery_key)
        except Exception:
            # A failure in the additive gesture topic must not suppress the
            # established keypoints/overlay/target outputs.
            self._gesture_errors += 1
            import traceback
            self._last_gesture_error_code = 'GESTURE_PUBLISH_ERROR'
            self._last_gesture_error_message = (
                traceback.format_exc().splitlines()[-1])
            self.get_logger().error(
                'gesture fallback publish failed; continuing keypoint path:\n'
                + traceback.format_exc())
            gesture_was_published = False
        if gesture_was_published:
            self._last_gesture_process_ms = (
                inference_ms
                + (time.monotonic() - gesture_fallback_started) * 1000.0
            )

        k = info_msg.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)

        if depth_msg is not None:
            if isinstance(depth_msg, CompressedImage):
                depth_raw = decode_compressed_depth(depth_msg)
            else:
                depth_raw = self.bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding='passthrough')
            if self.depth_alignment_validated:
                depth_map = self._rgb_sized_depth_map(
                    depth_raw, H, W, info_msg)
            else:
                # Keep 2D hand detections alive but invalidate every depth-
                # derived 3D point until alignment is explicitly approved.
                # Use the RGB size so keypoint UVs are never clipped into a
                # smaller native depth image.
                depth_map = np.full((H, W), np.nan, dtype=np.float32)
        elif self.use_mono_depth:
            depth_map = run_mono_depth(frame_bgr, self.torch, self.depth_processor,
                                        self.depth_model, self.device, self.dtype, H, W)
        else:
            depth_map = np.full((H, W), np.nan, dtype=np.float32)

        target_px = frame_corner_label = None
        if self.robot_position:
            target_px, frame_corner_label = robot_position_target_px(self.robot_position, W, H)

        stamp = color_msg.header.stamp
        source_ts_ms = stamp.sec * 1000 + stamp.nanosec // 1_000_000

        frame_depth_label = (
            self.depth_source_label
            if depth_msg is not None or self.use_mono_depth
            else 'RGB ONLY (REAL DEPTH UNAVAILABLE)'
        )
        if (
            self.palm_facing_estimator is not None
            and not self.palm_facing_mapping_verified
        ):
            frame_depth_label += ' | PALM MAP PROVISIONAL'

        row_hands, overlay, _ = process_frame(
            frame_bgr, depth_map, self.hand_det, self.mp, K, fx, fy, cx, cy, W, H,
            source_ts_ms,
            region=self.region, target_px=target_px, robot_position_label=self.robot_position,
            frame_corner_label=frame_corner_label, flip_handedness=self.flip_handedness,
            forced_handedness_label=self.forced_handedness_label,
            draw_overlay=self.publish_overlay,
            depth_source_label=frame_depth_label,
            # Preserve MediaPipe's 2-D result while live metric registration
            # is unavailable; depth-derived fields remain invalid.
            allow_2d_only=(
                self.rgb_only
                or (self.use_real_depth and depth_msg is None)
                or (depth_msg is not None and not self._metric_depth_ready)
            ),
            recognition_result=recognition_result,
            palm_facing_estimator=(
                None if self.palm_facing_estimator is None
                else self.palm_facing_estimator.estimate),
            palm_facing_filter=(
                None if self.palm_facing_filter is None
                else self.palm_facing_filter.update),
            gesture_profile=self.gesture_profile,
        )

        self._hands_last = len(row_hands)
        self._palm_facing_valid_last = sum(
            1 for hand in row_hands
            if bool((hand.get('palm_facing') or {}).get('has_facing', False))
        )
        rejections = {}
        for hand in row_hands:
            facing = hand.get('palm_facing') or {}
            reason = str(facing.get('rejection_reason', ''))
            if not reason:
                continue
            rejections[reason] = rejections.get(reason, 0) + 1
            if reason.startswith((
                'estimator_exception:', 'temporal_filter_exception:',
            )):
                self._palm_facing_errors += 1
        self._palm_facing_rejections_last = rejections

        kp_msg = HandKeypoints()
        kp_msg.header = color_msg.header
        kp_msg.depth_source = (
            'real' if depth_msg is not None
            else ('mono' if self.use_mono_depth else 'rgb_only')
        )
        kp_msg.hands = [_row_hand_to_msg(h) for h in row_hands]
        self.pub_keypoints.publish(kp_msg)

        if self.palm_facing_estimator is not None:
            palm_meta = estimator_metadata(
                self.palm_facing_estimator, self.palm_facing_filter)
            facing_msg = HandFacingArray()
            facing_msg.header = color_msg.header
            facing_msg.estimator_name = str(
                palm_meta.get('name', PALM_FACING_ESTIMATOR_NAME))
            facing_msg.estimator_version = str(
                palm_meta.get('version', PALM_FACING_ESTIMATOR_VERSION))
            facing_msg.estimator_spec_sha256 = str(
                palm_meta.get('spec_sha256', ''))
            facing_msg.calibration_version = str(
                palm_meta.get('calibration_version', ''))
            facing_msg.handedness_mapping_version = str(
                palm_meta.get('handedness_mapping_version', ''))
            facing_msg.supported_facings = [
                'PALM_UP', 'PALM_DOWN', 'EDGE']
            facing_msg.rejection_category = 'UNKNOWN'
            facing_msg.hands = [
                _row_facing_to_msg(hand) for hand in row_hands]
            self.pub_facing.publish(facing_msg)
        self.get_logger().info(
            f'published hand keypoints: {len(row_hands)} hands',
            throttle_duration_sec=1.0)

        if self.publish_overlay:
            overlay_msg = self.bridge.cv2_to_compressed_imgmsg(overlay, dst_format='jpg')
            overlay_msg.header = color_msg.header
            self.pub_overlay.publish(overlay_msg)

        if self.publish_target_pose:
            self._publish_target_pose(color_msg.header, row_hands)

    def _publish_target_pose(self, header, row_hands):
        """Publish the palm pose the robot should hand a tool to.

        With robot_position set, process_frame() has already reduced
        row_hands to the single handoff candidate. Without it, fall back to
        the first hand that has a valid palm_6d — the task coordinator
        activates this node only when it wants exactly one handover target,
        so an arbitrary-but-deterministic choice is better than publishing
        nothing at all.
        """
        for hand in row_hands:
            palm = hand['palm_6d']
            if palm is None:
                continue
            pose = PoseStamped()
            pose.header = header
            tx, ty, tz = palm['translation']
            qw, qx, qy, qz = palm['rotation_quat_wxyz']
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tx, ty, tz
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            self.pub_target_pose.publish(pose)
            return


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HandDetectionNode()

        if node.get_parameter('autostart').value:
            # Standalone / demo use: behave like the old non-lifecycle node.
            # Under surgical_task_coordinator this is set false so the
            # coordinator decides when this algorithm gets its turn.
            # Inside the try because loading MediaPipe + Depth-Anything V2
            # takes seconds, and a Ctrl-C during it would otherwise escape
            # main() as a raw KeyboardInterrupt.
            node.get_logger().info('autostart:=true -- self-configuring and activating')
            if node.trigger_configure() != TransitionCallbackReturn.SUCCESS:
                raise RuntimeError('Hand lifecycle configure failed')
            if node.trigger_activate() != TransitionCallbackReturn.SUCCESS:
                raise RuntimeError('Hand lifecycle activate failed')

        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Jazzy's default signal handler already shuts the context down on
        # SIGINT, so spin() raises ExternalShutdownException rather than
        # KeyboardInterrupt. try_shutdown() (not shutdown()) is what makes
        # the finally block idempotent -- calling shutdown() on an
        # already-shut-down context raises RCLError and exits non-zero,
        # which ros2 launch then reports as "process has died".
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                # Ctrl-C signals the whole foreground process group AND ros2
                # launch forwards its own SIGINT, so a second one routinely
                # lands mid-teardown. Nothing left worth saving at that point.
                pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
