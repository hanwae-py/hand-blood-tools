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
  /surgery/perception/cam4/hand_keypoints (hand_keypoint_interfaces/HandKeypoints)
    — this frame's detected hand(s), typed: per hand, hand_index,
    handedness, joints_3d (metres, camera optical frame), joints_2d
    (pixels), kp_valid_depth, palm_6d. Carries both the 2D and 3D result
    together (see hand_keypoint_interfaces/msg/HandKeypoints.msg) — no
    separate 2D->3D conversion node is needed downstream.
  /surgery/images/cam4/hand_overlay/compressed (sensor_msgs/CompressedImage)
    — annotated debug visualization (skeleton + palm gizmo drawn on the
    input frame). CompressedImage, not raw Image, to match every other
    image topic in the contract. Disable with publish_overlay:=false.
  /surgery/perception/cam4/hand_target_pose (geometry_msgs/PoseStamped)
    — the palm 6D pose the robot should hand a tool to, ready for a
    downstream TF/robot node to consume without JSON parsing. When
    robot_position is set this is that single handoff hand; otherwise it
    is the first detected hand that has a valid palm_6d.
  /surgery/perception/handkeypoint/health (std_msgs/String, JSON, 1 Hz)
  /surgery/perception/handkeypoint/diagnostics/json (std_msgs/String, 1 Hz)
    — the health/diagnostics pair every perception node in the contract
    publishes.

Depth input is accepted either as a raw ``sensor_msgs/Image`` or directly as
the synchronized camera provider's ``compressedDepth``
``sensor_msgs/CompressedImage``.  The latter is decoded in this node; no
republisher is required.  Decoding a 16UC1 PNG does not prove color alignment,
so metric 3D output remains invalid until ``depth_alignment_validated`` is
explicitly enabled after the RGB/depth frame and calibration contract is
approved.
"""
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
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from cv_bridge import CvBridge

from hand_keypoint_interfaces.msg import HandKeypoints, Hand, PalmPose6D, Point2D

from hand_keypoint_ros.core import (
    DEFAULT_DEPTH_MODEL,
    robot_position_target_px, load_mediapipe, load_mono_depth_model,
    run_mono_depth, process_frame,
)

_AUTO_DEPTH_DETECT_TIMEOUT_S = 3.0
_AUTO_DEPTH_DETECT_POLL_S = 0.2
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


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


class HandDetectionNode(LifecycleNode):
    def __init__(self):
        super().__init__('hand_detection_node')

        # ---- parameters -----------------------------------------------
        self.declare_parameter(
            'color_topic', '/synced/cam_4/color/image_raw/compressed')
        self.declare_parameter('color_transport', 'compressed')  # compressed | raw
        self.declare_parameter(
            'depth_topic',
            '/synced/cam_4/depth/image_rect_raw/compressedDepth')
        self.declare_parameter(
            'depth_transport', 'compressed_depth')  # compressed_depth | raw
        self.declare_parameter('depth_alignment_validated', False)
        self.declare_parameter(
            'camera_info_topic', '/synced/cam_4/color/camera_info')

        # Output topic names follow the ARPA-H interface contract
        # (/surgery/perception/..., /surgery/images/.../compressed) rather
        # than the old hand/* names, so downstream consumers can subscribe
        # from the contract table without a per-lab exception.
        self.declare_parameter('keypoints_topic', '/surgery/perception/cam4/hand_keypoints')
        self.declare_parameter('overlay_topic', '/surgery/images/cam4/hand_overlay/compressed')
        self.declare_parameter('target_pose_topic', '/surgery/perception/cam4/hand_target_pose')
        self.declare_parameter('health_topic', '/surgery/perception/handkeypoint/health')
        self.declare_parameter('diagnostics_topic',
                                '/surgery/perception/handkeypoint/diagnostics/json')

        self.declare_parameter('depth_source', 'auto')  # auto | real | mono
        self.declare_parameter('depth_model', DEFAULT_DEPTH_MODEL)
        self.declare_parameter('max_hands', 4)
        self.declare_parameter('cpu_only', False)
        self.declare_parameter('flip_handedness', False)
        self.declare_parameter('region_x_min', 0.0)
        self.declare_parameter('region_x_max', 1.0)
        self.declare_parameter('region_y_min', 0.0)
        self.declare_parameter('region_y_max', 1.0)
        self.declare_parameter('robot_position', '')  # '' = disabled (default: keep every hand)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('sync_slop_sec', 0.05)
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
        self._subs = []
        self._active = False

        self.use_real_depth = False
        self.depth_source_label = 'unconfigured'

        # ---- diagnostics counters ---------------------------------------
        self._frames = 0
        self._hands_last = 0
        self._last_process_ms = 0.0
        self._last_frame_at = None
        self._errors = 0
        self._rate_sample_at = time.monotonic()
        self._rate_sample_frames = 0
        self._processed_hz = 0.0
        self.input_stale_timeout_sec = float(
            self.get_parameter('input_stale_timeout_sec').value)
        self.depth_transport = str(
            self.get_parameter('depth_transport').value).strip().lower()
        self.depth_alignment_validated = bool(
            self.get_parameter('depth_alignment_validated').value)
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
            f'keypoints={g("keypoints_topic").value}')

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
        self.camera_info_topic = g('camera_info_topic').value
        depth_source_param = g('depth_source').value
        self.depth_model_name = g('depth_model').value
        self.max_hands = g('max_hands').value
        self.cpu_only = g('cpu_only').value
        self.flip_handedness = g('flip_handedness').value
        self.region = (g('region_x_min').value, g('region_x_max').value,
                       g('region_y_min').value, g('region_y_max').value)
        self.robot_position = g('robot_position').value or None
        self.publish_overlay = g('publish_overlay').value
        self.jpeg_quality = int(g('overlay_jpeg_quality').value)
        self.input_stale_timeout_sec = float(g('input_stale_timeout_sec').value)
        if self.input_stale_timeout_sec <= 0.0:
            self.get_logger().error('input_stale_timeout_sec must be > 0')
            return TransitionCallbackReturn.FAILURE
        sync_slop = g('sync_slop_sec').value

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
        elif depth_source_param == 'mono':
            self.use_real_depth = False
        else:
            self.use_real_depth = self._wait_for_publisher(self.depth_topic)
            if not self.use_real_depth:
                self.get_logger().warn(
                    f'depth_source=auto: no publisher seen for "{self.depth_topic}" within '
                    f'{_AUTO_DEPTH_DETECT_TIMEOUT_S}s -- falling back to monocular depth '
                    f'(Depth-Anything V2). If the depth publisher just starts later than this '
                    f'node, reconfigure this node, or set depth_source:=real explicitly.')
        if self.use_real_depth and self.depth_alignment_validated:
            self.depth_source_label = 'REAL DEPTH (ALIGNMENT VALIDATED)'
        elif self.use_real_depth:
            self.depth_source_label = 'REAL DEPTH (ALIGNMENT PENDING)'
        else:
            self.depth_source_label = 'MONO DEPTH (Depth-Anything V2)'
        self.get_logger().info(f'depth source: {self.depth_source_label}')

        # ---- models (this is the part that costs VRAM) --------------------
        try:
            self.mp, self.hand_det = load_mediapipe(self.max_hands, cpu_only=self.cpu_only)
            if not self.use_real_depth:
                (self.torch, self.depth_processor, self.depth_model,
                 self.device, self.dtype) = load_mono_depth_model(
                     self.depth_model_name, cpu_only=self.cpu_only)
        except Exception:
            import traceback
            self.get_logger().error('model loading failed:\n' + traceback.format_exc())
            self._release_models()
            return TransitionCallbackReturn.FAILURE

        # ---- subscribers (time-synchronized) ---------------------------
        rgb_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        info_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        depth_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        color_type = CompressedImage if self.color_transport == 'compressed' else Image
        color_sub = message_filters.Subscriber(
            self, color_type, self.color_topic, qos_profile=rgb_qos)
        info_sub = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=info_qos)
        self._subs = [color_sub, info_sub]
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
                [color_sub, depth_sub, info_sub], queue_size=10, slop=sync_slop)
            self.sync.registerCallback(self._on_synced_real)
        else:
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [color_sub, info_sub], queue_size=10, slop=sync_slop)
            self.sync.registerCallback(self._on_synced_mono)

        self.get_logger().info(
            f'configured. color={self.color_topic} ({self.color_transport}) '
            + (
                f'depth={self.depth_topic} ({self.depth_transport}, '
                f'alignment_validated={self.depth_alignment_validated}) '
                if self.use_real_depth else '(monocular depth) '
            )
            + f'camera_info={self.camera_info_topic}'
            + (f' | robot_position={self.robot_position}' if self.robot_position else ''))
        self._last_error_code = ''
        self._last_error_message = ''
        # A newly configured MediaPipe instance starts its own VIDEO timeline.
        self._last_mp_ts_ms = None
        self._last_source_ts_ms = None
        self._mp_timestamp_offset_ms = 0
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Our turn: start processing the frames we are already subscribed to."""
        self._active = True
        self._frames = 0
        self._rate_sample_at = time.monotonic()
        self._rate_sample_frames = 0
        self._processed_hz = 0.0
        self.get_logger().info('ACTIVE: processing camera frames')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Turn is over. Frames are ignored, but the models stay in VRAM so
        the next activation is instant."""
        self._active = False
        self.get_logger().info(
            f'INACTIVE: stopped processing after {self._frames} frames '
            f'(models still loaded)')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Release the models so their VRAM goes back to the pool."""
        self._active = False
        self._teardown_subscriptions()
        self._release_models()
        self.depth_source_label = 'unconfigured'
        self._last_error_code = 'MODEL_NOT_CONFIGURED'
        self._last_error_message = 'hand detector was cleaned up'
        self.get_logger().info('cleaned up: models released, VRAM freed')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._active = False
        self._teardown_subscriptions()
        self._release_models()
        return TransitionCallbackReturn.SUCCESS

    def _teardown_subscriptions(self):
        self.sync = None
        for sub in self._subs:
            # message_filters.Subscriber wraps the real rclpy subscription.
            inner = getattr(sub, 'sub', None)
            if inner is not None:
                try:
                    self.destroy_subscription(inner)
                except Exception:                          # noqa: BLE001
                    self.get_logger().debug('subscription teardown raised; ignoring')
        self._subs = []

    def _release_models(self):
        """Drop every reference that could be pinning GPU memory."""
        if self.hand_det is not None:
            try:
                self.hand_det.close()
            except Exception:                              # noqa: BLE001
                self.get_logger().debug('MediaPipe close() raised; ignoring')
        self.mp = self.hand_det = None

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
        now = time.monotonic()
        elapsed = now - self._rate_sample_at
        if elapsed > 0.0:
            self._processed_hz = (
                (self._frames - self._rate_sample_frames) / elapsed)
            self._rate_sample_at = now
            self._rate_sample_frames = self._frames
        stale_s = (None if self._last_frame_at is None
                   else round(now - self._last_frame_at, 2))
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
        if self.use_real_depth:
            depth_stream_ready = bool(
                self._depth_publisher_seen
                and self._last_real_depth_at is not None
                and now - self._last_real_depth_at <= self.input_stale_timeout_sec
            )
            depth_ready = bool(
                depth_stream_ready and self.depth_alignment_validated)
        else:
            # Monocular fallback: depth is inferred by Depth-Anything V2 from
            # the RGB frame we just processed, so there is no second stream
            # whose freshness could be checked. Demanding one reported every
            # mono-depth run as degraded even while keypoints were flowing.
            depth_stream_ready = self.depth_model is not None and rgb_ready
            depth_ready = depth_stream_ready
        model_ready = self.hand_det is not None
        inference_ready = bool(self._active and rgb_ready and depth_ready and model_ready)
        status = 'ok' if inference_ready else 'degraded'
        error_code = self._last_error_code
        error_message = self._last_error_message
        if not error_code:
            if not model_ready:
                error_code = 'MODEL_NOT_CONFIGURED'
                error_message = 'hand detector model is not configured'
            elif not rgb_ready:
                error_code = 'CAM4_RGB_STALE_OR_MISSING'
                error_message = 'fresh CAM4 RGB has not been processed'
            elif not depth_ready:
                if self.use_real_depth and not self._depth_publisher_seen:
                    # Distinguishable from "the stream stalled": the aligned
                    # -depth provider does not exist on this LAN yet.
                    error_code = 'CAM4_ALIGNED_DEPTH_NOT_PROVIDED'
                    error_message = f'no publisher on {depth_topic}'
                elif self.use_real_depth and depth_stream_ready and not self.depth_alignment_validated:
                    error_code = 'CAM4_DEPTH_ALIGNMENT_NOT_VALIDATED'
                    error_message = (
                        'compressed depth is received but RGB/depth alignment '
                        'and calibration are not approved')
                else:
                    error_code = 'CAM4_DEPTH_STALE_OR_MISSING'
                    error_message = 'real or monocular depth is not ready'
        self.pub_health.publish(String(data=json.dumps({
            'schema': 'pnu.hand_keypoint_health.v1',
            'node': self.get_name(),
            'lifecycle_state': state,
            'status': status,
            'ready': inference_ready,
            'rgb_ready': rgb_ready,
            'depth_stream_ready': depth_stream_ready,
            'depth_ready': depth_ready,
            'depth_alignment_validated': self.depth_alignment_validated,
            'model_ready': model_ready,
            'hand_inference_ready': inference_ready,
            'depth_source': self.depth_source_label,
            'seconds_since_last_frame': stale_s,
            'input_stale_timeout_sec': self.input_stale_timeout_sec,
            'last_error_code': error_code,
            'last_error_message': error_message,
        })))
        self.pub_diagnostics.publish(String(data=json.dumps({
            'node': self.get_name(),
            'lifecycle_state': state,
            'frames_processed': self._frames,
            'hands_last_frame': self._hands_last,
            'last_process_ms': round(self._last_process_ms, 1),
            'processed_hz_1s': round(self._processed_hz, 2),
            'errors': self._errors,
            'depth_source': self.depth_source_label,
            'color_topic': self.get_parameter('color_topic').value,
            'color_transport': self.get_parameter('color_transport').value,
            'camera_info_topic': self.get_parameter('camera_info_topic').value,
            'depth_topic': depth_topic,
            'depth_transport': self.get_parameter('depth_transport').value,
            'depth_alignment_validated': self.depth_alignment_validated,
            'depth_frame_id': self._last_depth_frame_id,
            'max_hands': self.get_parameter('max_hands').value,
            'cpu_only': self.get_parameter('cpu_only').value,
            'source_stamp_sec': self._last_source_stamp_sec,
            'source_stamp_nanosec': self._last_source_stamp_nanosec,
            'frame_id': self._last_source_frame_id,
            'error_code': error_code,
            'error_message': error_message,
        })))

    # ---------------------------------------------------------------------

    def _on_synced_mono(self, color_msg, info_msg):
        self._process(color_msg, info_msg, depth_msg=None)

    def _on_synced_real(self, color_msg, depth_msg, info_msg):
        self._process(color_msg, info_msg, depth_msg=depth_msg)

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
        started = time.monotonic()
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
            self._last_process_ms = (time.monotonic() - started) * 1000.0
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
        if isinstance(color_msg, CompressedImage):
            frame_bgr = self.bridge.compressed_imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8')
        else:
            frame_bgr = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8')
        H, W = frame_bgr.shape[:2]

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
                depth_map = depth_raw.astype(np.float32) / 1000.0
            else:
                # Keep 2D hand detections alive but invalidate every depth-
                # derived 3D point until alignment is explicitly approved.
                depth_map = np.full(depth_raw.shape, np.nan, dtype=np.float32)
        else:
            depth_map = run_mono_depth(frame_bgr, self.torch, self.depth_processor,
                                        self.depth_model, self.device, self.dtype, H, W)

        target_px = frame_corner_label = None
        if self.robot_position:
            target_px, frame_corner_label = robot_position_target_px(self.robot_position, W, H)

        stamp = color_msg.header.stamp
        source_ts_ms = stamp.sec * 1000 + stamp.nanosec // 1_000_000
        ts_ms = self._mediapipe_timestamp_ms(source_ts_ms)

        row_hands, overlay, _ = process_frame(
            frame_bgr, depth_map, self.hand_det, self.mp, K, fx, fy, cx, cy, W, H, ts_ms,
            region=self.region, target_px=target_px, robot_position_label=self.robot_position,
            frame_corner_label=frame_corner_label, flip_handedness=self.flip_handedness,
            draw_overlay=self.publish_overlay, depth_source_label=self.depth_source_label,
            # Publish MediaPipe's 2-D result during a live alignment check,
            # while kp_valid_depth and palm_6d remain invalid.
            allow_2d_only=(depth_msg is not None and not self.depth_alignment_validated))

        self._hands_last = len(row_hands)

        kp_msg = HandKeypoints()
        kp_msg.header = color_msg.header
        kp_msg.depth_source = 'real' if depth_msg is not None else 'mono'
        kp_msg.hands = [_row_hand_to_msg(h) for h in row_hands]
        self.pub_keypoints.publish(kp_msg)
        self.get_logger().info(f'published hand keypoints: {len(row_hands)} hands',
                                throttle_duration_sec=1.0)

        if self.publish_overlay:
            overlay_msg = self.bridge.cv2_to_compressed_imgmsg(overlay, dst_format='jpg')
            overlay_msg.header = color_msg.header
            self.pub_overlay.publish(overlay_msg)

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
            node.trigger_configure()
            node.trigger_activate()

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
