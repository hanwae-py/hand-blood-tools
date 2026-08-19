"""Replays a recorded RGB(+depth) clip as ROS2 camera topics, so
hand_detection_node can be tested end-to-end without real camera
hardware. This is NOT a camera driver — it's a test fixture, matching
the default topic names hand_detection_node subscribes to out of the box.

Parameters:
  rgb_path       (required) path to an RGB video file
  calib_path     (required) this lab's calibration JSON
                 ({"camera_info": {"/synced/<cam_key>/color/camera_info": {"k": [...]}}})
  cam_key        which camera in calib_path (default 'cam_4')
  depth_h5_path  optional real-depth HDF5 (dataset "depth", uint16 mm).
                 Omit to publish RGB only — hand_detection_node will then
                 exercise its monocular-depth fallback, same as running
                 run_hand_keypoints.py without a depth file.
  color_topic, depth_topic, camera_info_topic  topic name overrides
  rate_hz        publish rate (default 15.0, matching this lab's source clips)
  loop           loop back to frame 0 at the end (default true)
  preload_depth  preload the complete HDF5 depth array for fast replay (default
                 true). Set false to read one depth frame per timer tick.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class FakeCameraPublisher(Node):
    def __init__(self):
        super().__init__('fake_camera_publisher')

        self.declare_parameter('rgb_path', '')
        self.declare_parameter('calib_path', '')
        self.declare_parameter('cam_key', 'cam_4')
        self.declare_parameter('depth_h5_path', '')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('rate_hz', 15.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('preload_depth', True)
        # Use RELIABLE only when a test subscriber requires it. The default
        # BEST_EFFORT sensor QoS avoids replay-rate backpressure.
        self.declare_parameter('reliable_output', False)

        g = self.get_parameter
        self.rgb_path = g('rgb_path').value
        self.calib_path = g('calib_path').value
        self.cam_key = g('cam_key').value
        self.depth_h5_path = g('depth_h5_path').value
        self.rate_hz = g('rate_hz').value
        self.loop = g('loop').value
        self.preload_depth = g('preload_depth').value
        self.reliable_output = g('reliable_output').value

        if not self.rgb_path or not self.calib_path:
            raise RuntimeError('fake_camera_publisher requires rgb_path and calib_path parameters')

        with open(self.calib_path) as f:
            cfg = json.load(f)
        k = cfg['camera_info'][f'/synced/{self.cam_key}/color/camera_info']['k']
        self.fx, self.fy, self.cx, self.cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])

        self.cap = cv2.VideoCapture(self.rgb_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'could not open rgb_path={self.rgb_path}')
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create publishers BEFORE the (potentially slow, see below) depth
        # preload, gated on the depth_h5_path parameter rather than on
        # depth_ds. hand_detection_node's depth_source=auto only checks
        # whether a publisher exists on the depth topic -- it does NOT
        # wait for actual data. Creating pub_depth only after preloading
        # meant the topic wasn't advertised in time for that ~3s check,
        # so auto-detection consistently (mis)concluded no real depth was
        # available and fell back to mono depth. Publisher creation here
        # is cheap/instant; __init__ still can't return (so the publish
        # timer can't fire) until the preload below finishes, so this is
        # safe -- it only fixes the discovery-timing race, not data flow.
        self.bridge = CvBridge()
        # Camera streams normally use sensor-data QoS. Some test consumers
        # (notably the RF-DETR node) request RELIABLE input, so allow that
        # compatible publisher profile explicitly without changing the default.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20 if self.reliable_output else 5,
            reliability=(ReliabilityPolicy.RELIABLE if self.reliable_output
                         else ReliabilityPolicy.BEST_EFFORT),
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_color = self.create_publisher(
            Image, g('color_topic').value, sensor_qos)
        self.pub_info = self.create_publisher(
            CameraInfo, g('camera_info_topic').value, sensor_qos)
        self.pub_depth = (self.create_publisher(
                           Image, g('depth_topic').value, sensor_qos)
                           if self.depth_h5_path else None)

        self.depth_h5 = None
        self.depth_ds = None
        self.depth_executor = None
        self.depth_block = None
        self.depth_block_start = 0
        self.next_depth_future = None
        self.next_depth_start = None
        if self.depth_h5_path:
            self.depth_h5 = h5py.File(self.depth_h5_path, 'r')
            h5_depth_ds = self.depth_h5['depth']
            if self.preload_depth:
                # Fast recorded-data replay. Lazy h5py indexing was previously
                # measured to fall far behind the requested camera rate.
                _t0 = time.monotonic()
                self.depth_ds = h5_depth_ds[:]
                self.get_logger().info(
                    f'publishing real depth from {self.depth_h5_path} '
                    f'(preloaded {self.depth_ds.nbytes / 1e9:.2f} GB to RAM in '
                    f'{time.monotonic() - _t0:.1f}s)')
            else:
                # Low-memory test mode. The dataset is gzip-compressed in
                # multi-frame chunks, so indexing one frame at a time repeatedly
                # decompresses the same chunk. Keep the current and next aligned
                # chunk in a double buffer; publication is still one frame per
                # tick, while HDF5 decompression happens ahead of consumption.
                self.depth_ds = h5_depth_ds
                self.depth_block_size = (
                    int(self.depth_ds.chunks[0]) if self.depth_ds.chunks else 32
                )
                self.depth_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix='depth_h5_prefetch')
                self._reset_depth_stream()
                buffered_frames = min(
                    self.depth_block_size * 2, self.depth_ds.shape[0])
                buffered_bytes = int(
                    buffered_frames * np.prod(self.depth_ds.shape[1:])
                    * self.depth_ds.dtype.itemsize)
                self.get_logger().info(
                    f'streaming real depth frame-by-frame from {self.depth_h5_path} '
                    f'(background chunk prefetch, ~{buffered_bytes / 1e6:.1f} MB buffer; '
                    f'no full preload)')
        else:
            self.get_logger().info('no depth_h5_path given -- publishing RGB only '
                                    '(hand_detection_node will use its mono-depth fallback)')

        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1e-3), self._tick)
        self.get_logger().info(f'fake_camera_publisher: {self.rgb_path}  '
                                f'{self.W}x{self.H}, {self.n_total} frames, {self.rate_hz} Hz'
                                + (' (looping)' if self.loop else ''))

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self._rewind():
                return
            ok, frame = self.cap.read()
            if not ok:
                return

        if self.depth_ds is not None and self.frame_idx >= self.depth_ds.shape[0]:
            if not self._rewind():
                return
            ok, frame = self.cap.read()
            if not ok:
                return

        now = self.get_clock().now().to_msg()

        color_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        color_msg.header.stamp = now
        color_msg.header.frame_id = self.cam_key
        self.pub_color.publish(color_msg)

        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = self.cam_key
        info_msg.width = self.W
        info_msg.height = self.H
        info_msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        self.pub_info.publish(info_msg)

        if self.depth_ds is not None:
            if self.preload_depth:
                depth_frame = np.ascontiguousarray(self.depth_ds[self.frame_idx])
            else:
                depth_frame = self._get_streamed_depth_frame(self.frame_idx)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding='16UC1')
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = self.cam_key
            self.pub_depth.publish(depth_msg)

        self.frame_idx += 1

    def _read_depth_block(self, start):
        end = min(start + self.depth_block_size, self.depth_ds.shape[0])
        return np.ascontiguousarray(self.depth_ds[start:end])

    def _schedule_next_depth_block(self):
        start = self.depth_block_start + len(self.depth_block)
        if start >= self.depth_ds.shape[0]:
            self.next_depth_start = None
            self.next_depth_future = None
            return
        self.next_depth_start = start
        self.next_depth_future = self.depth_executor.submit(
            self._read_depth_block, start)

    def _reset_depth_stream(self):
        if self.next_depth_future is not None:
            self.next_depth_future.cancel()
        self.depth_block_start = 0
        self.depth_block = self._read_depth_block(0)
        self._schedule_next_depth_block()

    def _get_streamed_depth_frame(self, frame_idx):
        block_end = self.depth_block_start + len(self.depth_block)
        if not self.depth_block_start <= frame_idx < block_end:
            if self.next_depth_start == frame_idx and self.next_depth_future is not None:
                self.depth_block = self.next_depth_future.result()
                self.depth_block_start = self.next_depth_start
            else:
                self.depth_block_start = (
                    frame_idx // self.depth_block_size * self.depth_block_size)
                self.depth_block = self._read_depth_block(self.depth_block_start)
            self._schedule_next_depth_block()
        return np.ascontiguousarray(self.depth_block[frame_idx - self.depth_block_start])

    def _rewind(self):
        """Loop back to frame 0, or stop the timer at end of clip. Returns
        True if playback should continue (either looped, or wasn't at end).

        Re-opens the capture from scratch rather than
        cap.set(CAP_PROP_POS_FRAMES, 0). A multi-second stall was
        observed around a loop boundary during testing; benchmarking
        cap.set() in isolation (including from near end-of-clip) showed
        it completing in ~2ms, so it probably wasn't the actual cause --
        more likely the already-documented lazy per-tick HDF5 read
        (see the "Known limitation" note in the README). Kept this
        reopen anyway since it's no slower and is a simpler-to-reason
        about starting state than a long-lived capture handle."""
        if not self.loop:
            self.get_logger().info('source clip finished, stopping publisher')
            self.timer.cancel()
            return False
        self.cap.release()
        self.cap = cv2.VideoCapture(self.rgb_path)
        self.frame_idx = 0
        if self.depth_ds is not None and not self.preload_depth:
            self._reset_depth_stream()
        return True

    def destroy_node(self):
        self.cap.release()
        if self.depth_executor is not None:
            self.depth_executor.shutdown(wait=True, cancel_futures=True)
        if self.depth_h5 is not None:
            self.depth_h5.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # Construction is inside the try because it is SLOW -- preloading a
        # multi-GB depth HDF5 takes seconds, and a Ctrl-C landing in there
        # would otherwise escape main() as a raw KeyboardInterrupt.
        node = FakeCameraPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # See the matching note in hand_detection_node.main(): on SIGINT the
        # context is already down, so shutdown() would raise and exit 1.
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                # See the matching note in hand_detection_node.main().
                pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
