#!/usr/bin/env python3
"""Record EIR suction RGB-D as raw Images into an MCAP bag plus metadata.json.

The live EIR suction camera publishes JPEG and compressedDepth only.  This
process decodes those payloads locally, downsamples to the suction camera
size (848x480) at 4 FPS, and writes ``sensor_msgs/Image`` plus scaled
CameraInfo.  Recording starts immediately and stops on Ctrl+C.
It does not republish the raw images onto the ROS domain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from pnu_surgical_tool import decode_compressed_depth_16uc1


SCHEMA = "pnu.surgical.suction_raw_mcap.v1"
CAMERA = "suction"
SOURCE_PREFIX = "/eir/camera/suction"
COLOR_ENCODING = "bgr8"
DEPTH_ENCODING = "16UC1"
DEPTH_SCALE_M_PER_UNIT = 0.001
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 4.0
MCAP_STEM = "bloodcam"

SOURCE_TOPICS = {
    "color": f"{SOURCE_PREFIX}/color/image_raw/compressed",
    "depth": f"{SOURCE_PREFIX}/aligned_depth_to_color/image_raw/compressedDepth",
    "color_info": f"{SOURCE_PREFIX}/color/camera_info",
    "depth_info": f"{SOURCE_PREFIX}/aligned_depth_to_color/camera_info",
}
RECORDED_TOPICS = {
    "color": f"{SOURCE_PREFIX}/color/image_raw",
    "depth": f"{SOURCE_PREFIX}/aligned_depth_to_color/image_raw",
    "color_info": SOURCE_TOPICS["color_info"],
    "depth_info": SOURCE_TOPICS["depth_info"],
}
RECORDED_TYPES = {
    "color": "sensor_msgs/msg/Image",
    "depth": "sensor_msgs/msg/Image",
    "color_info": "sensor_msgs/msg/CameraInfo",
    "depth_info": "sensor_msgs/msg/CameraInfo",
}


def source_reader_qos() -> QoSProfile:
    """Match the EIR BEST_EFFORT writers without building a local backlog."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def decode_color_jpeg(payload: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode a JPEG payload to a contiguous BGR uint8 image."""
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the suction JPEG")
    return np.ascontiguousarray(image)


def resize_color(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample BGR to the suction recording size."""
    if width <= 0 or height <= 0:
        return image
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.ascontiguousarray(
        cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    )


def resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample metric depth without interpolating millimetre values."""
    if width <= 0 or height <= 0:
        return depth
    if depth.shape[1] == width and depth.shape[0] == height:
        return depth
    return np.ascontiguousarray(
        cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    )


def scale_camera_info(info: CameraInfo, width: int, height: int) -> CameraInfo:
    """Scale CameraInfo intrinsics to the recorded image size."""
    source_w = int(info.width)
    source_h = int(info.height)
    if width <= 0 or height <= 0 or (source_w == width and source_h == height):
        return info
    if source_w <= 0 or source_h <= 0:
        raise ValueError("CameraInfo width and height must be positive before scaling")
    sx = width / source_w
    sy = height / source_h
    scaled = CameraInfo()
    scaled.header = info.header
    scaled.width = width
    scaled.height = height
    scaled.distortion_model = info.distortion_model
    scaled.d = list(info.d)
    k = list(info.k)
    k[0] *= sx
    k[2] *= sx
    k[4] *= sy
    k[5] *= sy
    scaled.k = k
    scaled.r = list(info.r)
    p = list(info.p)
    p[0] *= sx
    p[2] *= sx
    p[3] *= sx
    p[5] *= sy
    p[6] *= sy
    p[7] *= sy
    scaled.p = p
    scaled.binning_x = info.binning_x
    scaled.binning_y = info.binning_y
    scaled.roi.x_offset = int(round(info.roi.x_offset * sx))
    scaled.roi.y_offset = int(round(info.roi.y_offset * sy))
    scaled.roi.width = int(round(info.roi.width * sx)) if info.roi.width else 0
    scaled.roi.height = int(round(info.roi.height * sy)) if info.roi.height else 0
    scaled.roi.do_rectify = info.roi.do_rectify
    return scaled


def keep_frame(
    first_ns: int | None, last_slot: int | None, stamp: int, fps: float
) -> tuple[bool, int]:
    """Keep at most one source frame in each 1/fps slot after the first stamp."""
    if fps <= 0:
        return True, 0
    if first_ns is None or stamp < first_ns:
        return True, 0
    slot = int((stamp - first_ns) * fps / 1_000_000_000)
    if last_slot is None or slot > last_slot:
        return True, slot
    return False, last_slot if last_slot is not None else 0


def numpy_to_image(header: Any, array: np.ndarray, encoding: str) -> Image:
    """Pack a NumPy image into a ``sensor_msgs/Image`` with the source header."""
    message = Image()
    message.header = header
    message.height = int(array.shape[0])
    message.width = int(array.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    if encoding == COLOR_ENCODING:
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f"color image must be uint8 HxWx3, got {array.dtype} {array.shape}")
        message.step = message.width * 3
    elif encoding == DEPTH_ENCODING:
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(f"depth image must be uint16 HxW, got {array.dtype} {array.shape}")
        message.step = message.width * 2
    else:
        raise ValueError(f"unsupported encoding: {encoding}")
    message.data = array.tobytes()
    return message


def stamp_ns(header: Any) -> int:
    """Return the header stamp in nanoseconds, or now when the stamp is zero."""
    value = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
    return value if value > 0 else time.time_ns()


def isoformat_utc(when: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""
    moment = when or datetime.now(timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_metadata(
    *,
    output_dir: Path,
    started_at: str,
    ended_at: str | None = None,
    message_counts: dict[str, int] | None = None,
    color_size: list[int] | None = None,
    depth_size: list[int] | None = None,
    color_frame_id: str = "",
    depth_frame_id: str = "",
    ros_domain_id: int = 0,
    fps: float = DEFAULT_FPS,
    record_width: int = DEFAULT_WIDTH,
    record_height: int = DEFAULT_HEIGHT,
    source_color_size: list[int] | None = None,
    source_depth_size: list[int] | None = None,
) -> dict[str, Any]:
    """Return the sidecar recording contract written next to the MCAP."""
    return {
        "schema": SCHEMA,
        "camera": CAMERA,
        "source": SOURCE_PREFIX,
        "payload": "raw",
        "color_encoding": COLOR_ENCODING,
        "depth_encoding": DEPTH_ENCODING,
        "depth_scale_m_per_unit": DEPTH_SCALE_M_PER_UNIT,
        "fps": fps,
        "record_width": record_width,
        "record_height": record_height,
        "source_was_compressed": True,
        "source_color_size": source_color_size,
        "source_depth_size": source_depth_size,
        "topics": dict(RECORDED_TOPICS),
        "source_topics": dict(SOURCE_TOPICS),
        "topic_types": dict(RECORDED_TYPES),
        "color_size": color_size,
        "depth_size": depth_size,
        "color_frame_id": color_frame_id,
        "depth_frame_id": depth_frame_id,
        "ros_domain_id": ros_domain_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "message_counts": message_counts or {name: 0 for name in RECORDED_TOPICS},
        "output_dir": str(output_dir),
        "mcap_stem": MCAP_STEM,
    }


def write_metadata_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically write the sidecar metadata.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def recording_folder_name(when: datetime) -> str:
    """Return MMDDHHMM for the minute recording actually starts."""
    return when.strftime("%m%d%H%M")


def default_output_dir(when: datetime | None = None) -> Path:
    """Return ``$HOME/Videos/recordings/<MMDDHHMM>`` for the start minute."""
    stamp = recording_folder_name(when or datetime.now())
    return Path.home() / "Videos" / "recordings" / stamp


def rename_bag_files(output_dir: Path, stem: str = MCAP_STEM) -> list[Path]:
    """Rename rosbag2 MCAP files so the stem is ``bloodcam``."""
    renamed: list[Path] = []
    yaml_path = output_dir / "metadata.yaml"
    yaml_text = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    for mcap in sorted(output_dir.glob("*.mcap")):
        suffix = mcap.suffix
        index = ""
        if "_" in mcap.stem and mcap.stem.rsplit("_", 1)[-1].isdigit():
            index = f"_{mcap.stem.rsplit('_', 1)[-1]}"
        dest = output_dir / f"{stem}{index}{suffix}"
        if dest == mcap:
            renamed.append(dest)
            continue
        if dest.exists():
            raise FileExistsError(f"refusing to overwrite {dest}")
        mcap.rename(dest)
        yaml_text = yaml_text.replace(mcap.name, dest.name)
        renamed.append(dest)
    if yaml_path.exists():
        yaml_path.write_text(yaml_text, encoding="utf-8")
    return renamed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Bag directory to create. Default: $HOME/Videos/recordings/<MMDDHHMM>",
    )
    parser.add_argument(
        "--first-frame-timeout-sec",
        type=float,
        default=8.0,
        help="Fail if no decoded color frame arrives within this time.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Recorded image width. 0 keeps the EIR source width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Recorded image height. 0 keeps the EIR source height.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Maximum recorded frame rate. 0 keeps every source frame.",
    )
    return parser.parse_args(argv)


class SuctionRawRecorder(Node):
    """Decode live EIR suction compressed streams and write raw Images to MCAP."""

    def __init__(
        self,
        output_dir: Path,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: float = DEFAULT_FPS,
    ) -> None:
        super().__init__("suction_raw_mcap_recorder")
        self._output_dir = output_dir
        self._width = width
        self._height = height
        self._fps = fps
        self._started_at = isoformat_utc()
        self._counts = {name: 0 for name in RECORDED_TOPICS}
        self._color_size: list[int] | None = None
        self._depth_size: list[int] | None = None
        self._source_color_size: list[int] | None = None
        self._source_depth_size: list[int] | None = None
        self._color_frame_id = ""
        self._depth_frame_id = ""
        self._first_color_at: float | None = None
        self._color_first_ns: int | None = None
        self._depth_first_ns: int | None = None
        self._color_slot: int | None = None
        self._depth_slot: int | None = None
        self._latest_color_info: CameraInfo | None = None
        self._latest_depth_info: CameraInfo | None = None
        self._closed = False
        self._writer = SequentialWriter()
        self._writer.open(
            StorageOptions(uri=str(output_dir), storage_id="mcap"),
            ConverterOptions("", ""),
        )
        for index, key in enumerate(RECORDED_TOPICS, start=1):
            self._writer.create_topic(
                TopicMetadata(
                    id=index,
                    name=RECORDED_TOPICS[key],
                    type=RECORDED_TYPES[key],
                    serialization_format="cdr",
                )
            )
        write_metadata_json(output_dir / "metadata.json", self._metadata())
        qos = source_reader_qos()
        self.create_subscription(
            CompressedImage, SOURCE_TOPICS["color"], self._on_color, qos)
        self.create_subscription(
            CompressedImage, SOURCE_TOPICS["depth"], self._on_depth, qos)
        self.create_subscription(
            CameraInfo, SOURCE_TOPICS["color_info"], self._on_color_info, qos)
        self.create_subscription(
            CameraInfo, SOURCE_TOPICS["depth_info"], self._on_depth_info, qos)
        self.get_logger().info(
            f"recording raw suction RGB-D to {output_dir} at "
            f"{width}x{height} {fps:g}fps from compressed EIR topics"
        )

    def _metadata(self, ended_at: str | None = None) -> dict[str, Any]:
        return build_metadata(
            output_dir=self._output_dir,
            started_at=self._started_at,
            ended_at=ended_at,
            message_counts=dict(self._counts),
            color_size=self._color_size,
            depth_size=self._depth_size,
            color_frame_id=self._color_frame_id,
            depth_frame_id=self._depth_frame_id,
            ros_domain_id=int(os.environ.get("ROS_DOMAIN_ID", "0")),
            fps=self._fps,
            record_width=self._width,
            record_height=self._height,
            source_color_size=self._source_color_size,
            source_depth_size=self._source_depth_size,
        )

    def _write(self, key: str, message: Any, header: Any) -> None:
        self._writer.write(
            RECORDED_TOPICS[key],
            serialize_message(message),
            stamp_ns(header),
        )
        self._counts[key] += 1
        if self._counts[key] in {1, 50} or self._counts[key] % 150 == 0:
            self.get_logger().info(
                f"wrote {key} frames={self._counts[key]} "
                f"color={self._counts['color']} depth={self._counts['depth']}"
            )

    def _on_color(self, message: CompressedImage) -> None:
        image = decode_color_jpeg(message.data)
        if self._source_color_size is None:
            self._source_color_size = [int(image.shape[1]), int(image.shape[0])]
        stamp = stamp_ns(message.header)
        keep, slot = keep_frame(
            self._color_first_ns, self._color_slot, stamp, self._fps)
        if not keep:
            return
        image = resize_color(image, self._width, self._height)
        packed = numpy_to_image(message.header, image, COLOR_ENCODING)
        if self._color_size is None:
            self._color_size = [int(image.shape[1]), int(image.shape[0])]
            self._color_frame_id = str(message.header.frame_id)
            self._first_color_at = time.monotonic()
            self._color_first_ns = stamp
        self._write("color", packed, message.header)
        self._color_slot = slot
        if self._latest_color_info is not None:
            info = scale_camera_info(
                self._latest_color_info, packed.width, packed.height)
            info.header = message.header
            self._write("color_info", info, message.header)

    def _on_depth(self, message: CompressedImage) -> None:
        depth = decode_compressed_depth_16uc1(message.data, message.format)
        if self._source_depth_size is None:
            self._source_depth_size = [int(depth.shape[1]), int(depth.shape[0])]
        stamp = stamp_ns(message.header)
        keep, slot = keep_frame(
            self._depth_first_ns, self._depth_slot, stamp, self._fps)
        if not keep:
            return
        depth = resize_depth(depth, self._width, self._height)
        packed = numpy_to_image(message.header, depth, DEPTH_ENCODING)
        if self._depth_size is None:
            self._depth_size = [int(depth.shape[1]), int(depth.shape[0])]
            self._depth_frame_id = str(message.header.frame_id)
            self._depth_first_ns = stamp
        self._write("depth", packed, message.header)
        self._depth_slot = slot
        if self._latest_depth_info is not None:
            info = scale_camera_info(
                self._latest_depth_info, packed.width, packed.height)
            info.header = message.header
            self._write("depth_info", info, message.header)

    def _on_color_info(self, message: CameraInfo) -> None:
        self._latest_color_info = message
        if self._color_size is not None and self._counts["color_info"] == 0:
            info = scale_camera_info(message, self._color_size[0], self._color_size[1])
            self._write("color_info", info, message.header)

    def _on_depth_info(self, message: CameraInfo) -> None:
        self._latest_depth_info = message
        if self._depth_size is not None and self._counts["depth_info"] == 0:
            info = scale_camera_info(message, self._depth_size[0], self._depth_size[1])
            self._write("depth_info", info, message.header)

    def received_color(self) -> bool:
        return self._first_color_at is not None

    def close(self) -> Path:
        if self._closed:
            return self._output_dir / "metadata.json"
        self._closed = True
        metadata_path = self._output_dir / "metadata.json"
        write_metadata_json(metadata_path, self._metadata(ended_at=isoformat_utc()))
        self._writer.close()
        rename_bag_files(self._output_dir)
        self.get_logger().info(
            f"closed bag color={self._counts['color']} depth={self._counts['depth']} "
            f"metadata={metadata_path}"
        )
        return metadata_path


def record(args: argparse.Namespace) -> Path:
    stop = False

    def request_stop(_signum: int | None = None, _frame: object | None = None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    output_dir = args.output_dir or default_output_dir(datetime.now())
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = SuctionRawRecorder(
        output_dir, width=args.width, height=args.height, fps=args.fps)
    first_deadline = time.monotonic() + args.first_frame_timeout_sec
    try:
        while rclpy.ok() and not stop:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except (KeyboardInterrupt, ExternalShutdownException):
                break
            now = time.monotonic()
            if not node.received_color() and now > first_deadline:
                raise TimeoutError(
                    "no suction color frame from "
                    f"{SOURCE_TOPICS['color']} within {args.first_frame_timeout_sec}s"
                )
    finally:
        metadata_path = node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata_path = record(args)
    except (FileExistsError, TimeoutError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"wrote {metadata_path}")
    print(f"bag {metadata_path.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
