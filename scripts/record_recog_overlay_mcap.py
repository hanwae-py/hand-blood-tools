#!/usr/bin/env python3
"""Record the published CAM3, CAM4, and right-EE recognition overlays.

Source is the compositor's distributed overlay, not camera raw ingress.
The three ``/perception/*/overlay/compressed`` JPEGs already contain
detector drawings.  They are resized to 640x480 at 10 FPS.  Recording
starts immediately and stops on Ctrl+C.
"""

from __future__ import annotations

import argparse
from datetime import datetime
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
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from sensor_msgs.msg import CompressedImage

from record_suction_raw_mcap import (
    decode_color_jpeg,
    default_output_dir,
    isoformat_utc,
    keep_frame,
    rename_bag_files,
    resize_color,
    source_reader_qos,
    stamp_ns,
    write_metadata_json,
)


SCHEMA = "pnu.surgical.recog_overlay_mcap.v1"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 10.0
MCAP_STEM = "recogcam"
JPEG_QUALITY = 85

OVERLAY_TOPICS = {
    "cam_3": "/perception/cam_3/overlay/compressed",
    "cam_4": "/perception/cam_4/overlay/compressed",
    "right_ee": "/perception/right_ee/overlay/compressed",
}


def encode_overlay_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a BGR overlay frame as JPEG."""
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise ValueError("OpenCV could not encode the overlay JPEG")
    return encoded.tobytes()


def numpy_to_compressed(header: Any, image: np.ndarray) -> CompressedImage:
    """Pack a resized overlay as the same CompressedImage type the stack publishes."""
    message = CompressedImage()
    message.header = header
    message.format = "jpeg"
    message.data = encode_overlay_jpeg(image)
    return message


def build_metadata(
    *,
    output_dir: Path,
    started_at: str,
    ended_at: str | None = None,
    message_counts: dict[str, int] | None = None,
    source_sizes: dict[str, list[int]] | None = None,
    recorded_sizes: dict[str, list[int]] | None = None,
    fps: float = DEFAULT_FPS,
    record_width: int = DEFAULT_WIDTH,
    record_height: int = DEFAULT_HEIGHT,
) -> dict[str, Any]:
    """Return the sidecar recording contract written next to the MCAP."""
    return {
        "schema": SCHEMA,
        "cameras": list(OVERLAY_TOPICS),
        "source": "published_overlay",
        "payload": "overlay_jpeg",
        "fps": fps,
        "record_width": record_width,
        "record_height": record_height,
        "topics": dict(OVERLAY_TOPICS),
        "topic_types": {
            name: "sensor_msgs/msg/CompressedImage" for name in OVERLAY_TOPICS
        },
        "source_sizes": source_sizes or {},
        "recorded_sizes": recorded_sizes or {},
        "started_at": started_at,
        "ended_at": ended_at,
        "message_counts": message_counts or {name: 0 for name in OVERLAY_TOPICS},
        "output_dir": str(output_dir),
        "mcap_stem": MCAP_STEM,
    }


def folder_has_bag(folder: Path) -> bool:
    """Return whether ``folder`` already holds a rosbag2 recording."""
    return folder.exists() and (
        (folder / "metadata.yaml").exists() or any(folder.glob("*.mcap"))
    )


def resolve_output_dir(
    when: datetime | None = None, *, base: Path | None = None
) -> Path:
    """Use the same date folder as bloodcam, or a recogcam subfolder if occupied."""
    folder = base if base is not None else default_output_dir(when)
    if folder_has_bag(folder):
        return folder / "recogcam"
    return folder


class RecogOverlayRecorder(Node):
    """Record the three published recognition-overlay JPEGs."""

    def __init__(
        self,
        output_dir: Path,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: float = DEFAULT_FPS,
    ) -> None:
        super().__init__("recog_overlay_mcap_recorder")
        self._output_dir = output_dir
        self._width = width
        self._height = height
        self._fps = fps
        self._started_at = isoformat_utc()
        self._counts = {name: 0 for name in OVERLAY_TOPICS}
        self._source_sizes: dict[str, list[int]] = {}
        self._recorded_sizes: dict[str, list[int]] = {}
        self._first_ns = {name: None for name in OVERLAY_TOPICS}
        self._slots = {name: None for name in OVERLAY_TOPICS}
        self._first_frame_at: float | None = None
        self._closed = False
        self._writer = SequentialWriter()
        self._writer.open(
            StorageOptions(uri=str(output_dir), storage_id="mcap"),
            ConverterOptions("", ""),
        )
        for index, name in enumerate(OVERLAY_TOPICS, start=1):
            self._writer.create_topic(
                TopicMetadata(
                    id=index,
                    name=OVERLAY_TOPICS[name],
                    type="sensor_msgs/msg/CompressedImage",
                    serialization_format="cdr",
                )
            )
        write_metadata_json(output_dir / "metadata.json", self._metadata())
        qos = source_reader_qos()
        for name, topic in OVERLAY_TOPICS.items():
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, camera=name: self._on_overlay(camera, message),
                qos,
            )
        self.get_logger().info(
            f"recording recognition overlays to {output_dir} at "
            f"{width}x{height} {fps:g}fps: {', '.join(OVERLAY_TOPICS.values())}"
        )

    def _metadata(self, ended_at: str | None = None) -> dict[str, Any]:
        return build_metadata(
            output_dir=self._output_dir,
            started_at=self._started_at,
            ended_at=ended_at,
            message_counts=dict(self._counts),
            source_sizes=dict(self._source_sizes),
            recorded_sizes=dict(self._recorded_sizes),
            fps=self._fps,
            record_width=self._width,
            record_height=self._height,
        )

    def _on_overlay(self, camera: str, message: CompressedImage) -> None:
        image = decode_color_jpeg(message.data)
        if camera not in self._source_sizes:
            self._source_sizes[camera] = [int(image.shape[1]), int(image.shape[0])]
        stamp = stamp_ns(message.header)
        keep, slot = keep_frame(
            self._first_ns[camera], self._slots[camera], stamp, self._fps)
        if not keep:
            return
        image = resize_color(image, self._width, self._height)
        packed = numpy_to_compressed(message.header, image)
        if camera not in self._recorded_sizes:
            self._recorded_sizes[camera] = [int(image.shape[1]), int(image.shape[0])]
            self._first_ns[camera] = stamp
            if self._first_frame_at is None:
                self._first_frame_at = time.monotonic()
        self._writer.write(OVERLAY_TOPICS[camera], serialize_message(packed), stamp)
        self._counts[camera] += 1
        self._slots[camera] = slot
        if self._counts[camera] in {1, 50} or self._counts[camera] % 150 == 0:
            self.get_logger().info(
                "wrote "
                + " ".join(f"{name}={self._counts[name]}" for name in OVERLAY_TOPICS)
            )

    def received_overlay(self) -> bool:
        return self._first_frame_at is not None

    def close(self) -> Path:
        if self._closed:
            return self._output_dir / "metadata.json"
        self._closed = True
        metadata_path = self._output_dir / "metadata.json"
        write_metadata_json(metadata_path, self._metadata(ended_at=isoformat_utc()))
        self._writer.close()
        rename_bag_files(self._output_dir, stem=MCAP_STEM)
        self.get_logger().info(
            "closed bag "
            + " ".join(f"{name}={self._counts[name]}" for name in OVERLAY_TOPICS)
            + f" metadata={metadata_path}"
        )
        return metadata_path


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
        help="Fail if no overlay frame arrives within this time.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    return parser.parse_args(argv)


def record(args: argparse.Namespace) -> Path:
    stop = False

    def request_stop(_signum: int | None = None, _frame: object | None = None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    output_dir = args.output_dir or resolve_output_dir(datetime.now())
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = RecogOverlayRecorder(
        output_dir, width=args.width, height=args.height, fps=args.fps)
    first_deadline = time.monotonic() + args.first_frame_timeout_sec
    try:
        while rclpy.ok() and not stop:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except (KeyboardInterrupt, ExternalShutdownException):
                break
            if not node.received_overlay() and time.monotonic() > first_deadline:
                raise TimeoutError(
                    "no overlay frame from "
                    + ", ".join(OVERLAY_TOPICS.values())
                    + f" within {args.first_frame_timeout_sec}s"
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
