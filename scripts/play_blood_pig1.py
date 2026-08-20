#!/usr/bin/env python3
"""Loop PNG/JPEG frames from a folder as a ROS compressed-image topic."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class ImageFolderPublisher(Node):
    def __init__(self, images_dir: Path, topic: str, fps: float) -> None:
        super().__init__("blood_pig1_image_replay")
        self._paths = sorted(
            path for path in images_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self._paths:
            raise RuntimeError(f"no PNG/JPEG files in {images_dir}")
        self._index = 0
        self._publisher = self.create_publisher(
            CompressedImage, topic, qos_profile_sensor_data
        )
        self.create_timer(1.0 / fps, self._publish_next)
        self.get_logger().info(
            f"replaying {len(self._paths)} images from {images_dir} at {fps:g} FPS on {topic}"
        )

    def _publish_next(self) -> None:
        path = self._paths[self._index]
        self._index = (self._index + 1) % len(self._paths)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error(f"cannot read {path}")
            return
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            self.get_logger().error(f"cannot JPEG-encode {path}")
            return
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "blood_pig1_color"
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self._publisher.publish(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--topic", default="/surgery/test/blood_pig1/color/image_raw/compressed"
    )
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")

    rclpy.init()
    node = ImageFolderPublisher(args.images_dir, args.topic, args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
