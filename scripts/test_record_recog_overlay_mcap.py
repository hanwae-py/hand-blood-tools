#!/usr/bin/env python3
"""Unit tests for the recognition-overlay MCAP recorder."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

from record_recog_overlay_mcap import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MCAP_STEM,
    OVERLAY_TOPICS,
    SCHEMA,
    build_metadata,
    encode_overlay_jpeg,
    numpy_to_compressed,
    resolve_output_dir,
)
from record_suction_raw_mcap import (
    decode_color_jpeg,
    keep_frame,
    rename_bag_files,
    resize_color,
    write_metadata_json,
)


class RecordRecogOverlayMcapTest(unittest.TestCase):
    def test_topics_are_the_three_published_overlays(self) -> None:
        self.assertEqual(
            OVERLAY_TOPICS,
            {
                "cam_3": "/perception/cam_3/overlay/compressed",
                "cam_4": "/perception/cam_4/overlay/compressed",
                "right_ee": "/perception/right_ee/overlay/compressed",
            },
        )
        document = build_metadata(
            output_dir=Path("/tmp/recog_test"),
            started_at="2026-08-31T00:00:00.000Z",
        )
        self.assertEqual(document["schema"], SCHEMA)
        self.assertEqual(document["mcap_stem"], "recogcam")
        self.assertEqual(document["fps"], DEFAULT_FPS)
        self.assertEqual(document["record_width"], DEFAULT_WIDTH)
        self.assertEqual(document["record_height"], DEFAULT_HEIGHT)
        self.assertEqual(document["payload"], "overlay_jpeg")
        self.assertEqual(document["source"], "published_overlay")

    def test_overlay_is_resized_to_640x480_jpeg(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[10, 20] = (0, 0, 255)
        resized = resize_color(frame, 640, 480)
        self.assertEqual(resized.shape, (480, 640, 3))
        header = Header()
        header.stamp.sec = 1
        packed = numpy_to_compressed(header, resized)
        self.assertEqual(packed.format, "jpeg")
        decoded = decode_color_jpeg(packed.data)
        self.assertEqual(decoded.shape, (480, 640, 3))
        self.assertGreater(len(encode_overlay_jpeg(resized)), 0)

    def test_keep_frame_throttles_to_10fps(self) -> None:
        keep, slot = keep_frame(None, None, 0, 10.0)
        self.assertTrue(keep)
        keep, slot = keep_frame(0, 0, 50_000_000, 10.0)
        self.assertFalse(keep)
        keep, slot = keep_frame(0, 0, 100_000_000, 10.0)
        self.assertTrue(keep)
        self.assertEqual(slot, 1)

    def test_occupied_date_folder_uses_recogcam_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            occupied = Path(tmp) / "08311101"
            occupied.mkdir()
            (occupied / "bloodcam_0.mcap").write_bytes(b"x")
            (occupied / "metadata.yaml").write_text("files: [bloodcam_0.mcap]\n")
            self.assertEqual(
                resolve_output_dir(base=occupied), occupied / "recogcam")
            empty = Path(tmp) / "08311102"
            self.assertEqual(resolve_output_dir(base=empty), empty)

    def test_writer_renames_mcap_to_recogcam(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "08311101"
            writer = SequentialWriter()
            writer.open(
                StorageOptions(uri=str(output), storage_id="mcap"),
                ConverterOptions("", ""),
            )
            writer.create_topic(
                TopicMetadata(
                    id=1,
                    name=OVERLAY_TOPICS["cam_3"],
                    type="sensor_msgs/msg/CompressedImage",
                    serialization_format="cdr",
                )
            )
            packed = numpy_to_compressed(
                Header(), np.zeros((480, 640, 3), dtype=np.uint8))
            writer.write(OVERLAY_TOPICS["cam_3"], serialize_message(packed), 1)
            writer.close()
            write_metadata_json(
                output / "metadata.json",
                build_metadata(
                    output_dir=output,
                    started_at="2026-08-31T00:00:00.000Z",
                ),
            )
            rename_bag_files(output, stem=MCAP_STEM)
            self.assertTrue((output / "recogcam_0.mcap").exists())
            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=str(output), storage_id="mcap"),
                ConverterOptions("", ""),
            )
            topic, payload, _stamp = reader.read_next()
            self.assertEqual(topic, OVERLAY_TOPICS["cam_3"])
            restored = deserialize_message(payload, CompressedImage)
            self.assertEqual(restored.format, "jpeg")


if __name__ == "__main__":
    unittest.main()
