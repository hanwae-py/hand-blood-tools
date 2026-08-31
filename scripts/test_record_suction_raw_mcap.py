#!/usr/bin/env python3
"""Unit tests for the suction raw-MCAP recording helpers."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from record_suction_raw_mcap import (
    COLOR_ENCODING,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEPTH_ENCODING,
    RECORDED_TOPICS,
    SCHEMA,
    SOURCE_TOPICS,
    build_metadata,
    decode_color_jpeg,
    default_output_dir,
    recording_folder_name,
    keep_frame,
    numpy_to_image,
    rename_bag_files,
    resize_color,
    resize_depth,
    scale_camera_info,
    write_metadata_json,
)
from pnu_surgical_tool import decode_compressed_depth_16uc1


class RecordSuctionRawMcapTest(unittest.TestCase):
    def test_metadata_names_raw_recorded_topics_and_compressed_sources(self) -> None:
        document = build_metadata(
            output_dir=Path("/tmp/suction_raw_test"),
            started_at="2026-08-31T00:00:00.000Z",
            color_size=[640, 480],
            depth_size=[640, 480],
        )
        self.assertEqual(document["schema"], SCHEMA)
        self.assertEqual(document["payload"], "raw")
        self.assertEqual(document["color_encoding"], COLOR_ENCODING)
        self.assertEqual(document["depth_encoding"], DEPTH_ENCODING)
        self.assertTrue(document["source_was_compressed"])
        self.assertEqual(
            document["topics"]["color"],
            "/eir/camera/suction/color/image_raw",
        )
        self.assertEqual(
            document["source_topics"]["color"],
            "/eir/camera/suction/color/image_raw/compressed",
        )
        self.assertEqual(document["topics"], RECORDED_TOPICS)
        self.assertEqual(document["source_topics"], SOURCE_TOPICS)
        self.assertEqual(document["fps"], DEFAULT_FPS)
        self.assertEqual(document["record_width"], DEFAULT_WIDTH)
        self.assertEqual(document["record_height"], DEFAULT_HEIGHT)
        self.assertEqual(document["mcap_stem"], "bloodcam")

    def test_resize_and_camera_info_follow_848x480(self) -> None:
        color = np.zeros((720, 1280, 3), dtype=np.uint8)
        color[10, 20] = (1, 2, 3)
        resized = resize_color(color, 848, 480)
        self.assertEqual(resized.shape, (480, 848, 3))
        depth = np.full((720, 1280), 1500, dtype=np.uint16)
        depth[0, 0] = 42
        self.assertEqual(resize_depth(depth, 848, 480)[0, 0], 42)
        info = CameraInfo()
        info.width = 1280
        info.height = 720
        info.k = [640.0, 0.0, 640.0, 0.0, 360.0, 360.0, 0.0, 0.0, 1.0]
        info.p = [640.0, 0.0, 640.0, 0.0, 0.0, 360.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        scaled = scale_camera_info(info, 848, 480)
        self.assertEqual(scaled.width, 848)
        self.assertEqual(scaled.height, 480)
        self.assertAlmostEqual(scaled.k[0], 640.0 * 848 / 1280)
        self.assertAlmostEqual(scaled.k[5], 360.0 * 480 / 720)

    def test_keep_frame_throttles_to_4fps(self) -> None:
        keep, slot = keep_frame(None, None, 0, 4.0)
        self.assertTrue(keep)
        self.assertEqual(slot, 0)
        keep, slot = keep_frame(0, 0, 100_000_000, 4.0)
        self.assertFalse(keep)
        keep, slot = keep_frame(0, 0, 250_000_000, 4.0)
        self.assertTrue(keep)
        self.assertEqual(slot, 1)
        keep, slot = keep_frame(0, 1, 500_000_000, 4.0)
        self.assertTrue(keep)
        self.assertEqual(slot, 2)

    def test_default_output_uses_videos_recordings_mmddhhmm(self) -> None:
        when = datetime(2026, 8, 31, 10, 22, 37)
        self.assertEqual(recording_folder_name(when), "08311022")
        self.assertEqual(
            default_output_dir(when),
            Path.home() / "Videos" / "recordings" / "08311022",
        )

    def test_color_and_depth_round_trip_into_raw_image_messages(self) -> None:
        color = np.zeros((8, 12, 3), dtype=np.uint8)
        color[2, 3] = (10, 20, 30)
        ok, jpeg = cv2.imencode(".jpg", color)
        self.assertTrue(ok)
        decoded = decode_color_jpeg(jpeg.tobytes())
        self.assertEqual(decoded.shape, (8, 12, 3))
        self.assertEqual(decoded.dtype, np.uint8)

        depth = np.full((8, 12), 1500, dtype=np.uint16)
        depth[0, 0] = 0
        ok, png = cv2.imencode(".png", depth)
        self.assertTrue(ok)
        native = decode_compressed_depth_16uc1(
            b"header12" + png.tobytes(), "16UC1; compressedDepth png"
        )
        header = Header()
        header.stamp.sec = 12
        header.stamp.nanosec = 34
        header.frame_id = "suction_color_optical_frame"
        color_msg = numpy_to_image(header, decoded, COLOR_ENCODING)
        depth_msg = numpy_to_image(header, native, DEPTH_ENCODING)
        self.assertEqual(color_msg.encoding, "bgr8")
        self.assertEqual(depth_msg.encoding, "16UC1")
        self.assertEqual(color_msg.height, 8)
        self.assertEqual(depth_msg.width, 12)
        self.assertEqual(int.from_bytes(depth_msg.data[0:2], "little"), 0)

    def test_writer_emits_mcap_and_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suction_raw_unit"
            document = build_metadata(
                output_dir=output,
                started_at="2026-08-31T00:00:00.000Z",
                ended_at="2026-08-31T00:00:01.000Z",
                message_counts={"color": 1, "depth": 1, "color_info": 0, "depth_info": 0},
            )
            writer = SequentialWriter()
            writer.open(
                StorageOptions(uri=str(output), storage_id="mcap"),
                ConverterOptions("", ""),
            )
            write_metadata_json(output / "metadata.json", document)
            writer.create_topic(
                TopicMetadata(
                    id=1,
                    name=RECORDED_TOPICS["color"],
                    type="sensor_msgs/msg/Image",
                    serialization_format="cdr",
                )
            )
            image = numpy_to_image(Header(), np.zeros((4, 6, 3), dtype=np.uint8), COLOR_ENCODING)
            writer.write(RECORDED_TOPICS["color"], serialize_message(image), 1)
            writer.close()
            rename_bag_files(output)

            sidecar = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["schema"], SCHEMA)
            self.assertTrue((output / "metadata.yaml").exists())
            self.assertTrue((output / "bloodcam_0.mcap").exists())
            self.assertFalse(list(output.glob("suction_raw_unit_*.mcap")))

            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=str(output), storage_id="mcap"),
                ConverterOptions("", ""),
            )
            topic, payload, _stamp = reader.read_next()
            self.assertEqual(topic, RECORDED_TOPICS["color"])
            restored = deserialize_message(payload, Image)
            self.assertEqual(restored.encoding, "bgr8")
            self.assertEqual(restored.width, 6)


if __name__ == "__main__":
    unittest.main()
