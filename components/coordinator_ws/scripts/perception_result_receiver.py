#!/usr/bin/env python3
"""Confirm that a downstream ROS node receives tool and hand results."""

from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from hand_keypoint_interfaces.msg import HandKeypoints
from surgical_perception_msgs.msg import ToolObservation2DArray, ToolPoseArray


class PerceptionResultReceiver(Node):
    def __init__(self) -> None:
        super().__init__("perception_result_receiver")
        self.declare_parameter(
            "tool_topic", "/surgery/perception/cam4/observations"
        )
        self.declare_parameter(
            "hand_topic", "/surgery/perception/cam4/hand_keypoints"
        )
        self.declare_parameter("timeout_sec", 300.0)

        self.got_tool = False
        self.got_tool_pose = False
        self.got_hand = False
        self.started = time.monotonic()
        self.create_subscription(
            ToolObservation2DArray,
            self.get_parameter("tool_topic").value,
            self._on_tool,
            10,
        )
        self.create_subscription(
            ToolPoseArray,
            "/surgery/perception/cam4/tool_poses",
            self._on_tool_pose,
            10,
        )
        self.create_subscription(
            HandKeypoints,
            self.get_parameter("hand_topic").value,
            self._on_hand,
            10,
        )
        self.create_timer(1.0, self._check_timeout)
        self.get_logger().info(
            "waiting for one real tool result and one real hand result"
        )

    def _on_tool(self, msg: ToolObservation2DArray) -> None:
        if self.got_tool:
            return
        if not msg.instances:
            self.get_logger().info(
                "received typed tool message with 0 instances; waiting for a real detection"
            )
            return
        classes = sorted({str(instance.class_name) for instance in msg.instances})
        self.get_logger().info(
            f"RECEIVED TOOL RESULT v1.3: {len(msg.instances)} instances; "
            f"classes={classes}; observation_id={msg.observation_id}"
        )
        self.got_tool = True
        self._finish_if_complete()

    def _on_tool_pose(self, msg: ToolPoseArray) -> None:
        if self.got_tool_pose:
            return
        valid = sum(bool(item.position_valid and item.orientation_valid) for item in msg.tools)
        self.get_logger().info(
            f"RECEIVED TOOL POSE ARRAY v1.3: {len(msg.tools)} tools; "
            f"valid_poses={valid}; observation_id={msg.observation_id}"
        )
        self.got_tool_pose = True
        self._finish_if_complete()
    def _on_hand(self, msg: HandKeypoints) -> None:
        if self.got_hand:
            return
        valid_3d = sum(sum(bool(value) for value in hand.kp_valid_depth) for hand in msg.hands)
        palm_poses = sum(bool(hand.has_palm_6d) for hand in msg.hands)
        if not msg.hands or valid_3d == 0 or palm_poses == 0:
            self.get_logger().info(
                f"received hand message but no usable 3D target yet: "
                f"hands={len(msg.hands)}, valid_3d_keypoints={valid_3d}, "
                f"palm_poses={palm_poses}; continuing"
            )
            return
        self.get_logger().info(
            f"RECEIVED HAND RESULT: {len(msg.hands)} hands; "
            f"valid_3d_keypoints={valid_3d}; palm_poses={palm_poses}; "
            f"depth_source={msg.depth_source}"
        )
        self.got_hand = True
        self._finish_if_complete()

    def _finish_if_complete(self) -> None:
        if self.got_tool and self.got_tool_pose and self.got_hand:
            self.get_logger().info(
                "SUCCESS: downstream node received both real detector outputs"
            )
            rclpy.shutdown()

    def _check_timeout(self) -> None:
        if time.monotonic() - self.started < self.get_parameter("timeout_sec").value:
            return
        missing = []
        if not self.got_tool:
            missing.append("tool")
        if not self.got_tool_pose:
            missing.append("tool_pose")
        if not self.got_hand:
            missing.append("hand")
        self.get_logger().error(f"timeout waiting for: {', '.join(missing)}")
        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = PerceptionResultReceiver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
