"""Signal-driven take-turn controller for tool, hand and blood perception."""

from __future__ import annotations

import json
import threading

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from surgical_task_coordinator.lifecycle_detector import (
    LifecycleDetector,
    LifecycleDetectorError,
)


COMMANDS = {
    "TOOL": "tool", "RUN_TOOL": "tool", "DETECT_TOOL": "tool",
    "HAND": "hand", "RUN_HAND": "hand", "DETECT_HAND": "hand",
    "BLOOD": "blood", "RUN_BLOOD": "blood", "DETECT_BLOOD": "blood",
    "STOP": "idle", "IDLE": "idle", "ABORT": "idle",
}


class PerceptionModeCoordinator(Node):
    def __init__(self) -> None:
        super().__init__("perception_mode_coordinator")
        self.declare_parameter("command_topic", "/surgery/perception/mode_command")
        self.declare_parameter("state_topic", "/surgery/perception/mode_state")
        self.declare_parameter("tool_detector_node", "tool_detection_node")
        self.declare_parameter("hand_detector_node", "hand_detection_node")
        self.declare_parameter("blood_detector_node", "blood_detection_node")
        self.declare_parameter("release_gpu_between_modes", False)
        self.declare_parameter("preload_models_on_startup", True)
        self.declare_parameter("model_load_timeout_sec", 180.0)

        g = self.get_parameter
        self.release_gpu = bool(g("release_gpu_between_modes").value)
        self.preload_models = bool(g("preload_models_on_startup").value)
        self.load_timeout = float(g("model_load_timeout_sec").value)
        self.detectors = {
            "tool": LifecycleDetector(self, g("tool_detector_node").value),
            "hand": LifecycleDetector(self, g("hand_detector_node").value),
            "blood": LifecycleDetector(self, g("blood_detector_node").value),
        }
        self.pub_state = self.create_publisher(String, g("state_topic").value, 10)
        self.create_subscription(String, g("command_topic").value, self._on_command, 10)

        self._active_mode = "idle"
        self._requested_mode = None
        self._request_lock = threading.Lock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        if self.preload_models:
            self._publish_state(
                "PRELOADING", "loading Tool, Hand and Blood models into memory")
        else:
            self._publish_state("IDLE", "waiting for DETECT_TOOL/HAND/BLOOD")
        self._worker.start()
        self.get_logger().info(
            f"signal-driven coordinator ready on {g('command_topic').value}")

    def _on_command(self, msg: String) -> None:
        raw = msg.data.strip().upper()
        requested = COMMANDS.get(raw)
        if requested is None:
            self.get_logger().warn(
                f"unknown command {msg.data!r}; expected "
                "DETECT_TOOL, DETECT_HAND, DETECT_BLOOD or STOP")
            return
        with self._request_lock:
            self._requested_mode = requested
        self._request_event.set()
        self.get_logger().info(f"received signal: {raw} -> {requested}")

    def _take_request(self):
        with self._request_lock:
            requested = self._requested_mode
            self._requested_mode = None
            self._request_event.clear()
        return requested

    def _worker_loop(self) -> None:
        if self.preload_models:
            try:
                self._preload_detectors()
            except LifecycleDetectorError as exc:
                self.get_logger().error(f"startup preload failed: {exc}")
                self._publish_state("FAULT", f"startup preload failed: {exc}")
            else:
                self._publish_state(
                    "IDLE", "all detector models preloaded; waiting for command")

        while not self._stop_event.is_set() and rclpy.ok():
            if not self._request_event.wait(timeout=0.2):
                continue
            requested = self._take_request()
            if requested is None:
                continue
            try:
                self._switch_to(requested)
            except LifecycleDetectorError as exc:
                self.get_logger().error(f"mode switch failed: {exc}")
                self._active_mode = "idle"
                self._publish_state("FAULT", str(exc))

    def _preload_detectors(self) -> None:
        """Configure every detector sequentially and leave each INACTIVE."""
        for name, detector in self.detectors.items():
            self.get_logger().info(f"preloading {name} detector")
            detector.configure(load_timeout_sec=self.load_timeout)
            self.get_logger().info(
                f"preloaded {name} detector: {detector.state_name()}")

    def _switch_to(self, requested: str) -> None:
        if requested == self._active_mode:
            self._publish_state(requested.upper(), f"{requested} already active")
            return
        previous = self._active_mode
        self._publish_state("SWITCHING", f"stopping {previous}; starting {requested}")

        # Stop all non-requested detectors. This recovers safely even if the
        # coordinator restarted after an earlier detector was left active.
        for name, detector in self.detectors.items():
            if name != requested:
                detector.deactivate(release_gpu=self.release_gpu)

        if requested == "idle":
            for detector in self.detectors.values():
                detector.deactivate(release_gpu=self.release_gpu)
            self._active_mode = "idle"
            self._publish_state("IDLE", "all perception algorithms stopped")
            return

        self.detectors[requested].activate(load_timeout_sec=self.load_timeout)
        self._active_mode = requested
        self._publish_state(requested.upper(), f"{requested}_detection_node ACTIVE")

    def _publish_state(self, state: str, detail: str) -> None:
        payload = {"state": state, "active_mode": self._active_mode, "detail": detail}
        self.pub_state.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"[{state}] {detail}")

    def stop(self) -> None:
        self._stop_event.set()
        self._request_event.set()
        for detector in self.detectors.values():
            detector.deactivate(release_gpu=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionModeCoordinator()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
