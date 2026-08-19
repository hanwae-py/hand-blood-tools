"""Lifecycle gate that controls a preloaded v1.4 Tool node."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


def reliable_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class V14ToolLifecycleGate(LifecycleNode):
    """Expose lifecycle services for v1.4 without changing its model code path."""

    def __init__(self):
        super().__init__('tool_detection_node')
        self.declare_parameter(
            'gate_topic', '/surgery/perception/cam4/tool_processing_enabled'
        )
        self._topic = str(self.get_parameter('gate_topic').value)
        self._publisher = self.create_publisher(Bool, self._topic, reliable_qos())
        self._publish(False)
        self.get_logger().info(
            f'v1.4 Tool lifecycle gate created (unconfigured). gate={self._topic}'
        )

    def _publish(self, enabled: bool) -> None:
        self._publisher.publish(Bool(data=enabled))

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self._publish(False)
        self.get_logger().info('configured: v1.4 model remains preloaded, processing paused')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._publish(True)
        self.get_logger().info('ACTIVE: enabling v1.4 RGB-D Tool processing')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._publish(False)
        self.get_logger().info('INACTIVE: v1.4 Tool processing paused, model still loaded')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._publish(False)
        self.get_logger().info('cleaned up: gate paused; stop the v1.4 node to release GPU memory')
        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    rclpy.init(args=args)
    node = V14ToolLifecycleGate()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
