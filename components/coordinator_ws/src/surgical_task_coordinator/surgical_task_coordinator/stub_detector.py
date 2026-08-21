"""A stand-in for a detection node that does not exist on this machine yet.

Two of the three algorithms (tool detection, blood detection) belong to other
team members and are not installed here. This stub takes their place so the
whole take-turn flow can be driven end-to-end today, and -- more usefully --
so it DOCUMENTS BY EXAMPLE what the coordinator requires of a detector:

  * be a ROS 2 lifecycle node, with a node name the coordinator knows
  * load your model in on_configure(), FREE IT in on_cleanup()
  * only consume camera frames while ACTIVE
  * publish geometry_msgs/PoseStamped on your result topic when you find
    your target, and keep publishing while you still see it
  * publish std_msgs/String health on <ns>/health

Everything else -- which camera topic you read, what your model is, what
extra rich messages you publish for other consumers -- is yours.

The stub fakes a model by sleeping for `fake_model_load_sec` in on_configure
and holding a dummy allocation, so switching turns has realistic timing.

Launch one per role, e.g.:
    ros2 run surgical_task_coordinator stub_detector --ros-args \
        -r __node:=tool_detection_node \
        -p result_topic:=/perception/cam_4/tool/target_pose
"""
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from std_msgs.msg import String


class StubDetector(LifecycleNode):

    def __init__(self):
        super().__init__('stub_detector')

        self.declare_parameter('result_topic', '/perception/cam_4/tool/target_pose')
        self.declare_parameter('health_topic', '')      # '' -> derive from node name
        self.declare_parameter('frame_id', 'cam4_color_optical_frame')
        self.declare_parameter('publish_rate_hz', 10.0)
        # How long the stub "searches" before it reports a target. Set to 0
        # to report immediately; set high to exercise the coordinator's
        # detection_timeout_sec.
        self.declare_parameter('detection_delay_sec', 1.5)
        self.declare_parameter('fake_model_load_sec', 2.0)
        # Fixed pose reported as the target, [x, y, z] in metres.
        self.declare_parameter('target_xyz', [0.10, 0.05, 0.40])

        self._model = None
        self._timer = None
        self._pub_pose = None
        self._activated_at = None

        health_topic = self.get_parameter('health_topic').value
        if not health_topic:
            name = self.get_name()
            task = 'hand' if 'hand' in name else 'blood' if 'blood' in name else 'tool'
            health_topic = f'/perception/cam_4/{task}/health'
        self._pub_health = self.create_lifecycle_publisher(String, health_topic, 10)
        self.create_timer(1.0, self._publish_health)

        self.get_logger().info(
            f'{self.get_name()} stub created (unconfigured). '
            f'result_topic={self.get_parameter("result_topic").value}')

    # -- lifecycle --------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Where a real detector loads its weights onto the GPU."""
        delay = self.get_parameter('fake_model_load_sec').value
        self.get_logger().info(f'configuring: loading model ({delay:.1f}s fake)...')
        time.sleep(delay)
        self._model = bytearray(1024)        # stands in for GPU weights
        self._pub_pose = self.create_lifecycle_publisher(
            PoseStamped, self.get_parameter('result_topic').value, 10)
        self.get_logger().info('configured: model resident, not processing')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Our turn -- start consuming frames."""
        self._activated_at = time.monotonic()
        rate = max(self.get_parameter('publish_rate_hz').value, 1e-3)
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info('ACTIVE: searching for target')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Turn is over -- stop processing, but keep the model loaded."""
        if self._timer is not None:
            self._timer.cancel()
            self.destroy_timer(self._timer)
            self._timer = None
        self.get_logger().info('INACTIVE: stopped processing, model still loaded')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """Release the model so its VRAM goes back to the pool.

        A real detector does the equivalent here: drop references to the
        model/session and call torch.cuda.empty_cache().
        """
        self._model = None
        if self._pub_pose is not None:
            self.destroy_lifecycle_publisher(self._pub_pose)
            self._pub_pose = None
        self.get_logger().info('cleaned up: model released')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._model = None
        return TransitionCallbackReturn.SUCCESS

    # -- fake detection ---------------------------------------------------

    def _tick(self):
        elapsed = time.monotonic() - self._activated_at
        delay = self.get_parameter('detection_delay_sec').value
        if elapsed < delay:
            return
        x, y, z = self.get_parameter('target_xyz').value
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
        msg.pose.orientation.w = 1.0
        self._pub_pose.publish(msg)
        self.get_logger().info(f'target reported at ({x:.2f}, {y:.2f}, {z:.2f})',
                                throttle_duration_sec=2.0)

    def _publish_health(self):
        # Matches the ARPA-H contract's std_msgs/String health convention.
        state = self._state_machine.current_state[1] if self._state_machine else 'unknown'
        self._pub_health.publish(String(data=f'{{"node": "{self.get_name()}", '
                                                f'"state": "{state}", "stub": true}}'))


def main(args=None):
    rclpy.init(args=args)
    node = StubDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
