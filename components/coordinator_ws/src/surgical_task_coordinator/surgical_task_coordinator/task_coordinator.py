"""Surgical robot task coordinator -- decides WHOSE TURN it is.

Three perception algorithms, written by three different people, share one
desktop and one GPU:

  tool detection   (RF-DETR)              -- 최비결
  hand keypoints   (MediaPipe + depth)    -- this repo's sibling, hand_keypoint_ros
  blood detection                         -- third member

They must never run at the same time. This node owns that sequencing, and
nothing else: it does NOT import any of the three algorithms, does not touch
a camera, and does not do any detection itself. It only

  1. listens for a task command (today typed by hand, later from the voice
     team's node),
  2. lifecycles exactly one detector into ACTIVE,
  3. waits for that detector's result pose,
  4. sends the robot a long-running action, and
  5. hands the turn to the next algorithm.

Command interface (std_msgs/String on /surgery/task/command)
    REQUEST_TOOL              start the tool -> grasp -> hand -> handover flow
    REQUEST_TOOL:scalpel      same, naming which tool
    SUCK_BLOOD                start the blood -> suction flow
    ABORT                     cancel whatever is running, return to IDLE

    std_msgs/String is deliberate: the voice team has not defined their
    message yet, and a plain string is what a human can publish from a
    terminal today. Swap it for their typed message once it exists -- only
    _on_command() below needs to change.

State is published as surgical_task_interfaces/TaskState on
/surgery/task/state, on every transition plus a 1 Hz heartbeat.

Threading: the state machine runs on its own worker thread and BLOCKS on
service calls, action results and detector poses. The executor spins in the
main thread. This is why every client here uses a ReentrantCallbackGroup --
a single-threaded/mutually-exclusive setup would deadlock the moment the
worker waited on something the executor still had to deliver.
"""
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from surgical_task_interfaces.msg import TaskState
from surgical_task_interfaces.action import GraspTool, HandoverTool, SuctionBlood

from surgical_task_coordinator.lifecycle_detector import (
    LifecycleDetector, LifecycleDetectorError,
)


class State:
    """Keep in sync with the constants in TaskState.msg."""
    IDLE = 'IDLE'
    TOOL_DETECTION = 'TOOL_DETECTION'
    ROBOT_GRASP_TOOL = 'ROBOT_GRASP_TOOL'
    HAND_DETECTION = 'HAND_DETECTION'
    ROBOT_HANDOVER = 'ROBOT_HANDOVER'
    BLOOD_DETECTION = 'BLOOD_DETECTION'
    ROBOT_SUCTION = 'ROBOT_SUCTION'
    FAULT = 'FAULT'


class _PoseWaiter:
    """Latches the most recent pose on one detector's result topic.

    Cleared before a detector is activated so a stale pose from the previous
    round can never be mistaken for a fresh detection -- the detectors keep
    publishing right up until they are deactivated, so without this the
    coordinator would routinely "succeed" instantly on last round's data.
    """

    def __init__(self, node, topic, callback_group):
        self._event = threading.Event()
        self._pose = None
        self._lock = threading.Lock()
        self.topic = topic
        node.create_subscription(
            PoseStamped, topic, self._on_pose, 10, callback_group=callback_group)

    def _on_pose(self, msg):
        with self._lock:
            self._pose = msg
        self._event.set()

    def reset(self):
        with self._lock:
            self._pose = None
        self._event.clear()

    def wait(self, timeout_sec, abort_event=None):
        """Return the first pose received after reset(), or None on timeout.

        Waits in short slices rather than one long Event.wait() so that an
        ABORT does not have to sit through the full detection timeout before
        anyone notices it.
        """
        deadline = time.monotonic() + timeout_sec
        while True:
            if abort_event is not None and abort_event.is_set():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self._event.wait(timeout=min(0.1, remaining)):
                with self._lock:
                    return self._pose


class TaskCoordinator(Node):

    def __init__(self):
        super().__init__('task_coordinator')

        self.declare_parameter('command_topic', '/surgery/task/command')
        self.declare_parameter('state_topic', '/surgery/task/state')

        # Lifecycle node names. These must match the node names the three
        # detectors are launched with.
        self.declare_parameter('tool_detector_node', 'tool_detection_node')
        self.declare_parameter('hand_detector_node', 'hand_detection_node')
        self.declare_parameter('blood_detector_node', 'blood_detection_node')

        # Where each detector publishes the pose the robot should go to.
        # Uniformly geometry_msgs/PoseStamped so this node needs no build
        # dependency on any detector's private message package.
        self.declare_parameter('tool_pose_topic', '/surgery/perception/cam4/tool_target_pose')
        self.declare_parameter('hand_pose_topic', '/surgery/perception/cam4/hand_target_pose')
        self.declare_parameter('blood_pose_topic', '/surgery/perception/cam4/blood_target_pose')

        # True: clean detectors all the way to UNCONFIGURED between turns so
        # their models leave VRAM. Costs a model reload (seconds) each turn.
        # Set false once you have confirmed with `nvidia-smi` that all three
        # models co-fit -- turns then switch almost instantly.
        self.declare_parameter('release_gpu_between_tasks', True)

        self.declare_parameter('detection_timeout_sec', 30.0)
        self.declare_parameter('model_load_timeout_sec', 120.0)
        self.declare_parameter('robot_action_timeout_sec', 120.0)
        self.declare_parameter('default_tool_name', 'mayo_scissors')

        g = self.get_parameter
        self.release_gpu = g('release_gpu_between_tasks').value
        self.detection_timeout = g('detection_timeout_sec').value
        self.model_load_timeout = g('model_load_timeout_sec').value
        self.robot_timeout = g('robot_action_timeout_sec').value
        self.default_tool_name = g('default_tool_name').value

        group = ReentrantCallbackGroup()

        self.detectors = {
            'tool': LifecycleDetector(self, g('tool_detector_node').value),
            'hand': LifecycleDetector(self, g('hand_detector_node').value),
            'blood': LifecycleDetector(self, g('blood_detector_node').value),
        }
        self.poses = {
            'tool': _PoseWaiter(self, g('tool_pose_topic').value, group),
            'hand': _PoseWaiter(self, g('hand_pose_topic').value, group),
            'blood': _PoseWaiter(self, g('blood_pose_topic').value, group),
        }

        self.grasp_client = ActionClient(
            self, GraspTool, '/surgery/robot/grasp_tool', callback_group=group)
        self.handover_client = ActionClient(
            self, HandoverTool, '/surgery/robot/handover_tool', callback_group=group)
        self.suction_client = ActionClient(
            self, SuctionBlood, '/surgery/robot/suction_blood', callback_group=group)

        self.pub_state = self.create_publisher(TaskState, g('state_topic').value, 10)
        self.create_subscription(
            String, g('command_topic').value, self._on_command, 10, callback_group=group)

        self._state = State.IDLE
        self._active_detector = ''
        self._detail = 'waiting for a command'
        self._state_entered_at = self.get_clock().now()
        self._state_lock = threading.Lock()

        self._abort = threading.Event()
        self._worker = None            # the thread running the current flow
        self._goal_handle = None       # in-flight robot goal, for ABORT

        self.create_timer(1.0, self._publish_state, callback_group=group)

        self.get_logger().info(
            f'task_coordinator ready. Send a command with:\n'
            f"  ros2 topic pub --once -w 1 {g('command_topic').value} std_msgs/msg/String "
            f'"{{data: \'REQUEST_TOOL\'}}"\n'
            f'  (-w 1 matters: without it the publisher can exit before discovery '
            f'matches, and the command is silently dropped)\n'
            f'  release_gpu_between_tasks={self.release_gpu}')

    # -- state bookkeeping ------------------------------------------------

    def _set_state(self, state, detail, active_detector=''):
        with self._state_lock:
            self._state = state
            self._detail = detail
            self._active_detector = active_detector
            self._state_entered_at = self.get_clock().now()
        self.get_logger().info(f'[{state}] {detail}')
        self._publish_state()

    def _publish_state(self):
        msg = TaskState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._state_lock:
            msg.state = self._state
            msg.detail = self._detail
            msg.active_detector = self._active_detector
            msg.state_entered_at = self._state_entered_at.to_msg()
        self.pub_state.publish(msg)

    def _busy(self):
        return self._worker is not None and self._worker.is_alive()

    # -- command entry point ----------------------------------------------

    def _on_command(self, msg):
        raw = msg.data.strip()
        if not raw:
            return
        verb, _, argument = raw.partition(':')
        verb = verb.strip().upper()
        argument = argument.strip()

        if verb == 'ABORT':
            self._request_abort()
            return

        if self._busy():
            self.get_logger().warn(
                f'ignoring "{raw}" -- still running {self._state}. Send ABORT first.')
            return

        if verb == 'REQUEST_TOOL':
            flow, args = self._flow_tool_handover, (argument or self.default_tool_name,)
        elif verb == 'SUCK_BLOOD':
            flow, args = self._flow_blood_suction, ()
        else:
            self.get_logger().warn(
                f'unknown command "{raw}" -- expected REQUEST_TOOL, SUCK_BLOOD or ABORT')
            return

        self._abort.clear()
        self._worker = threading.Thread(target=self._run_flow, args=(flow, args), daemon=True)
        self._worker.start()

    def _request_abort(self):
        if not self._busy():
            self.get_logger().info('ABORT ignored -- nothing is running')
            return
        self.get_logger().warn('ABORT requested')
        self._abort.set()
        # Cancel the robot goal too; a robot mid-motion will not notice a
        # Python flag on its own.
        handle = self._goal_handle
        if handle is not None:
            handle.cancel_goal_async()

    def _run_flow(self, flow, args):
        try:
            flow(*args)
        except LifecycleDetectorError as exc:
            self._fail(f'detector error: {exc}')
        except Exception as exc:                       # noqa: BLE001 - last resort
            import traceback
            self.get_logger().error('flow crashed:\n' + traceback.format_exc())
            self._fail(f'internal error: {exc}')
        finally:
            self._goal_handle = None
            self._shutdown_all_detectors()

    def _fail(self, detail):
        self._set_state(State.FAULT, detail)
        # FAULT is observable but must not be a dead end -- fall back to IDLE
        # so the next command is accepted.
        time.sleep(1.0)
        self._set_state(State.IDLE, f'recovered from: {detail}')

    def _shutdown_all_detectors(self):
        """Guarantee the invariant: at most one detector active, and after a
        flow ends, none. Called on every exit path including crash/abort."""
        for detector in self.detectors.values():
            detector.deactivate(release_gpu=self.release_gpu)

    # -- shared step helpers ----------------------------------------------

    def _aborted(self):
        if self._abort.is_set():
            self._set_state(State.IDLE, 'aborted by command')
            return True
        return False

    def _detect(self, key, state, description):
        """Activate one detector, wait for its pose, deactivate it again.

        Returns the PoseStamped, or None if the flow should stop.
        """
        detector = self.detectors[key]
        waiter = self.poses[key]

        # Check before spending seconds loading a model onto the GPU that we
        # are about to throw away anyway.
        if self._aborted():
            return None

        self._set_state(state, f'{description} (activating {detector.name})', detector.name)

        waiter.reset()          # before activate, so no stale pose can win
        detector.activate(load_timeout_sec=self.model_load_timeout)

        # An ABORT that arrived DURING activate() -- model loading is the
        # longest blocking step in the whole flow, so this is the most likely
        # place for one to land. Shut the detector straight back down instead
        # of running a pointless detection cycle first.
        if self._abort.is_set():
            detector.deactivate(release_gpu=self.release_gpu)
            self._set_state(State.IDLE, 'aborted by command')
            return None

        self._set_state(state, description, detector.name)
        try:
            pose = waiter.wait(self.detection_timeout, abort_event=self._abort)
        finally:
            detector.deactivate(release_gpu=self.release_gpu)

        if self._aborted():
            return None
        if pose is None:
            self._fail(f'{description}: nothing found within '
                       f'{self.detection_timeout:.0f}s on {waiter.topic}')
            return None
        self.get_logger().info(
            f'{key} target: ({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, '
            f'{pose.pose.position.z:.3f}) in frame "{pose.header.frame_id}"')
        return pose

    def _run_robot_action(self, client, goal, state, description):
        """Send a robot goal and block on its result. True on success."""
        self._set_state(state, description)
        if not client.wait_for_server(timeout_sec=10.0):
            self._fail(f'{description}: robot action server unavailable')
            return False

        send_future = client.send_goal_async(
            goal, feedback_callback=lambda fb: self.get_logger().info(
                f'  {description}: {fb.feedback.stage} '
                f'({fb.feedback.progress * 100:.0f}%)', throttle_duration_sec=1.0))
        if self._wait(send_future, self.robot_timeout) is None:
            self._fail(f'{description}: robot never accepted the goal')
            return False

        handle = send_future.result()
        if not handle.accepted:
            self._fail(f'{description}: robot rejected the goal')
            return False

        self._goal_handle = handle
        result_future = handle.get_result_async()
        wrapper = self._wait(result_future, self.robot_timeout)
        self._goal_handle = None

        if wrapper is None:
            self._fail(f'{description}: timed out after {self.robot_timeout:.0f}s')
            return False
        if self._aborted():
            return False
        if not wrapper.result.success:
            self._fail(f'{description}: {wrapper.result.message}')
            return False
        return True

    def _wait(self, future, timeout_sec):
        """Block until `future` resolves. Does NOT spin -- the executor is
        spinning on the main thread; spinning here would be re-entrant."""
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if self._abort.is_set() or not rclpy.ok():
                return None
            if time.monotonic() > deadline:
                return None
            time.sleep(0.02)
        return future.result()

    # -- the two flows ----------------------------------------------------

    def _flow_tool_handover(self, tool_name):
        """REQUEST_TOOL: tool detection -> grasp -> hand detection -> handover."""
        tool_pose = self._detect('tool', State.TOOL_DETECTION, f'locating tool "{tool_name}"')
        if tool_pose is None:
            return

        goal = GraspTool.Goal(tool_name=tool_name, tool_pose=tool_pose)
        if not self._run_robot_action(self.grasp_client, goal,
                                       State.ROBOT_GRASP_TOOL, f'grasping "{tool_name}"'):
            return
        if self._aborted():
            return

        # Only now does the hand algorithm get its turn -- the surgeon's hand
        # position at grasp time is irrelevant, what matters is where it is
        # when the robot is ready to hand over.
        hand_pose = self._detect('hand', State.HAND_DETECTION, "locating surgeon's hand")
        if hand_pose is None:
            return

        goal = HandoverTool.Goal(tool_name=tool_name, hand_pose=hand_pose)
        if not self._run_robot_action(self.handover_client, goal,
                                       State.ROBOT_HANDOVER, f'handing over "{tool_name}"'):
            return

        self._set_state(State.IDLE, f'"{tool_name}" handed to the surgeon')

    def _flow_blood_suction(self):
        """SUCK_BLOOD: blood detection -> suction."""
        blood_pose = self._detect('blood', State.BLOOD_DETECTION, 'locating blood')
        if blood_pose is None:
            return

        goal = SuctionBlood.Goal(target_pose=blood_pose, duration_sec=3.0)
        if not self._run_robot_action(self.suction_client, goal,
                                       State.ROBOT_SUCTION, 'suctioning blood'):
            return

        self._set_state(State.IDLE, 'suction complete')


def main(args=None):
    rclpy.init(args=args)
    node = TaskCoordinator()
    # MultiThreadedExecutor is required, not a preference: the worker thread
    # blocks on futures that this executor has to deliver.
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
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
