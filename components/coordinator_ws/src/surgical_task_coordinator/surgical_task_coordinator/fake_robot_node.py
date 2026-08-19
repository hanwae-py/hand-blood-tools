"""A stand-in for the robot-control node (flow "sections 3-5").

Serves the three long-running operations the coordinator drives, by moving
nowhere and reporting success after a delay. Its purpose is to make the
state machine runnable before the robot team's node exists, and to pin down
the action contract they will have to implement:

    /surgery/robot/grasp_tool       surgical_task_interfaces/action/GraspTool
    /surgery/robot/handover_tool    surgical_task_interfaces/action/HandoverTool
    /surgery/robot/suction_blood    surgical_task_interfaces/action/SuctionBlood

Actions, not services, because these take seconds, can fail, and the
coordinator must be able to cancel them when the surgeon aborts. This stub
honours cancellation so the ABORT path is genuinely exercised.

Set fail_next:=true to make the next goal fail, to test the coordinator's
FAULT handling without needing a real robot to go wrong.
"""
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from surgical_task_interfaces.action import GraspTool, HandoverTool, SuctionBlood


class FakeRobotNode(Node):

    def __init__(self):
        super().__init__('fake_robot_node')

        self.declare_parameter('motion_duration_sec', 3.0)
        self.declare_parameter('fail_next', False)

        group = ReentrantCallbackGroup()
        self._servers = [
            ActionServer(self, GraspTool, '/surgery/robot/grasp_tool',
                          execute_callback=self._make_executor('grasp', GraspTool),
                          goal_callback=self._on_goal, cancel_callback=self._on_cancel,
                          callback_group=group),
            ActionServer(self, HandoverTool, '/surgery/robot/handover_tool',
                          execute_callback=self._make_executor('handover', HandoverTool),
                          goal_callback=self._on_goal, cancel_callback=self._on_cancel,
                          callback_group=group),
            ActionServer(self, SuctionBlood, '/surgery/robot/suction_blood',
                          execute_callback=self._make_executor('suction', SuctionBlood),
                          goal_callback=self._on_goal, cancel_callback=self._on_cancel,
                          callback_group=group),
        ]
        self.get_logger().info(
            'fake_robot_node ready: grasp_tool / handover_tool / suction_blood')

    def _on_goal(self, goal_request):
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        self.get_logger().warn('robot goal cancellation accepted')
        return CancelResponse.ACCEPT

    def _make_executor(self, label, action_type):
        def execute(goal_handle):
            duration = self.get_parameter('motion_duration_sec').value
            self.get_logger().info(f'{label}: starting ({duration:.1f}s)')

            steps = max(int(duration * 5), 1)
            for i in range(steps):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().warn(f'{label}: canceled')
                    return action_type.Result(success=False, message='canceled')
                feedback = action_type.Feedback()
                feedback.stage = f'{label} in progress'
                feedback.progress = (i + 1) / steps
                goal_handle.publish_feedback(feedback)
                time.sleep(duration / steps)

            if self.get_parameter('fail_next').value:
                # One-shot: clear it so the next goal succeeds again.
                self.set_parameters(
                    [rclpy.parameter.Parameter('fail_next',
                                                rclpy.Parameter.Type.BOOL, False)])
                goal_handle.abort()
                self.get_logger().error(f'{label}: injected failure')
                return action_type.Result(success=False,
                                           message=f'{label} failed (injected)')

            goal_handle.succeed()
            self.get_logger().info(f'{label}: done')
            return action_type.Result(success=True, message=f'{label} complete')
        return execute


def main(args=None):
    rclpy.init(args=args)
    node = FakeRobotNode()
    # Multi-threaded so a goal's blocking execute_callback cannot stop the
    # server from servicing a cancel request for that same goal.
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
