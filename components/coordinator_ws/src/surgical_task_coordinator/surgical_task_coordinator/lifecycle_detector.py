"""Thin client for driving one detection node through its ROS 2 lifecycle.

Why lifecycle rather than a boolean "enabled" flag: the whole point of the
take-turn design is that the tool, hand and blood algorithms never run at
once, because they do not all fit on one RTX 3060 (12 GB) at the same time.
A boolean flag makes a node ignore camera frames but its model STAYS
RESIDENT IN VRAM -- which solves the CPU/latency problem and not the memory
problem. Lifecycle gives two distinct "off" levels:

  ACTIVE      -- processing frames
  INACTIVE    -- model still loaded, frames ignored. Re-activation is
                 instant. Costs VRAM.
  UNCONFIGURED-- on_cleanup() has released the model and emptied the CUDA
                 cache. Costs no VRAM, but re-entry pays the full model
                 load again (seconds, for MediaPipe + Depth-Anything V2).

The coordinator picks between those two via its release_gpu_between_modes
parameter, so the same code works whether or not all three models fit.
"""
import time

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.callback_groups import ReentrantCallbackGroup


# Human-readable names for the primary states we care about.
_STATE_NAMES = {
    State.PRIMARY_STATE_UNKNOWN: 'unknown',
    State.PRIMARY_STATE_UNCONFIGURED: 'unconfigured',
    State.PRIMARY_STATE_INACTIVE: 'inactive',
    State.PRIMARY_STATE_ACTIVE: 'active',
    State.PRIMARY_STATE_FINALIZED: 'finalized',
}


class LifecycleDetectorError(RuntimeError):
    """A transition did not reach the requested state in time."""


class LifecycleDetector:
    """Drives one lifecycle node by name.

    All methods BLOCK, and are meant to be called from the coordinator's
    worker thread -- never from an executor callback, or the service
    responses they wait on can never be delivered.
    """

    def __init__(self, node, detector_name, service_timeout_sec=10.0):
        self._node = node
        self.name = detector_name
        self._timeout = service_timeout_sec
        # Reentrant so these service calls can be in flight while the
        # coordinator's own subscriptions and timers keep being serviced.
        group = ReentrantCallbackGroup()
        self._get_state = node.create_client(
            GetState, f'/{detector_name}/get_state', callback_group=group)
        self._change_state = node.create_client(
            ChangeState, f'/{detector_name}/change_state', callback_group=group)

    # -- plumbing ---------------------------------------------------------

    def _spin_for(self, future, timeout_sec):
        """Wait for `future` WITHOUT spinning -- the executor is spinning in
        another thread, so spinning here too would be re-entrant."""
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                return None
            time.sleep(0.02)
        return future.result()

    def wait_until_available(self, timeout_sec=30.0):
        """True once the node's lifecycle services exist, i.e. it has started."""
        return (self._get_state.wait_for_service(timeout_sec=timeout_sec)
                and self._change_state.wait_for_service(timeout_sec=timeout_sec))

    def current_state(self):
        """Primary lifecycle state id, or None if the node did not answer."""
        if not self._get_state.service_is_ready():
            return None
        response = self._spin_for(
            self._get_state.call_async(GetState.Request()), self._timeout)
        return None if response is None else response.current_state.id

    def _transition(self, transition_id, label):
        request = ChangeState.Request()
        request.transition.id = transition_id
        response = self._spin_for(
            self._change_state.call_async(request), self._timeout)
        if response is None:
            raise LifecycleDetectorError(
                f'{self.name}: no response to "{label}" within {self._timeout}s')
        if not response.success:
            raise LifecycleDetectorError(f'{self.name}: "{label}" was rejected')

    # -- the two operations the coordinator actually needs ----------------

    def configure(self, load_timeout_sec=120.0):
        """Load a detector once and leave it INACTIVE.

        This is used during system startup to preload every model before a
        command arrives. INACTIVE detectors keep their models resident but
        do not consume camera frames.
        """
        if not self.wait_until_available(timeout_sec=self._timeout):
            raise LifecycleDetectorError(
                f'{self.name}: lifecycle services never appeared -- is the node running?')

        state = self.current_state()
        if state in (State.PRIMARY_STATE_INACTIVE, State.PRIMARY_STATE_ACTIVE):
            return
        if state != State.PRIMARY_STATE_UNCONFIGURED:
            raise LifecycleDetectorError(
                f'{self.name}: cannot configure from state '
                f'"{_STATE_NAMES.get(state, state)}"')

        saved, self._timeout = self._timeout, load_timeout_sec
        try:
            self._transition(Transition.TRANSITION_CONFIGURE, 'configure')
        finally:
            self._timeout = saved

    def activate(self, load_timeout_sec=120.0):
        """Bring the detector to ACTIVE from wherever it currently is.

        load_timeout_sec has to be generous: coming from UNCONFIGURED this
        includes loading MediaPipe and (for the hand node) Depth-Anything V2
        onto the GPU, which is far slower than a normal service call.
        """
        self.configure(load_timeout_sec=load_timeout_sec)
        state = self.current_state()
        if state == State.PRIMARY_STATE_ACTIVE:
            return

        if state != State.PRIMARY_STATE_INACTIVE:
            raise LifecycleDetectorError(
                f'{self.name}: cannot activate from state '
                f'"{_STATE_NAMES.get(state, state)}"')
        self._transition(Transition.TRANSITION_ACTIVATE, 'activate')

    def deactivate(self, release_gpu):
        """Stop the detector processing frames.

        release_gpu=True additionally cleans it up to UNCONFIGURED so its
        model leaves VRAM -- use when the three models do not co-fit.
        Never raises: shutting a detector down must not be able to strand
        the state machine, so failures are logged and swallowed.
        """
        try:
            state = self.current_state()
            if state is None:
                return
            if state == State.PRIMARY_STATE_ACTIVE:
                self._transition(Transition.TRANSITION_DEACTIVATE, 'deactivate')
                state = State.PRIMARY_STATE_INACTIVE
            if release_gpu and state == State.PRIMARY_STATE_INACTIVE:
                self._transition(Transition.TRANSITION_CLEANUP, 'cleanup')
        except LifecycleDetectorError as exc:
            self._node.get_logger().warn(f'deactivating {self.name} failed: {exc}')

    def state_name(self):
        return _STATE_NAMES.get(self.current_state(), 'unreachable')