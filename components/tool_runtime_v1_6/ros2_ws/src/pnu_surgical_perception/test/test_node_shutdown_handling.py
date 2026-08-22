"""Regression tests for controlled systemd/executor shutdowns."""

import pytest

from pnu_surgical_perception import final_overlay_compositor
from pnu_surgical_perception import native_depth_pose_node
from pnu_surgical_perception import perception_ingress


@pytest.mark.parametrize(
    ('module', 'node_name', 'needs_stop'),
    (
        (native_depth_pose_node, 'NativeDepthPoseNode', True),
        (perception_ingress, 'PerceptionIngress', False),
        (final_overlay_compositor, 'FinalOverlayCompositor', False),
    ),
)
def test_main_handles_external_shutdown_without_a_traceback(
    monkeypatch, module, node_name, needs_stop
):
    """An executor shutdown is normal during systemd unit replacement."""
    events = []

    class FakeNode:
        def stop(self):
            events.append('stop')

        def destroy_node(self):
            events.append('destroy')

    fake = FakeNode()
    monkeypatch.setattr(module, node_name, lambda: fake)
    monkeypatch.setattr(module.rclpy, 'init', lambda *, args=None: events.append('init'))

    def external_shutdown(_node):
        raise module.ExternalShutdownException()

    monkeypatch.setattr(module.rclpy, 'spin', external_shutdown)
    if needs_stop:
        monkeypatch.setattr(module.rclpy, 'ok', lambda: True)
        monkeypatch.setattr(module.rclpy, 'shutdown', lambda: events.append('shutdown'))
    else:
        monkeypatch.setattr(module.rclpy, 'try_shutdown', lambda: events.append('shutdown'))

    module.main()

    assert events[0] == 'init'
    if needs_stop:
        assert events[1:] == ['stop', 'destroy', 'shutdown']
    else:
        assert events[1:] == ['destroy', 'shutdown']
