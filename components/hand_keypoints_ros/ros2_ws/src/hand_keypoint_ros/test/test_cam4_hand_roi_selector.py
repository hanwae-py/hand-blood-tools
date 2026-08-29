import importlib.util
from pathlib import Path

import pytest
import yaml


def _find_repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'scripts' / 'select_cam4_hand_roi.py').exists():
            return parent
    raise RuntimeError('repository root not found')


ROOT = _find_repo_root()
SPEC = importlib.util.spec_from_file_location(
    'select_cam4_hand_roi', ROOT / 'scripts' / 'select_cam4_hand_roi.py')
ROI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROI)


def test_pixel_box_is_sorted_clamped_and_normalized():
    assert ROI.pixel_box_to_normalized(
        (800, 600, 200, 100), 1000, 800) == pytest.approx(
            (0.2, 0.801, 0.125, 0.75125))


def test_last_pixel_reaches_full_frame_boundary():
    assert ROI.pixel_box_to_normalized(
        (0, 0, 1279, 719), 1280, 720) == ROI.FULL_FRAME_ROI


def test_tiny_or_invalid_roi_is_rejected():
    with pytest.raises(ValueError, match='too small'):
        ROI.pixel_box_to_normalized((10, 10, 20, 20), 1280, 720)
    with pytest.raises(ValueError, match='x_min'):
        ROI.validate_normalized_roi((0.8, 0.2, 0.1, 0.9))


def test_atomic_config_round_trip_contains_only_roi_parameters(tmp_path):
    path = tmp_path / 'cam4_hand_roi.yaml'
    expected = (0.125, 0.875, 0.2, 0.8)
    ROI.write_roi_config_atomic(
        path, expected, image_width=1280, image_height=720)
    assert ROI.load_roi_config(path) == pytest.approx(expected)
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    params = payload['/**']['ros__parameters']
    assert set(params) == set(ROI.ROI_KEYS)


def test_atomic_write_refuses_concurrent_config_change(tmp_path):
    path = tmp_path / 'cam4_hand_roi.yaml'
    ROI.write_roi_config_atomic(path, ROI.FULL_FRAME_ROI)
    original_sha = ROI.config_sha256(path)
    changed = ROI.render_roi_config((0.1, 0.9, 0.1, 0.9))
    path.write_text(changed, encoding='utf-8')
    changed_bytes = path.read_bytes()

    with pytest.raises(RuntimeError, match='changed while'):
        ROI.write_roi_config_atomic(
            path, (0.2, 0.8, 0.2, 0.8),
            expected_sha256=original_sha)
    assert path.read_bytes() == changed_bytes


def test_second_selector_lock_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    config = tmp_path / 'cam4_hand_roi.yaml'
    ROI.write_roi_config_atomic(config, ROI.FULL_FRAME_ROI)
    with ROI.exclusive_selector_lock(config):
        with pytest.raises(RuntimeError, match='already running'):
            with ROI.exclusive_selector_lock(config):
                pass


def test_repo_config_is_valid_and_runner_applies_it_after_calibration():
    config = ROOT / 'config' / 'cam4_hand_roi.yaml'
    configured_roi = ROI.load_roi_config(config)
    assert ROI.validate_normalized_roi(configured_roi) == configured_roi
    payload = yaml.safe_load(config.read_text(encoding='utf-8'))
    assert set(payload['/**']['ros__parameters']) == set(ROI.ROI_KEYS)

    runner = (ROOT / 'scripts' / 'run_hand_cam4.sh').read_text(
        encoding='utf-8')
    calibration = 'config/cam4_depth_to_color.yaml'
    roi = 'config/cam4_hand_roi.yaml'
    assert runner.index(calibration) < runner.index(roi)


def test_roi_selector_runner_restarts_cam4_and_final_overlay_together():
    runner = (ROOT / 'scripts' / 'run_cam4_hand_roi_selector.sh').read_text(
        encoding='utf-8')
    cam4 = '--restart-service "taskplanner-perception-cam4-ingress.service"'
    final = '--restart-service "taskplanner-perception-final-overlay.service"'
    assert runner.count('--restart-service') == 2
    assert runner.index(cam4) < runner.index(final)


def test_restart_services_are_deduplicated_in_one_systemd_transaction(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return type('Completed', (), {'returncode': 0, 'stderr': ''})()

    monkeypatch.setattr(ROI.subprocess, 'run', fake_run)
    services, completed = ROI.restart_user_services([
        'cam4.service', 'final.service', 'cam4.service'])
    assert services == ['cam4.service', 'final.service']
    assert completed.returncode == 0
    assert calls == [(
        ['systemctl', '--user', 'restart', 'cam4.service', 'final.service'],
        {'text': True, 'capture_output': True},
    )]
