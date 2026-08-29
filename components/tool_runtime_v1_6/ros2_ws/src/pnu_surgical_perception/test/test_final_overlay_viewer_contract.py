from pathlib import Path


def _find_repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'scripts' / 'run_final_overlay_viewer.sh').exists():
            return parent
    raise RuntimeError('repository root not found')


ROOT = _find_repo_root()


def test_viewer_unit_uses_transport_safe_local_overlay():
    unit = (
        ROOT
        / 'config/systemd/user/taskplanner-perception-quad-viewer.service'
    ).read_text(encoding='utf-8')
    assert 'Wants=taskplanner-perception-final-overlay.service' in unit
    assert 'Requires=taskplanner-perception-final-overlay.service' not in unit
    assert 'ExecStart=%h/projects/hand-blood-tools/scripts/run_final_overlay_viewer.sh' in unit


def test_viewer_runner_subscribes_directly_to_four_final_overlay_topics():
    runner = (ROOT / 'scripts/run_final_overlay_viewer.sh').read_text(
        encoding='utf-8')
    assert 'source "${RQT_OVERLAY}/install/setup.bash"' in runner
    for topic in (
        '/perception/cam_3/overlay',
        '/perception/cam_4/overlay',
        '/perception/suction/overlay',
        '/perception/right_ee/overlay',
    ):
        assert f'launch_view {topic}' in runner
    assert '/perception/debug/final_overlay' not in runner
    assert 'position_x11_window.py' in runner
    assert '--pid "${pid}"' in runner


def test_viewer_window_geometry_uses_window_manager_protocol():
    helper = (ROOT / 'scripts/position_x11_window.py').read_text(
        encoding='utf-8')
    assert "b'_NET_MOVERESIZE_WINDOW'" in helper
    assert "selector.add_argument('--pid', type=int)" in helper
    assert "parser.add_argument('--width', type=int, required=True)" in helper
    assert "parser.add_argument('--height', type=int, required=True)" in helper


def test_local_rqt_patch_selects_and_persists_item_data():
    source = (
        ROOT
        / 'components/rqt_image_view_overlay_ws/src/rqt_image_view'
        / 'src/rqt_image_view/image_view.cpp'
    ).read_text(encoding='utf-8')
    assert source.count('currentData().toString()') >= 2
    assert 'findData(QVariant(topic))' in source
    assert 'index = ui_.topics_combo_box->count() - 1;' in source
