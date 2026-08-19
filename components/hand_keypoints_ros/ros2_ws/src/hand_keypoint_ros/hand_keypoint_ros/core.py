"""Shared hand-keypoints + palm-6D detection logic.

This module has NO argparse / file-loop / ROS code — it's the single
source of truth for the per-frame math, used by both:
  - scripts/run_hand_keypoints.py   (offline CLI, reads video files)
  - the ROS2 hand-detection node    (reads live/bag camera topics)

Keeping this logic in one place means both consumers stay in sync
automatically instead of drifting apart as separate copies.
"""
import contextlib
import json
import os
import sys

import cv2
import numpy as np

MEDIAPIPE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/latest/hand_landmarker.task')
MEDIAPIPE_CACHE = os.path.expanduser('~/.cache/mediapipe')
DEFAULT_DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'

JOINT_NAMES = [
    'wrist',
    'thumb_CMC', 'thumb_MCP', 'thumb_IP', 'thumb_TIP',
    'index_MCP', 'index_PIP', 'index_DIP', 'index_TIP',
    'middle_MCP', 'middle_PIP', 'middle_DIP', 'middle_TIP',
    'ring_MCP', 'ring_PIP', 'ring_DIP', 'ring_TIP',
    'pinky_MCP', 'pinky_PIP', 'pinky_DIP', 'pinky_TIP',
]

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def load_intrinsics(calib_json_path, cam_key='cam_4'):
    with open(calib_json_path) as f:
        cfg = json.load(f)
    info = cfg['camera_info'][f'/synced/{cam_key}/color/camera_info']
    k = info['k']
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


_WIN_OFFSETS_CACHE = {}
def _win_offsets(win):
    if win not in _WIN_OFFSETS_CACHE:
        dv, du = np.meshgrid(np.arange(-win, win + 1),
                              np.arange(-win, win + 1), indexing='ij')
        _WIN_OFFSETS_CACHE[win] = (du.ravel(), dv.ravel())
    return _WIN_OFFSETS_CACHE[win]


def sample_depth_batch(depth_frame, uv, win=3):
    """Median of valid (>0) depth in a (2*win+1)^2 window per keypoint.
    depth_frame must already be in METRES. Returns (depth_m, valid_mask)."""
    H, W = depth_frame.shape
    du, dv = _win_offsets(win)
    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, H - 1)
    us = np.clip(u[:, None] + du[None, :], 0, W - 1)
    vs = np.clip(v[:, None] + dv[None, :], 0, H - 1)
    patches = depth_frame[vs, us].astype(np.float32)
    valid_2d = patches > 0
    patches_nan = np.where(valid_2d, patches, np.nan)
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(patches_nan, axis=1)
    valid = ~np.isnan(med)
    med = np.where(valid, med, 0.0)
    return med, valid


def palm_frame_v2(j0, j2, j9, j17):
    """v2 palm frame: origin = midpoint(wrist, middle_MCP);
    X: wrist->middle_MCP, Y: Gram-Schmidt(midpoint(wrist,pinky_MCP) - index_MCP, X), Z = X x Y."""
    origin = 0.5 * (j0 + j9)
    x_raw = j9 - j0
    y_raw = 0.5 * (j0 + j17) - j2
    x_axis = x_raw / (np.linalg.norm(x_raw) + 1e-9)
    y_axis = y_raw - x_axis * float(x_axis @ y_raw)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    z_axis = np.cross(x_axis, y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    return origin.astype(np.float32), R


def rot_to_quat_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / (t + 1.0) ** 0.5
        return [float(0.25 / s), float((R[2, 1] - R[1, 2]) * s),
                float((R[0, 2] - R[2, 0]) * s), float((R[1, 0] - R[0, 1]) * s)]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        return [float((R[2, 1] - R[1, 2]) / s), float(0.25 * s),
                float((R[0, 1] + R[1, 0]) / s), float((R[0, 2] + R[2, 0]) / s)]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        return [float((R[0, 2] - R[2, 0]) / s), float((R[0, 1] + R[1, 0]) / s),
                float(0.25 * s), float((R[1, 2] + R[2, 1]) / s)]
    s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
    return [float((R[1, 0] - R[0, 1]) / s), float((R[0, 2] + R[2, 0]) / s),
            float((R[1, 2] + R[2, 1]) / s), float(0.25 * s)]


def project(K, pts3d_m):
    pts = pts3d_m @ K.T
    z = pts[:, 2:3]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return pts[:, :2] / z


def draw_skeleton_bones(frame, uv, valid, colour=(180, 180, 180),
                        draw_invalid=False):
    """Draw a hand skeleton.

    ``valid`` normally means a keypoint has usable metric depth. During a
    live RGB/depth-alignment check we still want to show MediaPipe's 2-D hand
    result while deliberately withholding its 3-D values; ``draw_invalid``
    makes that 2-D-only case visible without changing validity data.
    """
    for a, b in HAND_BONES:
        if not draw_invalid and (not valid[a] or not valid[b]):
            continue
        cv2.line(frame, (int(uv[a][0]), int(uv[a][1])),
                  (int(uv[b][0]), int(uv[b][1])), colour, 2)
    for i in range(21):
        if not draw_invalid and not valid[i]:
            continue
        cv2.circle(frame, (int(uv[i][0]), int(uv[i][1])), 5, (255, 255, 255), -1)
        cv2.circle(frame, (int(uv[i][0]), int(uv[i][1])), 5, (0, 0, 0), 1)


def draw_gizmo(frame, K, origin, R, axis_len_m=0.06):
    o2 = project(K, origin[None, :])[0]
    o = (int(o2[0]), int(o2[1]))
    for i, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        tip = origin + R[:, i] * axis_len_m
        t2 = project(K, tip[None, :])[0]
        cv2.arrowedLine(frame, o, (int(t2[0]), int(t2[1])), colour, 3, tipLength=0.25)


def ensure_mediapipe_model():
    os.makedirs(MEDIAPIPE_CACHE, exist_ok=True)
    p = os.path.join(MEDIAPIPE_CACHE, 'hand_landmarker.task')
    if not os.path.exists(p):
        import urllib.request
        print(f'downloading {MEDIAPIPE_MODEL_URL} ...')
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, p)
    return p


def robot_position_target_px(robot_position, W, H):
    """Map a robot's physical corner position to the target pixel corner
    in the FRAME (not the robot's own left/right). This camera is not
    mirrored, so left/right is flipped going from robot-space to
    frame-space; top/bottom is unchanged. Only 'top-left' -> frame
    'top-right' is empirically validated so far.
    Returns ((x_px, y_px), frame_corner_label)."""
    vert, horiz = robot_position.split('-')
    frame_horiz = 'right' if horiz == 'left' else 'left'
    y = 0 if vert == 'top' else H
    x = 0 if frame_horiz == 'left' else W
    return (float(x), float(y)), f'{vert}-{frame_horiz}'


def find_depth_h5(rgb_path):
    """Auto-detect a sibling real-depth HDF5 for this lab's directory layout:
    <root>/rgb/<prefix>_rgb_<suffix>.avi -> <root>/depth_raw/<prefix>_depth_raw_<suffix>.h5
    Returns the path if it exists, else None."""
    base = os.path.basename(rgb_path)
    if '_rgb_' not in base:
        return None
    depth_base = base.replace('_rgb_', '_depth_raw_').rsplit('.', 1)[0] + '.h5'
    rgb_dir = os.path.dirname(os.path.abspath(rgb_path))
    depth_dir = os.path.join(os.path.dirname(rgb_dir), 'depth_raw')
    candidate = os.path.join(depth_dir, depth_base)
    return candidate if os.path.isfile(candidate) else None


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------

def load_mediapipe(max_hands, cpu_only=False):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    model_path = ensure_mediapipe_model()
    common_kwargs = dict(
        running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    if cpu_only:
        base_opts = BaseOptions(model_asset_path=model_path)
        hand_det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_opts, **common_kwargs))
        print('MediaPipe hand landmarker ready (CPU, forced by --cpu-only)')
        return mp, hand_det
    try:
        base_opts = BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU)
        hand_det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_opts, **common_kwargs))
        print('MediaPipe hand landmarker ready (GPU delegate)')
    except Exception as e:
        print(f'GPU delegate failed ({e}) -- falling back to CPU')
        base_opts = BaseOptions(model_asset_path=model_path)
        hand_det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_opts, **common_kwargs))
        print('MediaPipe hand landmarker ready (CPU)')
    return mp, hand_det


def load_mono_depth_model(depth_model_name, cpu_only=False):
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForDepthEstimation
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    device = torch.device('cuda' if (torch.cuda.is_available() and not cpu_only) else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    processor = AutoImageProcessor.from_pretrained(depth_model_name)
    model = AutoModelForDepthEstimation.from_pretrained(
        depth_model_name, torch_dtype=dtype).to(device).eval()
    print(f'mono depth model ready: {depth_model_name}  ({device}, {dtype})')
    return torch, processor, model, device, dtype


def run_mono_depth(rgb_bgr, torch, processor, depth_model, device, dtype, H, W):
    """RGB (BGR, HxWx3 uint8) -> depth map in METRES (HxW float32), via Depth-Anything V2."""
    from PIL import Image
    rgb_np = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    rgb_pil = Image.fromarray(rgb_np)
    with torch.no_grad():
        inp = processor(images=rgb_pil, return_tensors='pt')
        inp = {k: (v.to(device=device, dtype=dtype) if v.dtype.is_floating_point
                   else v.to(device=device)) for k, v in inp.items()}
        dpred = depth_model(**inp).predicted_depth
        dpred = torch.nn.functional.interpolate(
            dpred.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze()
    return dpred.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# the shared per-frame pipeline
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet_native_stderr():
    """Silence MediaPipe's C++ stderr for the duration of one detect call.

    With the GPU delegate, MediaPipe 0.10.x logs
        tensor.cc:410] Tensors are designed for single writes...
    at ERROR level several times PER FRAME. It's spurious here -- the
    landmarker is only ever driven from a single thread (the rclpy spin
    thread, or the offline CLI's frame loop) -- but at 15+ Hz it buries
    every other line of output.

    It has to be suppressed at the file-descriptor level: MediaPipe 0.10.18
    logs via absl, not glog, so the usual GLOG_minloglevel/GLOG_logtostderr
    env vars have no effect on it (verified -- the spam survives them).

    Scoped to detect_for_video() only, so Python-side logging (ROS loggers,
    tracebacks) is never swallowed. Set HAND_KEYPOINTS_MEDIAPIPE_LOGS=1 to
    disable the suppression and see MediaPipe's native output again.
    """
    if os.environ.get('HAND_KEYPOINTS_MEDIAPIPE_LOGS') == '1':
        yield
        return
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def process_frame(frame_bgr, depth_map_m, hand_det, mp, K, fx, fy, cx, cy, W, H,
                   ts_ms, region=(0.0, 1.0, 0.0, 1.0),
                   target_px=None, robot_position_label=None, frame_corner_label=None,
                   flip_handedness=False, draw_overlay=True, depth_source_label='REAL DEPTH',
                   allow_2d_only=False):
    """Run one frame through: MediaPipe detection -> per-keypoint depth
    lookup -> metric backprojection -> optional robot-handoff single-hand
    selection -> palm 6D -> (optional) overlay drawing.

    depth_map_m: HxW float32, metres, already aligned to frame_bgr.
    ts_ms: integer milliseconds, MUST be strictly increasing across
        successive calls on the same hand_det instance (MediaPipe's VIDEO
        running mode requirement) — e.g. frame_idx * 1000 / fps for a
        video file, or a ROS message header stamp in ms for a live topic.
    region: (x_min, x_max, y_min, y_max), normalised [0,1] keep-box
        evaluated against each hand's 2D centroid.
    target_px + frame_corner_label: precomputed via robot_position_target_px()
        (pass both, or leave both None to keep every hand — original,
        default behaviour).
    robot_position_label: the raw --robot-position string, only used to
        annotate the overlay text; pass None to omit.

    Returns (row_hands, overlay_frame, total_valid_kps) — row_hands is a
    list of dicts: hand_index, handedness, joints_3d, joints_2d,
    kp_scores, kp_valid_depth, palm_6d. Identical shape whether called
    from the offline CLI or a ROS node.
    """
    x_min, x_max, y_min, y_max = region

    mp_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_rgb)
    with _quiet_native_stderr():
        result = hand_det.detect_for_video(mp_img, ts_ms)
    hands = result.hand_landmarks or []
    handedness = result.handedness or []

    candidates = []
    overlay = frame_bgr
    total_valid_kps = 0

    for h_i, lms in enumerate(hands):
        uv = np.array([(lm.x * W, lm.y * H) for lm in lms], dtype=np.float32)
        cu = float(np.mean(uv[:, 0])) / W
        cv_ = float(np.mean(uv[:, 1])) / H
        if not (x_min <= cu <= x_max and y_min <= cv_ <= y_max):
            continue

        depths_m, depth_ok = sample_depth_batch(depth_map_m, uv, win=3)
        valid = depth_ok
        z = depths_m
        X = (uv[:, 0] - cx) / fx * z
        Y = (uv[:, 1] - cy) / fy * z
        joints_3d = np.stack([X, Y, z], axis=1).astype(np.float32)
        joints_3d[~valid] = 0

        n_valid = int(valid.sum())
        total_valid_kps += n_valid
        # During an unapproved live RGB/depth-alignment check, retain the
        # 2-D MediaPipe result but keep all metric values invalid. Offline
        # and alignment-validated depth processing keep the previous gate.
        if n_valid < 4 and not allow_2d_only:
            continue

        hd = None
        if h_i < len(handedness) and handedness[h_i]:
            name = handedness[h_i][0].category_name
            if flip_handedness:
                name = 'Right' if name == 'Left' else 'Left'
            hd = {'label': name, 'score': float(handedness[h_i][0].score)}

        palm_pose = None
        gizmo = None
        if all(valid[i] for i in [0, 2, 9, 17]):
            origin, R = palm_frame_v2(joints_3d[0], joints_3d[2], joints_3d[9], joints_3d[17])
            palm_pose = {
                'translation': [float(v) for v in origin],
                'rotation_matrix': R.round(6).tolist(),
                'rotation_quat_wxyz': rot_to_quat_wxyz(R),
            }
            gizmo = (origin, R)

        candidates.append({
            'h_i': h_i, 'uv': uv, 'valid': valid, 'joints_3d': joints_3d,
            'hd': hd, 'palm_pose': palm_pose, 'gizmo': gizmo,
            'centroid_px': (cu * W, cv_ * H),
        })

    # ---- EXTRA, opt-in: robot handoff mode ------------------------------
    # Keeps only the single hand nearest the robot's target corner.
    # No-op (all candidates kept, original behaviour) when target_px is None.
    if target_px is not None and candidates:
        tx, ty = target_px
        def _dist2_to_target(c):
            px, py = c['centroid_px']
            return (px - tx) ** 2 + (py - ty) ** 2
        candidates = [min(candidates, key=_dist2_to_target)]

    row_hands = []
    for c in candidates:
        uv, valid, joints_3d = c['uv'], c['valid'], c['joints_3d']
        hd, palm_pose = c['hd'], c['palm_pose']

        if draw_overlay:
            if c['gizmo'] is not None:
                origin, R = c['gizmo']
                draw_gizmo(overlay, K, origin, R, axis_len_m=0.06)
            draw_skeleton_bones(overlay, uv, valid,
                                draw_invalid=allow_2d_only)
            if hd is not None:
                lbl = f'{hd["label"]} {hd["score"]:.2f}'
                wu, wv = int(uv[0][0]), int(uv[0][1])
                cv2.putText(overlay, lbl, (wu - 30, wv + 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(overlay, lbl, (wu - 30, wv + 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 1, cv2.LINE_AA)

        row_hands.append({
            'hand_index': c['h_i'],
            'handedness': hd,
            'joints_3d': joints_3d.round(4).tolist(),
            'joints_2d': uv.round(2).tolist(),
            'kp_scores': [1.0] * 21,
            'kp_valid_depth': valid.tolist(),
            'palm_6d': palm_pose,
        })

    if draw_overlay:
        tag = depth_source_label
        if robot_position_label:
            tag += f' | robot={robot_position_label} -> nearest {frame_corner_label}'
        cv2.putText(overlay, f'MediaPipe + {tag}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, f'MediaPipe + {tag}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

    return row_hands, overlay, total_valid_kps
