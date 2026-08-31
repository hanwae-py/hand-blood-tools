"""Shared hand-keypoints + palm-6D detection logic.

This module has NO argparse / file-loop / ROS code — it's the single
source of truth for the per-frame math, used by both:
  - scripts/run_hand_keypoints.py   (offline CLI, reads video files)
  - the ROS2 hand-detection node    (reads live/bag camera topics)

Keeping this logic in one place means both consumers stay in sync
automatically instead of drifting apart as separate copies.
"""
import contextlib
import hashlib
import json
import os
import sys

import cv2
import numpy as np

from hand_keypoint_ros.topview_gesture import classify as classify_topview_gesture

MEDIAPIPE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/latest/hand_landmarker.task')
GESTURE_RECOGNIZER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/'
    'gesture_recognizer/float16/1/gesture_recognizer.task')
GESTURE_RECOGNIZER_MODEL_VERSION = 'float16/1'
GESTURE_RECOGNIZER_MODEL_SHA256 = (
    '97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482')
MEDIAPIPE_CACHE = os.path.expanduser('~/.cache/mediapipe')
DEFAULT_GESTURE_RECOGNIZER_MODEL = os.path.join(
    MEDIAPIPE_CACHE, 'gesture_recognizer.task')
DEFAULT_DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'

# MediaPipe's canned classifier has seven named gestures plus a rejection
# category. Keep this order stable on the ROS wire even if MediaPipe happens
# to return its Category list in a different order.
CANNED_GESTURE_NAMES = (
    'Closed_Fist',
    'Open_Palm',
    'Pointing_Up',
    'Thumb_Down',
    'Thumb_Up',
    'Victory',
    'ILoveYou',
)
GESTURE_OUTPUT_NAMES = ('None',) + CANNED_GESTURE_NAMES

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
    med = np.zeros(len(patches), dtype=np.float32)
    valid = np.any(valid_2d, axis=1)
    if np.any(valid):
        patches_nan = np.where(valid_2d[valid], patches[valid], np.nan)
        med[valid] = np.nanmedian(patches_nan, axis=1)
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_gesture_recognizer_model(model_path=None):
    """Return the pinned official Gesture Recognizer asset path.

    The task bundle includes the hand detector, hand landmarks and canned
    gesture classifier. A digest check prevents silently running a different
    or partially downloaded model while publishing the official-v1 contract.
    """
    path = os.path.expanduser(
        model_path or DEFAULT_GESTURE_RECOGNIZER_MODEL)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(path):
        import urllib.request
        partial = path + '.partial'
        print(f'downloading {GESTURE_RECOGNIZER_MODEL_URL} ...')
        try:
            urllib.request.urlretrieve(
                GESTURE_RECOGNIZER_MODEL_URL, partial)
            partial_digest = _sha256_file(partial)
            if partial_digest != GESTURE_RECOGNIZER_MODEL_SHA256:
                raise RuntimeError(
                    'unexpected downloaded MediaPipe Gesture Recognizer '
                    'asset digest: '
                    f'expected={GESTURE_RECOGNIZER_MODEL_SHA256} '
                    f'actual={partial_digest} path={partial}')
            os.replace(partial, path)
        finally:
            if os.path.exists(partial):
                os.unlink(partial)
    actual = _sha256_file(path)
    if actual != GESTURE_RECOGNIZER_MODEL_SHA256:
        raise RuntimeError(
            'unexpected MediaPipe Gesture Recognizer asset digest: '
            f'expected={GESTURE_RECOGNIZER_MODEL_SHA256} actual={actual} '
            f'path={path}')
    return path


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


def load_gesture_recognizer(max_hands, model_path=None, cpu_only=False):
    """Load MediaPipe's official canned Gesture Recognizer in VIDEO mode.

    The recognizer returns landmarks, handedness and gestures from one graph,
    so callers do not need to run a second HandLandmarker and then guess how
    two independent per-frame hand indices correspond.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe.tasks.python.components.processors import classifier_options

    asset_path = ensure_gesture_recognizer_model(model_path)
    canned_options = classifier_options.ClassifierOptions(
        max_results=-1,
        score_threshold=0.0,
    )
    common_kwargs = dict(
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        canned_gesture_classifier_options=canned_options,
    )
    if cpu_only:
        base_opts = BaseOptions(
            model_asset_path=asset_path,
            delegate=BaseOptions.Delegate.CPU,
        )
        recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=base_opts, **common_kwargs))
        print('MediaPipe gesture recognizer ready (CPU, forced by --cpu-only)')
        return mp, recognizer
    try:
        base_opts = BaseOptions(
            model_asset_path=asset_path,
            delegate=BaseOptions.Delegate.GPU,
        )
        recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=base_opts, **common_kwargs))
        print('MediaPipe gesture recognizer ready (GPU delegate)')
    except Exception as exc:
        print(f'GPU delegate failed ({exc}) -- falling back to CPU')
        base_opts = BaseOptions(
            model_asset_path=asset_path,
            delegate=BaseOptions.Delegate.CPU,
        )
        recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=base_opts, **common_kwargs))
        print('MediaPipe gesture recognizer ready (CPU)')
    return mp, recognizer


def normalize_canned_gesture(categories):
    """Normalize the public API's winning canned-gesture Category.

    MediaPipe 0.10.18's combined-prediction graph exposes only its winning
    label, even when the underlying classifier is configured with
    ``max_results=-1``. Do not fabricate an unavailable full score vector.
    """
    candidates = [
        category for category in (categories or ())
        if str(getattr(category, 'category_name', '') or '')
        in GESTURE_OUTPUT_NAMES
    ]
    if not candidates:
        return {
            'has_gesture': False,
            'category_name': '',
            'score': 0.0,
        }
    winner = max(candidates, key=lambda category: float(category.score))
    return {
        'has_gesture': True,
        'category_name': str(winner.category_name),
        'score': float(winner.score),
    }


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
    """Silence MediaPipe's C++ stderr for one landmarker/recognizer call.

    With the GPU delegate, MediaPipe 0.10.x logs
        tensor.cc:410] Tensors are designed for single writes...
    at ERROR level several times PER FRAME. It's spurious here -- the
    landmarker is only ever driven from a single thread (the rclpy spin
    thread, or the offline CLI's frame loop) -- but at 15+ Hz it buries
    every other line of output.

    It has to be suppressed at the file-descriptor level: MediaPipe 0.10.18
    logs via absl, not glog, so the usual GLOG_minloglevel/GLOG_logtostderr
    env vars have no effect on it (verified -- the spam survives them).

    Scoped to the MediaPipe call only, so Python-side logging (ROS loggers,
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


def inference_crop_box(frame_shape, region, margin=0.0):
    """Return a clipped pixel ROI for inference, or ``None`` for full-frame.

    ``region`` remains the public hand-centroid keep box.  The optional margin
    preserves detector context around that box; results are remapped to the
    original image grid before any downstream filtering or publication.
    """
    height, width = (int(frame_shape[0]), int(frame_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError('frame dimensions must be positive')
    if region is None:
        return None
    x_min, x_max, y_min, y_max = (float(value) for value in region)
    margin = max(0.0, float(margin))
    x_min = max(0.0, x_min - margin)
    x_max = min(1.0, x_max + margin)
    y_min = max(0.0, y_min - margin)
    y_max = min(1.0, y_max + margin)
    if not (x_min < x_max and y_min < y_max):
        raise ValueError('inference ROI must have positive area')
    x0 = max(0, min(width - 1, int(np.floor(x_min * width))))
    x1 = max(x0 + 1, min(width, int(np.ceil(x_max * width))))
    y0 = max(0, min(height - 1, int(np.floor(y_min * height))))
    y1 = max(y0 + 1, min(height, int(np.ceil(y_max * height))))
    if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
        return None
    return x0, y0, x1, y1


def remap_result_landmarks(result, crop_box, frame_shape):
    """Map MediaPipe normalized crop landmarks back to the source frame."""
    if crop_box is None:
        return result
    height, width = (int(frame_shape[0]), int(frame_shape[1]))
    x0, y0, x1, y1 = crop_box
    scale_x = (x1 - x0) / float(width)
    scale_y = (y1 - y0) / float(height)
    offset_x = x0 / float(width)
    offset_y = y0 / float(height)
    for landmarks in getattr(result, 'hand_landmarks', None) or ():
        for landmark in landmarks:
            landmark.x = offset_x + float(landmark.x) * scale_x
            landmark.y = offset_y + float(landmark.y) * scale_y
    return result


def recognize_frame(
    frame_bgr, hand_det, mp, ts_ms, *, inference_region=None,
    inference_margin=0.0,
):
    """Run one MediaPipe VIDEO task, optionally on a padded selection ROI."""
    crop_box = inference_crop_box(
        frame_bgr.shape, inference_region, inference_margin)
    inference_bgr = (
        frame_bgr
        if crop_box is None
        else np.ascontiguousarray(
            frame_bgr[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])
    )
    mp_rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_rgb)
    with _quiet_native_stderr():
        if hasattr(hand_det, 'recognize_for_video'):
            result = hand_det.recognize_for_video(mp_img, ts_ms)
        else:
            result = hand_det.detect_for_video(mp_img, ts_ms)
    return remap_result_landmarks(result, crop_box, frame_bgr.shape)


def world_landmarks_for_hand(result, hand_index):
    """Return one finite MediaPipe world skeleton as 21x3, else ``None``.

    The Tasks API may omit world landmarks on legacy HandLandmarker results.
    Keeping this extraction index-safe prevents one malformed hand from
    shifting the world skeleton attached to another hand in the same frame.
    """
    world_hands = getattr(result, 'hand_world_landmarks', None) or []
    if hand_index < 0 or hand_index >= len(world_hands):
        return None
    try:
        points = np.asarray([
            (landmark.x, landmark.y, landmark.z)
            for landmark in world_hands[hand_index]
        ], dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return None
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return None
    return points


def _palm_facing_exception(reason):
    """Fail one additive facing observation closed without dropping a hand."""
    return {
        'has_facing': False,
        'label': 'UNKNOWN',
        'palm_up_score': 0.0,
        'normal_cam': [0.0, 0.0, 0.0],
        'plane_residual_m': 0.0,
        'support_height_m': 0.0,
        'valid_depth_points': 0,
        'quality_valid': False,
        'rejection_reason': str(reason),
        'calibration_version': '',
    }


def _effective_handedness(result_handedness, hand_index,
                          flip_handedness=False,
                          forced_handedness_label=''):
    """Return the handedness policy result for one detected hand.

    A forced label represents an explicit camera-view constraint, not a
    MediaPipe classification.  Its score is therefore the deterministic
    policy confidence.  The ROS node exposes the active policy in health and
    diagnostics so this override is not mistaken for raw model evidence.
    """
    forced_label = str(forced_handedness_label or '').strip()
    if forced_label:
        if forced_label not in ('Left', 'Right'):
            raise ValueError(
                'forced_handedness_label must be empty, Left, or Right')
        return {'label': forced_label, 'score': 1.0}

    if hand_index >= len(result_handedness) or not result_handedness[hand_index]:
        return None
    category = result_handedness[hand_index][0]
    name = category.category_name
    if flip_handedness:
        name = 'Right' if name == 'Left' else 'Left'
    return {'label': name, 'score': float(category.score)}


def gesture_rows_from_result(result, W, H, region=(0.0, 1.0, 0.0, 1.0),
                             target_px=None, flip_handedness=False,
                             forced_handedness_label='',
                             gesture_profile='topview'):
    """Extract depth-independent gesture rows from one MediaPipe result.

    The returned ``hand_index`` is the frame-local index shared by the same
    result's landmarks. ROI and optional nearest-target filtering mirror the
    keypoint path without performing depth lookup or palm-pose computation.
    """
    x_min, x_max, y_min, y_max = region
    hands = result.hand_landmarks or []
    handedness = result.handedness or []
    candidates = []
    for hand_index, landmarks in enumerate(hands):
        uv = np.array(
            [(landmark.x * W, landmark.y * H) for landmark in landmarks],
            dtype=np.float32,
        )
        centroid = (float(np.mean(uv[:, 0])), float(np.mean(uv[:, 1])))
        normalized_x = centroid[0] / W
        normalized_y = centroid[1] / H
        if not (
            x_min <= normalized_x <= x_max
            and y_min <= normalized_y <= y_max
        ):
            continue

        hand_label = _effective_handedness(
            handedness,
            hand_index,
            flip_handedness=flip_handedness,
            forced_handedness_label=forced_handedness_label,
        )
        gesture = classify_topview_gesture(
            uv,
            world_landmarks_for_hand(result, hand_index),
            profile=gesture_profile,
        )
        candidates.append({
            'hand_index': hand_index,
            'handedness': hand_label,
            'gesture': gesture,
            'centroid_px': centroid,
        })

    if target_px is not None and candidates:
        target_x, target_y = target_px
        candidates = [min(
            candidates,
            key=lambda candidate: (
                (candidate['centroid_px'][0] - target_x) ** 2
                + (candidate['centroid_px'][1] - target_y) ** 2
            ),
        )]

    return [
        {
            'hand_index': candidate['hand_index'],
            'handedness': candidate['handedness'],
            'gesture': candidate['gesture'],
        }
        for candidate in candidates
    ]


def process_frame(frame_bgr, depth_map_m, hand_det, mp, K, fx, fy, cx, cy, W, H,
                   ts_ms, region=(0.0, 1.0, 0.0, 1.0),
                   target_px=None, robot_position_label=None, frame_corner_label=None,
                   flip_handedness=False, draw_overlay=True, depth_source_label='REAL DEPTH',
                   allow_2d_only=False, recognition_result=None,
                   palm_facing_estimator=None, palm_facing_filter=None,
                   forced_handedness_label='', gesture_profile='topview'):
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
    recognition_result: optional result already produced by ``recognize_frame``
        for this exact RGB frame. This lets the ROS node publish an RGB-only
        gesture topic immediately and later reuse the same result for the
        synchronized depth/keypoint path without duplicate inference.

    Returns (row_hands, overlay_frame, total_valid_kps) — row_hands is a
    list of dicts: hand_index, handedness, joints_3d, joints_2d,
    kp_scores, kp_valid_depth, palm_6d, gesture. ``gesture`` is always derived
    from the returned 21 landmarks by the selected VIPLab landmark profile;
    the official canned result, when present, is intentionally ignored.
    ``palm_facing_estimator`` is an optional callable that consumes registered
    metric joints, depth validity, and handedness. It is deliberately absent
    from the RGB-only gesture path.
    """
    x_min, x_max, y_min, y_max = region

    result = recognition_result
    if result is None:
        result = recognize_frame(frame_bgr, hand_det, mp, ts_ms)
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

        hd = _effective_handedness(
            handedness,
            h_i,
            flip_handedness=flip_handedness,
            forced_handedness_label=forced_handedness_label,
        )

        gesture = classify_topview_gesture(
            uv,
            world_landmarks_for_hand(result, h_i),
            profile=gesture_profile,
        )

        palm_facing = None
        if palm_facing_estimator is not None:
            try:
                palm_facing = palm_facing_estimator(joints_3d, valid, hd)
            except Exception as exc:  # additive feature must not drop keypoints
                palm_facing = _palm_facing_exception(
                    f'estimator_exception:{type(exc).__name__}')
            if palm_facing_filter is not None:
                try:
                    palm_facing = palm_facing_filter(
                        palm_facing,
                        centroid_norm=(cu, cv_),
                        handedness=hd,
                    )
                except Exception as exc:  # preserve established outputs
                    palm_facing = _palm_facing_exception(
                        f'temporal_filter_exception:{type(exc).__name__}')

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
            'hd': hd, 'gesture': gesture,
            'palm_facing': palm_facing,
            'palm_pose': palm_pose, 'gizmo': gizmo,
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
        hd, gesture, palm_facing, palm_pose = (
            c['hd'], c['gesture'], c['palm_facing'], c['palm_pose'])

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
            if gesture is not None and gesture['has_gesture']:
                gesture_label = (
                    f'{gesture["category_name"]} {gesture["score"]:.2f}')
                wu, wv = int(uv[0][0]), int(uv[0][1])
                cv2.putText(
                    overlay, gesture_label, (wu - 30, wv + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(
                    overlay, gesture_label, (wu - 30, wv + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 200, 0), 1, cv2.LINE_AA)
            if palm_facing is not None:
                facing_label = (
                    f'{palm_facing["label"]} '
                    f'{float(palm_facing["palm_up_score"]):+.2f}'
                    if palm_facing.get('has_facing', False)
                    else 'PALM_UNKNOWN'
                )
                wu, wv = int(uv[0][0]), int(uv[0][1])
                cv2.putText(
                    overlay, facing_label, (wu - 30, wv + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(
                    overlay, facing_label, (wu - 30, wv + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 255, 180), 1, cv2.LINE_AA)

        row_hands.append({
            'hand_index': c['h_i'],
            'handedness': hd,
            'joints_3d': joints_3d.round(4).tolist(),
            'joints_2d': uv.round(2).tolist(),
            'kp_scores': [1.0] * 21,
            'kp_valid_depth': valid.tolist(),
            'palm_6d': palm_pose,
            'palm_facing': palm_facing,
            'gesture': gesture,
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
