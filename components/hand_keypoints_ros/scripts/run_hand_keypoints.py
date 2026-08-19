"""Hand keypoints + palm 6D pipeline — MediaPipe (GPU) 2D detector, with
automatic depth source selection:

  1. If a real depth stream (aligned, per-pixel, HDF5 uint16 mm) is given
     or auto-detected next to the RGB clip -> use it directly.
  2. Otherwise -> fall back to monocular depth estimation
     (Depth-Anything V2, metric-indoor) run on the RGB frame itself.

Both paths feed the same downstream math: backprojection through camera
intrinsics -> metric 3D joints -> palm 6D pose (v2 formula). Output JSON
records which depth source was actually used for each run.

This is a thin CLI wrapper (video file I/O + argparse) around the
per-frame logic in hand_keypoints_core.py — the same module the ROS2
node uses, so both stay in sync automatically.

Camera intrinsics come from a calibration JSON with the structure written
by this lab's capture rig:
    { "camera_info": { "/synced/<cam_key>/color/camera_info": { "k": [fx,0,cx, 0,fy,cy, 0,0,1] } } }

Usage — real depth auto-detected (looks for a sibling depth_raw/*.h5
   next to the rgb/ clip, same filename pattern with "_rgb_" ->
   "_depth_raw_"):
    python run_hand_keypoints.py \
        --rgb   data/rgb/gnu_0704_rgb_03.avi \
        --calib data/calibration/gnu_0704_calibration_03.json \
        --out   results/example_output

Usage — force monocular depth even if a real depth file exists:
    python run_hand_keypoints.py --rgb ... --calib ... --out ... --force-mono-depth

Usage — explicit depth file (skips auto-detection):
    python run_hand_keypoints.py --rgb ... --calib ... --depth path/to/depth_raw.h5 --out ...

Usage — CPU only, no GPU at all:
    python run_hand_keypoints.py --rgb ... --calib ... --out ... --cpu-only

Usage — robot handoff mode (extra, opt-in; default behaviour above is
   untouched when this flag is omitted). Keeps only the ONE hand nearest
   to wherever the robot is positioned, per frame, instead of every
   detected hand:
    python run_hand_keypoints.py --rgb ... --calib ... --out ... --robot-position top-left

   NOTE on the corner mapping: this camera is not mirrored like a selfie
   camera (see --flip-handedness above), so a robot standing at physical
   position "top-left" of the room is nearest to the hand that appears
   in the "top-right" CORNER OF THE FRAME, not the frame's top-left. This
   script mirrors left/right (keeps top/bottom as-is) when turning
   --robot-position into a target pixel corner, matching the
   already-validated convention this lab used for its earlier
   top-right-quadrant filter. Only that one case (top-left robot ->
   top-right frame corner) has been empirically confirmed; the other
   three are inferred by the same left/right mirror and should be
   sanity-checked against overlay.mp4 before trusting them operationally.
"""
import argparse
import csv
import json
import os
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import cv2
import numpy as np

from hand_keypoints_core import (
    JOINT_NAMES, DEFAULT_DEPTH_MODEL,
    load_intrinsics, robot_position_target_px, find_depth_h5,
    load_mediapipe, load_mono_depth_model, run_mono_depth, process_frame,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rgb', required=True, help='input RGB video (.avi/.mp4)')
    ap.add_argument('--calib', required=True, help='calibration JSON with camera_info')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--cam-key', default='cam_4', help='camera key inside the calibration JSON')
    ap.add_argument('--depth', default=None,
                     help='real depth HDF5 (dataset "depth", uint16 mm, aligned to RGB). '
                          'If omitted, auto-detected next to --rgb; if not found, falls back to mono depth.')
    ap.add_argument('--force-mono-depth', action='store_true',
                     help='ignore any real depth file and always use Depth-Anything V2')
    ap.add_argument('--depth-model', default=DEFAULT_DEPTH_MODEL,
                     help='HuggingFace model id used for the monocular-depth fallback')
    ap.add_argument('--max-hands', type=int, default=4)
    ap.add_argument('--cpu-only', action='store_true',
                     help='EXTRA, opt-in: force CPU everywhere (MediaPipe CPU delegate, '
                          'mono-depth model on CPU if used). Default (omitted) = try GPU first. '
                          'Real-depth mode has no GPU dependency besides MediaPipe itself.')
    ap.add_argument('--stride', type=int, default=1, help='process every Nth frame')
    ap.add_argument('--flip-handedness', action='store_true',
                     help='swap L/R labels (MediaPipe assumes a selfie/mirrored camera)')
    ap.add_argument('--region-x-min', type=float, default=0.0)
    ap.add_argument('--region-x-max', type=float, default=1.0)
    ap.add_argument('--region-y-min', type=float, default=0.0)
    ap.add_argument('--region-y-max', type=float, default=1.0)
    ap.add_argument('--robot-position', default=None,
                     choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                     help='EXTRA, opt-in: keep only the single hand nearest the robot '
                          'each frame (see module docstring for the corner-mapping caveat). '
                          'Default (omitted) = original behaviour, all hands kept. '
                          'Applied after --region-* filtering.')
    args = ap.parse_args()

    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('--cpu-only set: forcing CPU for MediaPipe and any mono-depth model')

    os.makedirs(args.out, exist_ok=True)
    fx, fy, cx, cy = load_intrinsics(args.calib, args.cam_key)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)
    print(f'intrinsics ({args.cam_key}): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}')

    # ---- depth source selection --------------------------------------
    depth_path = None if args.force_mono_depth else (args.depth or find_depth_h5(args.rgb))
    use_real_depth = depth_path is not None and os.path.isfile(depth_path)
    if use_real_depth:
        print(f'depth source: REAL DEPTH  ({depth_path})')
        depth_source_tag = 'realsense_h5_aligned_mm'
    else:
        print('depth source: MONOCULAR (Depth-Anything V2) -- no real depth file found/given')
        depth_source_tag = f'monocular_{os.path.basename(args.depth_model)}'
    depth_source_label = 'REAL DEPTH' if use_real_depth else 'MONO DEPTH (Depth-Anything V2)'

    mp, hand_det = load_mediapipe(args.max_hands, cpu_only=args.cpu_only)

    depth_h5 = None
    depth_ds = None
    n_depth = 0
    torch = processor = depth_model = device = dtype = None
    if use_real_depth:
        import h5py
        depth_h5 = h5py.File(depth_path, 'r')
        depth_ds_lazy = depth_h5['depth']
        print(f'depth: {depth_ds_lazy.shape} {depth_ds_lazy.dtype}  (preloading to RAM)')
        _t0 = time.perf_counter()
        depth_ds = depth_ds_lazy[:]
        print(f'  preload took {time.perf_counter() - _t0:.1f}s  ({depth_ds.nbytes / 1e9:.2f} GB)')
        n_depth = depth_ds.shape[0]
    else:
        torch, processor, depth_model, device, dtype = load_mono_depth_model(args.depth_model, cpu_only=args.cpu_only)

    cap = cv2.VideoCapture(args.rgb)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'RGB: {args.rgb}  {W}x{H} @ {fps:.2f}fps  frames={n_total}')

    target_px = None
    frame_corner_label = None
    if args.robot_position:
        target_px, frame_corner_label = robot_position_target_px(args.robot_position, W, H)
        print(f'robot handoff mode: robot={args.robot_position}  ->  '
              f'keeping the hand nearest frame corner "{frame_corner_label}" '
              f'(pixel {target_px}) each frame')

    writer = cv2.VideoWriter(os.path.join(args.out, 'overlay.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    csv_path = os.path.join(args.out, 'depth_valid.csv')
    fcsv = open(csv_path, 'w', newline='')
    wr = csv.writer(fcsv)
    wr.writerow(['frame_idx', 't_s', 'n_hands', 'mean_valid_kps_per_hand'])

    frames_out = []
    frame_idx = 0
    n_out = 0
    n_hands = 0
    t_start = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        if use_real_depth and frame_idx >= n_depth:
            print(f'  no depth for frame {frame_idx}, stopping')
            break

        # ---- depth map for this frame, always in METRES ---------------
        if use_real_depth:
            depth_map = depth_ds[frame_idx].astype(np.float32) / 1000.0
        else:
            depth_map = run_mono_depth(frame, torch, processor, depth_model, device, dtype, H, W)

        ts_ms = int(round(frame_idx * 1000 / max(fps, 1)))
        row_hands, overlay, total_valid_kps = process_frame(
            frame, depth_map, hand_det, mp, K, fx, fy, cx, cy, W, H, ts_ms,
            region=(args.region_x_min, args.region_x_max, args.region_y_min, args.region_y_max),
            target_px=target_px, robot_position_label=args.robot_position,
            frame_corner_label=frame_corner_label, flip_handedness=args.flip_handedness,
            draw_overlay=True, depth_source_label=depth_source_label)
        n_hands += len(row_hands)

        frames_out.append({'frame_idx': frame_idx, 't_s': round(frame_idx / max(fps, 1), 3),
                            'hands': row_hands})
        writer.write(overlay)
        wr.writerow([frame_idx, round(frame_idx / max(fps, 1), 3), len(row_hands),
                     round(total_valid_kps / max(len(row_hands), 1), 2)])

        n_out += 1
        frame_idx += 1
        if n_out % 60 == 0:
            elapsed = time.perf_counter() - t_start
            print(f'  {frame_idx}/{n_total}  hands so far: {n_hands}  '
                  f'infer_fps={n_out / max(elapsed, 1e-3):.2f}')

    cap.release()
    writer.release()
    fcsv.close()
    if depth_h5 is not None:
        depth_h5.close()

    elapsed = time.perf_counter() - t_start
    infer_fps = n_out / max(elapsed, 1e-3)
    print(f'\nframes processed: {n_out}   hands: {n_hands}')
    print(f'wall time: {elapsed:.1f}s   infer_fps={infer_fps:.2f}   source={fps:.1f}fps')

    doc = {
        'video': os.path.basename(args.rgb),
        'n_frames': len(frames_out),
        'fps': round(fps, 3),
        'resolution': [W, H],
        'camera_used': args.cam_key,
        'camera_intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
        'coordinate_frame': f'{args.cam_key} optical frame. Units: metres.',
        'depth_source': depth_source_tag,
        'robot_handoff_mode': None if not args.robot_position else {
            'robot_position': args.robot_position,
            'target_frame_corner': frame_corner_label,
            'target_pixel': list(target_px),
            'selection': 'single nearest hand per frame (Euclidean, 2D pixel centroid)',
        },
        'joint_names': JOINT_NAMES,
        'palm_6d_notes': {
            'formula_version': 'v2 (2026-07-21 revision)',
            'translation': 'midpoint(wrist, middle_MCP), metres, camera frame',
            'rotation': 'palm coord frame R (3x3). Columns are '
                        '(X: wrist->middle_MCP, Y: across palm, Z: palm normal).',
            'quat_convention': '(w, x, y, z), Hamilton',
        },
        'perf': {'infer_fps': round(infer_fps, 2), 'source_fps': round(fps, 2),
                  'wall_time_s': round(elapsed, 1)},
        'frames': frames_out,
    }
    out_json = os.path.join(args.out, 'keypoints.json')
    with open(out_json, 'w') as f:
        json.dump(doc, f)
    print(f'wrote:\n  {out_json}\n  {os.path.join(args.out, "overlay.mp4")}\n  {csv_path}')


if __name__ == '__main__':
    main()
