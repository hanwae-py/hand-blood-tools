# Blood Detection (offline first)

This is separate from Tool and Hand. It uses the supplied `blood_detection.pth`
checkpoint with RF-DETR Seg-Small.

- Explicit model class: `blood`
- Background / `not blood`: implicit background, not a second output class
- Default confidence threshold: `0.5`
- Input: PNG/JPEG frames in `$HOME/data/blood/imgs`
- Output: per-frame binary mask, overlay image, `overlay.mp4`, `mask.mp4`, and `blood_results.jsonl` with
  boxes, confidence, centroid, and COCO RLE masks.

ROS Blood uses `scripts/run_blood_cam4.sh` (local `/perception/ingress/cam_4`
after `scripts/run_perception_ingress.sh`) and `BLOOD_CHECKPOINT` from
`config/system.env`. For this offline script, point `--checkpoint` at the same
`.pth`.

First smoke test (five frames, avoids inference compilation):

```bash
"${RFDETR_PYTHON}" components/blood_detection/offline_blood_segmentation.py \
  --checkpoint "$HOME/models/blood_detection.pth" \
  --images-dir "$HOME/data/blood/imgs" \
  --output-dir "$HOME/results/blood_smoke_test" \
  --max-frames 5 \
  --no-optimize
```

Run all images after that:

```bash
"${RFDETR_PYTHON}" components/blood_detection/offline_blood_segmentation.py \
  --checkpoint "$HOME/models/blood_detection.pth" \
  --images-dir "$HOME/data/blood/imgs" \
  --output-dir "$HOME/results/blood_pig1"
```

The first full optimized run can take longer because Torch prepares an
inference graph. That setup time is not per-frame inference time.
