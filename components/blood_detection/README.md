# Blood Detection (offline first)

This is separate from Tool and Hand. It uses the supplied `blood_detection.pth`
checkpoint with RF-DETR Seg-Small.

- Explicit model class: `blood`
- Background / `not blood`: implicit background, not a second output class
- Default confidence threshold: `0.5`
- Input: PNG/JPEG frames in `/home/hanwae/blood/pig1/imgs`
- Output: per-frame binary mask, overlay image, `overlay.mp4`, `mask.mp4`, and `blood_results.jsonl` with
  boxes, confidence, centroid, and COCO RLE masks.

First smoke test (five frames, avoids inference compilation):

```bash
source /home/hanwae/surgical_robot/rfdetr_perception_ros/.venv/bin/activate
python ~/surgical_robot/blood_detection/offline_blood_segmentation.py \
  --output-dir ~/surgical_robot/results/blood_smoke_test \
  --max-frames 5 \
  --no-optimize
```

Run all images after that:

```bash
source /home/hanwae/surgical_robot/rfdetr_perception_ros/.venv/bin/activate
python ~/surgical_robot/blood_detection/offline_blood_segmentation.py \
  --output-dir ~/surgical_robot/results/blood_pig1
```

The first full optimized run can take longer because Torch prepares an
inference graph. That setup time is not per-frame inference time.
