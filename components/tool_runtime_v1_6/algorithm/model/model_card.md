# CAM4 RF-DETRSegSmall regular-resume epoch-13 model card

## Identification

- model version: `cam4-rfdetr-seg-small-regular-resume-e13-best`
- architecture: `RFDETRSegSmall`
- RF-DETR version: `1.8.3`
- frozen class count: 8
- checkpoint: `cam4_rfdetr_seg_small_regular_resume_e13_best.pth`
- SHA-256: `253617aa5337fec219d694ca50537e4867fb8c403ce60f3a6945bbe15fecf430`
- source checkpoint: `checkpoint_best_regular.pth`
- epoch / global step: 13 / 1274
- input color contract: OpenCV BGR

## Runtime defaults

```text
confidence_threshold = 0.30
enable_class_agnostic_nms = true
class_agnostic_nms_iou = 0.80
```

NMS removes overlapping bbox candidates in descending confidence order without using class IDs. Set
`enable_class_agnostic_nms=false` to receive raw thresholded predictions.

## Training provenance

- dataset: `cam4_round4_r1mask_sam2refined_v1`
- original run: `20260820_r1mask_sam2refined_round5_full_e20`
- resumed run: `20260820_round5_resume_full_to_e20_noearlystop`
- resume source: original Round5 `last.ckpt`
- mask policy: rigid tool only; Bovie electrode tip included; cable and hand excluded
- validation: reviewed case-disjoint development cases, not an independent sealed final test

## Unlabeled screening

Reference rosbag CAM4 RGB, 447 frames, confidence 0.30:

- raw: 428 detection frames, 1,067 instances
- class-agnostic NMS 0.80: 428 detection frames, 956 instances

All 17 CAM4 0704 cases, 1,020 sampled frames, confidence 0.30 and NMS 0.80:

- 699 frames with detections
- 2,138 masks

These counts are not accuracy, precision, recall, IoU, or mAP. Compared with the previous Round5 best,
the reference-bag output is more conservative and its class distribution changes substantially. The
selected checkpoint was explicitly chosen for this delivery, but labeled acceptance remains required.

## Limitations

- Human-GT class and mask acceptance is pending.
- Overlapping distinct tools can be incorrectly suppressed by bbox NMS.
- NMS does not remove isolated person/background false positives; a tray/Mayo ROI and size filtering are
  still recommended.
- Do not use this output directly for clinical decisions or robot commands without validity checks and a
  downstream safety policy.
