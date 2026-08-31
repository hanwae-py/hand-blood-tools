# Blood Detection (RF-DETR + Cutie)

Fused **RF-DETR Seg-Small + Cutie** overlay (`BloodPipeline`). Do not ship
detector-only masks. This component is isolated from Tool: it uses its own
Python interpreter (`BLOOD_PYTHON`), not `RFDETR_PYTHON`.

- Explicit model class: `blood`
- Background / `not blood`: implicit background
- Detector: RF-DETR SegSmall, resolution 384, score threshold `0.5`
- Tracker: Cutie, `max_internal_size` 480
- Re-detect: every processed frame (`redetect_interval: 1`)
- Target: **10 Hz** end-to-end at 720×1280. **8–9 Hz is acceptable.** A ~30 Hz
  camera is dropped to latest-frame, not processed frame-for-frame.
- Weights (gitignored `.pth`): `pretrained/detr_blood.pth`,
  `pretrained/cutie_blood.pth`

ROS Blood uses `scripts/run_blood_cam4.sh`. The integrated stack starts
**FLIR** (`bash scripts/run_blood_cam4.sh flir`). `cam_4` is for a single-worker
debug session after `scripts/run_perception_ingress.sh`.

## Third-party trees (clone)

This repo ships an empty `third_party/` folder only. `rfdetr` and `cutie` are
not committed. From the repository root, create the folder if needed and clone
into it:

```bash
mkdir -p components/blood_detection/third_party
git clone https://github.com/roboflow/rf-detr.git components/blood_detection/third_party/rfdetr
git clone https://github.com/hkchengrex/Cutie.git components/blood_detection/third_party/cutie
```

Deploy tree: RF-DETR **1.10.0.dev**. Exact commit/tag: **TBD** (match the
BloodDetection deploy tree). Run this before `setup_env.sh`.

## Pretrained weights (Google Drive)

Copy into `components/blood_detection/pretrained/` or `$HOME/models/` and set
`BLOOD_CHECKPOINT` / `BLOOD_CUTIE_CHECKPOINT` in `config/system.env`.

| File | Drive |
|---|---|
| `detr_blood.pth` | **TBD** |
| `cutie_blood.pth` | **TBD** |

The old single-file RF-DETR-only checkpoint is not the live overlay.

Training, datasets, and evaluation stay in the BloodDetection repository.
This tree is inference and ROS only.

## Environment

Do not combine this env with Hand (`torch==2.11.0`) or Tool (`rfdetr==1.8.3`).
Use Python 3.12 so ROS 2 Jazzy `rclpy` can import.

Conda:

```bash
conda env create -f components/blood_detection/environment.yml
conda activate blood
bash components/blood_detection/setup_env.sh
```

If a BloodDetection training env already uses the name `blood`, create
`blood-ros` instead and point `BLOOD_PYTHON` at that interpreter.

venv:

```bash
python3.12 -m venv .venv-blood
source .venv-blood/bin/activate
bash components/blood_detection/setup_env.sh
deactivate
```

Point `BLOOD_PYTHON` at `$(conda info --base)/envs/blood/bin/python` or
`$PWD/.venv-blood/bin/python`. Export `PYTHONNOUSERSITE=1` (the run scripts
already do this).

## Offline smoke

```bash
"${BLOOD_PYTHON}" components/blood_detection/offline_blood_segmentation.py \
  --checkpoint "$HOME/models/detr_blood.pth" \
  --cutie-checkpoint "$HOME/models/cutie_blood.pth" \
  --images-dir "$HOME/data/blood/imgs" \
  --output-dir "$HOME/results/blood_smoke_test" \
  --max-frames 5 \
  --no-video
```

## Isolation

Tool, Hand, ingress, overlay, and `run.sh` are unchanged. Only
`scripts/run_blood_cam4.sh` prepends `components/blood_detection` onto
`PYTHONPATH` for the Blood process. GPU memory is still shared if Tool/Hand
run on the same card.
