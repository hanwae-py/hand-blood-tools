"""Checkpoint loading used by the live RF-DETR + Cutie pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

COMPONENT_ROOT = Path(__file__).resolve().parents[2]
CUTIE_ROOT = COMPONENT_ROOT / "third_party" / "cutie"
PRETRAINED_DIR = COMPONENT_ROOT / "pretrained"


def resolve_rfdetr_checkpoint(path: str | Path | None) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        for name in ("detr_blood.pth", "blood_detr.pth", "checkpoint_best_total.pth", "last.ckpt", "blood_detection_full_all.pth"):
            cand = p / name
            if cand.is_file():
                return cand
            cand = p / "rfdetr" / name
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No RF-DETR checkpoint under {p}")
    default = PRETRAINED_DIR / "detr_blood.pth"
    if default.is_file():
        return default
    raise FileNotFoundError(f"No RF-DETR checkpoint at {default}")


def resolve_cutie_checkpoint(path: str | Path | None) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        matches = list(p.glob("*_last.pth")) + list(p.glob("**/*_last.pth"))
        if matches:
            return matches[0]
        for name in ("cutie_blood.pth", "blood_cutie.pth", "cutie_blood_full_all.pth"):
            named = p / name
            if named.is_file():
                return named
        raise FileNotFoundError(f"No Cutie checkpoint under {p}")
    default = PRETRAINED_DIR / "cutie_blood.pth"
    if default.is_file():
        return default
    raise FileNotFoundError(f"No Cutie checkpoint at {default}")


def union_detections(dets, score_thr: float) -> tuple[np.ndarray | None, list[tuple[np.ndarray, float]]]:
    if dets is None or dets.mask is None or len(dets) == 0:
        return None, []
    keep = dets.confidence > score_thr
    instances = []
    for mask, score, ok in zip(dets.mask, dets.confidence, keep):
        if ok:
            instances.append((mask.astype(bool), float(score)))
    if not instances:
        h, w = dets.mask.shape[1:]
        return np.zeros((h, w), dtype=bool), []
    union = np.zeros_like(instances[0][0], dtype=bool)
    for mask, _ in instances:
        union |= mask
    return union, instances


def load_rfdetr(ckpt: Path, device: str = "cuda"):
    from rfdetr import RFDETR

    return RFDETR.from_checkpoint(str(ckpt), trust_checkpoint=True)


def load_cutie(weights_path: str | Path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from cutie.model.cutie import CUTIE
    import torch

    GlobalHydra.instance().clear()
    initialize_config_dir(
        version_base="1.3.2",
        config_dir=str(CUTIE_ROOT / "cutie" / "config"),
        job_name="cutie_eval",
    )
    cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg.weights = str(weights_path)
        cfg.use_long_term = False
        cfg.mem_every = 5
        cfg.max_internal_size = 480
    get_dataset_cfg(cfg)
    model = CUTIE(cfg).cuda().eval()
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "weights" in state:
        state = state["weights"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_weights(state)
    return model, cfg
