"""Training loop shared by all learned methods (P1 benchmark uses an L1 loss only)."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import WaldDataset
from .metrics import psnr
from .models import build_model, count_params

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


class _StopRequest:
    """SIGTERM/SIGINT in the main training process ask for a checkpoint and a clean exit (scheduled pauses)."""

    def __init__(self):
        self.flag = False
        self.pid = os.getpid()
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        if os.getpid() == self.pid:     # DataLoader workers inherit the handler: ignore it there
            self.flag = True


def config_fingerprint(spec: dict, cfg: dict, mtf: dict, patch_dir: Path, seed: int) -> str:
    """Identifies method, recipe, MTF table, seed and training data (size, mtime and patch count of the shards)."""
    import hashlib

    import h5py

    shards = []
    for p in (Path(patch_dir) / "train.h5", Path(patch_dir) / "val.h5"):
        if p.exists():
            with h5py.File(p, "r") as f:
                shards.append((p.name, p.stat().st_size, int(p.stat().st_mtime), int(f["x20"].shape[0])))
    blob = json.dumps([spec, cfg, mtf, seed, shards], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def masked_l1(sr, tgt, valid, per_sample: bool = False):
    """L1 over SCL-valid pixels (patches keep up to 5 % invalid pixels)."""
    m = valid.unsqueeze(1).to(sr.dtype).expand_as(sr)
    if per_sample:
        return ((sr - tgt).abs() * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)
    return ((sr - tgt).abs() * m).sum() / m.sum().clamp_min(1)


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    l1, ps, n = 0.0, 0.0, 0
    for b in loader:
        lr, guide, tgt = b["lr"].to(device), b["guide"].to(device), b["target"].to(device)
        v = b["valid"].to(device)
        sr = model(lr, guide)
        l1 += masked_l1(sr, tgt, v, per_sample=True).sum().item()
        ps += psnr(sr, tgt, mask=v).sum().item()
        n += tgt.shape[0]
    model.train()
    return dict(val_l1=l1 / n, val_psnr=ps / n)


def train(method: str, spec: dict, cfg: dict, mtf: dict, patch_dir: Path, run_dir: Path,
          seed: int) -> Path:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(spec, mtf).to(device)
    if not getattr(model, "trainable", False):
        raise ValueError(f"{method} is not trainable")
    n_params = count_params(model)
    (run_dir / "config.json").write_text(json.dumps(
        dict(method=method, spec=spec, train=cfg, seed=seed, n_params=n_params), indent=2))
    log.info("%s seed %d: %.2f M parameters", method, seed, n_params / 1e6)

    tr = WaldDataset(patch_dir / "train.h5", mtf, augment=True)
    va = WaldDataset(patch_dir / "val.h5", mtf, augment=False)
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                    drop_last=True, generator=g, persistent_workers=cfg["workers"] > 0)
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["workers"])

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")
    best, bad, step = math.inf, 0, 0
    last = run_dir / "last.pt"
    fingerprint = config_fingerprint(spec, cfg, mtf, patch_dir, seed)
    if (run_dir / "DONE").exists():
        done = (run_dir / "DONE").read_text()
        if f"fingerprint {fingerprint}" not in done:
            raise SystemExit(f"{run_dir}/DONE belongs to another configuration or data; move {run_dir} away")
        log.info("%s seed %d already finished", method, seed)
        return run_dir / "best.pt"
    if last.exists():  # resume after an interruption (data order after resuming is not reproduced)
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck.get("fingerprint") != fingerprint:
            raise SystemExit(f"{last} was written with another configuration or data; move {run_dir} away "
                             "to start from scratch")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        best, bad, step = ck["best"], ck["bad"], ck["step"]
        meta = run_dir / "best.json"     # best.pt may be newer than last.pt if interrupted between the two saves
        if meta.exists():
            best = min(best, json.loads(meta.read_text())["val_l1"])
        torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        log.info("resumed at step %d (best val L1 %.6f)", step, best)
    log_f = open(run_dir / "log.csv", "a" if last.exists() else "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=["step", "lr", "train_l1", "val_l1", "val_psnr", "sec"])
    if not last.exists():
        writer.writeheader()
    t0, run_l1 = time.time(), 0.0
    stop = _StopRequest()

    def save_last():
        tmp = run_dir / "last.pt.tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        best=best, bad=bad, step=step, rng=torch.get_rng_state(), fingerprint=fingerprint), tmp)
        tmp.replace(last)
    done = step >= cfg["iterations"] or bad >= cfg["patience"]
    while not done:
        for b in tl:
            lr_now = cosine_lr(step, cfg["iterations"], cfg["lr"], cfg["warmup"])
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            lr_in, guide, tgt = b["lr"].to(device), b["guide"].to(device), b["target"].to(device)
            with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
                out = model(lr_in, guide)
            loss = masked_l1(out.float(), tgt, b["valid"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_l1 += loss.item()
            step += 1
            if step % cfg["val_every"] == 0:
                v = validate(model, vl, device)
                writer.writerow(dict(step=step, lr=lr_now, train_l1=run_l1 / cfg["val_every"],
                                     sec=round(time.time() - t0), **v))
                log_f.flush()
                run_l1 = 0.0
                if v["val_l1"] < best:
                    best, bad = v["val_l1"], 0
                    tmp_best = run_dir / "best.pt.tmp"
                    torch.save(model.state_dict(), tmp_best)
                    tmp_best.replace(run_dir / "best.pt")
                    (run_dir / "best.json.tmp").write_text(json.dumps(dict(val_l1=best, step=step)))
                    (run_dir / "best.json.tmp").replace(run_dir / "best.json")
                else:
                    bad += 1
                if bad >= cfg["patience"]:
                    log.info("early stop at step %d", step)
                    done = True
                save_last()
            if step >= cfg["iterations"]:
                done = True
            if stop.flag and not done:   # paused: keep the progress since the last validation and exit 143
                save_last()
                log_f.close()
                log.info("%s seed %d: stopped at step %d on request; checkpoint saved", method, seed, step)
                sys.exit(143)
            if done:
                break
    log_f.close()
    if not math.isfinite(best) or not (run_dir / "best.pt").exists():
        raise SystemExit(f"{method} seed {seed}: no finite validation loss, no usable best.pt")
    (run_dir / "DONE").write_text(f"step {step}\nbest_val_l1 {best}\nfingerprint {fingerprint}\n")
    return run_dir / "best.pt"
