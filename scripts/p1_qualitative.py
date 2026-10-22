"""Qualitative figure of the P1 paper: one SEN2NEON tile, bands B8A and B11.

Tile rule (fixed before viewing): candidates are SEN2NEON tiles ordered by how close their DSen2 - RCAN-G HP-corr
difference (seed 0, P3) is to the median over tiles (a typical tile, not a favourable one); the first candidate
with at least 90 % valid pixels is used. The crop is the 64 x 64 px (640 m) window with the largest high-pass
energy of the reference. Each panel is stretched to its own 2-98 % range, because the SEN2NEON reference is
harmonised with a lower contrast than Sentinel-2 and HP-corr, the primary metric, ignores the gain.

  python scripts/p1_qualitative.py --candidates candidates.txt --out fig_qualitative.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s2sr.bands import target_index  # noqa: E402
from s2sr.degrade import upsample  # noqa: E402
from s2sr.models import build_model  # noqa: E402

COLS = [("input", "Sentinel-2 20 m"), ("bicubic", "Bicubic"), ("rcan_g", "RCAN-G"), ("dsen2", "DSen2"),
        ("dsen2_global", "DSen2-global"), ("ref", "Reference (10 m)")]
BANDS = ["B8A", "B11"]
CROP = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--refs", default="data/p1_refs/sen2neon")
    ap.add_argument("--out", default="fig_qualitative.pdf")
    a = ap.parse_args()
    mtf = yaml.safe_load((ROOT / "configs/mtf.yaml").read_text())["mtf_nyquist"]
    specs = yaml.safe_load((ROOT / "configs/methods.yaml").read_text())["methods"]
    root = ROOT / a.refs
    for key in Path(a.candidates).read_text().split():
        z = np.load(root / f"{key}.npz")
        if z["valid"].mean() >= 0.9:
            break
    print("tile", key, "valid", round(float(z["valid"].mean()), 3))
    lr = torch.from_numpy(z["lr"].astype(np.float32))[None]
    guide = torch.from_numpy(z["guide"].astype(np.float32))[None]
    ref = z["ref"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"input": upsample(lr)[0].numpy() * 0 + np.repeat(np.repeat(z["lr"], 2, 1), 2, 2), "ref": ref}
    for m in ("bicubic", "rcan_g", "dsen2", "dsen2_global"):
        net = build_model(specs[m], mtf).to(dev).eval()
        if getattr(net, "trainable", False):
            net.load_state_dict(torch.load(ROOT / "runs/p1" / m / "seed0/best.pt", map_location=dev,
                                           weights_only=True))
        with torch.no_grad():
            out[m] = net(lr.to(dev), guide.to(dev)).float()[0].cpu().numpy()
    # crop with the most reference detail (B8A high-pass energy, valid pixels only)
    b = target_index("B8A")
    hp = ref[b] - ndimage.gaussian_filter(ref[b], 2)
    energy = ndimage.uniform_filter(np.where(z["valid"], hp ** 2, 0), CROP)
    ok = ndimage.minimum_filter(z["valid"].astype(np.uint8), CROP) > 0
    energy[~ok] = -1
    h = CROP // 2
    energy[:h], energy[-h:], energy[:, :h], energy[:, -h:] = -1, -1, -1, -1
    cy, cx = np.unravel_index(np.argmax(energy), energy.shape)
    sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42,
                         "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"]})
    fig, axes = plt.subplots(len(BANDS), len(COLS), figsize=(5.0, 1.95))   # 5 in: VJCS text width
    for i, band in enumerate(BANDS):
        k = target_index(band)
        for j, (m, title) in enumerate(COLS):
            img = out[m][k][sl]
            lo, hi = np.percentile(img, [2, 98])
            ax = axes[i, j]
            ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=6.5)
            if j == 0:
                ax.set_ylabel(band, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, dpi=300)
    pd.Series(dict(tile=key, crop_row=int(cy - h), crop_col=int(cx - h))).to_csv(Path(a.out).with_suffix(".csv"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
