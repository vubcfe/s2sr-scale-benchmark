"""Download the SEN2NEON tiles used in P1 (LR stack + canonical 2.5 m reference).

Only tiles whose Sentinel-2 / NEON time gap is at most --max-dt days are fetched (default 15),
which keeps the download to about a third of the dataset. The selected metadata is written to
<out>/metadata.csv, so Sen2NeonDataset reads exactly the downloaded tiles.

  .venv/bin/python scripts/download_sen2neon.py --out data/sen2neon --max-dt 15 --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2sr.data.sen2neon import select_tiles  # noqa: E402

REPO = "isp-uv-es/SEN2NEON"


def main():
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sen2neon")
    ap.add_argument("--max-dt", type=float, default=15)
    ap.add_argument("--limit", type=int, default=None, help="download only the first N selected tiles")
    ap.add_argument("--workers", type=int, default=1, help="parallel downloads")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(hf_hub_download(REPO, "metadata.csv", repo_type="dataset", local_dir=out / "_index"))
    sel = select_tiles(meta, max_dt_days=a.max_dt)
    if a.limit:
        sel = sel.head(a.limit)
    print(f"{len(sel)} of {len(meta)} tiles with |dt| <= {a.max_dt} days")
    todo = [rel for row in sel.itertuples() for rel in (row.lr, row.hr_2_5m_path)
            if not (out / rel).exists()]
    fetch = lambda rel: hf_hub_download(REPO, rel, repo_type="dataset", local_dir=out)  # noqa: E731
    with ThreadPoolExecutor(a.workers) as pool:
        list(tqdm(pool.map(fetch, todo), total=len(todo)))
    sel.to_csv(out / "metadata.csv", index=False)
    print(f"wrote {out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
