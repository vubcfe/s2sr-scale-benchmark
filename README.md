# Reduced-scale vs real-resolution benchmark of guided super-resolution for Sentinel-2 20 m bands

Code and data lists for the paper *Does reduced-scale accuracy transfer to real resolution? A benchmark of guided
super-resolution for Sentinel-2 red-edge and SWIR bands over Vietnamese agricultural landscapes*
(C. V. Bui; submitted to the Vietnam Journal of Computer Science).

Nine methods (bicubic, ATPRK, DSen2, DSen2+CA+HPF, LFC-guided, EDSR-U/G, RCAN-U/G) and two controls
(EDSR-G with a scrambled guide, original global DSen2 weights) sharpen the six Sentinel-2 20 m bands
(B5, B6, B7, B8A, B11, B12) to 10 m with the four 10 m bands as guide. All learned methods are trained under
Wald's protocol on Vietnamese scenes and evaluated under three protocols:

| Protocol | What is compared |
|---|---|
| P1, reduced scale | 40 m -> 20 m output vs the observed 20 m bands (Vietnamese test blocks, held-out region, and the Sentinel-2 images of the real-reference tiles: *site-matched P1*) |
| P2, native consistency | 10 m output degraded back to 20 m vs the observed 20 m bands |
| P3, real references | 10 m output vs registered SEN2NEON (all six bands, 31 sites) and SEN2VENuS (B5-B8A, five sites) references |

Primary metrics are contrast-invariant: correlation of high-pass components (HP-corr) and edge-position error.

## Layout

```
src/s2sr/            package: degradation (MTF), models, training, metrics, registration, evaluation, statistics
scripts/run.py       download | blocks | patches | train | eval | analyze | count-params
scripts/p1_prepare_refs.py      real references: quality control, aggregation to 10 m, registration
scripts/fetch_sen2venus_swir.py B11/B12 for SEN2VENuS patches (Sentinel-2 L2A of the same date)
scripts/download_sen2neon.py    SEN2NEON tiles
scripts/p1_tables.py, p1_figures.py, p1_qualitative.py   tables and figures of the paper
scripts/run_all.sh   the whole pipeline in order
configs/             regions, data, MTF values, methods, training recipe
splits/              block grids with their train/val/test/held-out assignment (GeoPackage, UTM 48N) and
                     the list of the 144 Sentinel-2 L2A scenes used (scenes.csv)
references/          one row per SEN2NEON tile / SEN2VENuS patch: time gap, QC status, estimated shift
                     (dy20, dx20 in 20 m pixels; shift10 in 10 m pixels), correlation peak, residual
                     misregistration and the reason for exclusion, if any
results/             analysis outputs behind the tables and figures (per-protocol tables, Kendall tau,
                     scale gap, paired RQ3/RQ4 differences, registration sensitivity, summary.md, and
                     site_level.csv: seed-averaged scores per site, method and protocol on the reference tiles)
tests/               unit tests (pytest)
```

## Installation

```bash
pip install -e ".[download,dev]"
pytest -q
```

Python >= 3.10 and PyTorch >= 2.1. Training was run on NVIDIA RTX 4090 GPUs.

## Data

* **Sentinel-2 L2A** (Sen2Cor) from the Microsoft Planetary Computer: `python scripts/run.py download` fetches the
  scenes of `configs/regions.yaml`; `splits/scenes.csv` lists the scenes used in the paper. Copy `splits/*.gpkg`
  to `data/blocks/` to reuse the paper's block splits (`run.py blocks` regenerates them with the same seed).
* **SEN2NEON** (Donike et al., IEEE GRSL 2026): `python scripts/download_sen2neon.py`.
* **SEN2VENuS** (Michel et al., Data 2022, Zenodo): download and unpack the sites ANJI, ESTUAMAR, FGMANAUS,
  LERIDA-1, MAD-AMBO and SUDOUE-4 into `data/sen2venus/<SITE>`.
* **DSen2 global weights** (control): `s2_032_lr_1e-04.hdf5` from https://github.com/lanha/DSen2, saved as
  `weights/dsen2_s2_032_lr_1e-04.hdf5`.

## Reproducing the paper

```bash
bash scripts/run_all.sh
```

or step by step:

```bash
python scripts/run.py train --method rcan_g --seed 0
python scripts/run.py eval  --method rcan_g --seed 0
python scripts/run.py analyze
python scripts/p1_tables.py  --res results/p1 --out results/p1/tables
python scripts/p1_figures.py --res results/p1 --out results/p1/figures
```

All learned methods share one recipe (`configs/train.yaml`): L1 loss over valid pixels, Adam, learning rate 1e-4
with warm-up and cosine decay, at most 100,000 iterations, early stopping on the validation blocks, three seeds.
Trained weights will be attached to a release of this repository.

## License

MIT, see `LICENSE`. Sentinel-2, SEN2NEON and SEN2VENuS data remain under their own licenses.
