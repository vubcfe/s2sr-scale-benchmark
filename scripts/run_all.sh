#!/usr/bin/env bash
# Full P1 pipeline, sequentially: data, real references, training (three seeds), evaluation, analysis,
# tables and figures. Each step can also be run on its own; training resumes from runs/p1/<method>/seed<k>/last.pt.
#
#   bash scripts/run_all.sh            # everything
#   SKIP_DATA=1 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
TRAINED="dsen2 dsen2_ca_hpf lfc_guided edsr_u edsr_g rcan_u rcan_g edsr_g_scr"
FIXED="bicubic atprk dsen2_global"   # no training (dsen2_global needs weights/DSen2, see README)

if [ -z "${SKIP_DATA:-}" ]; then
  $PY scripts/run.py download                     # Sentinel-2 L2A scenes (Planetary Computer)
  mkdir -p data/blocks && cp splits/*.gpkg data/blocks/   # released block splits (or: run.py blocks)
  $PY scripts/run.py patches
  $PY scripts/download_sen2neon.py --out data/sen2neon
  # SEN2VENuS: unpack the sites listed in the README into data/sen2venus/<SITE> (Zenodo) before this step
  $PY scripts/fetch_sen2venus_swir.py --root data/sen2venus   # B11/B12 for the SEN2VENuS patches
  $PY scripts/p1_prepare_refs.py sen2neon --root data/sen2neon      # QC, aggregation to 10 m, registration
  $PY scripts/p1_prepare_refs.py sen2venus --root data/sen2venus
fi

for m in $TRAINED; do
  for s in 0 1 2; do
    $PY scripts/run.py train --method "$m" --seed "$s"
    $PY scripts/run.py eval --method "$m" --seed "$s"
  done
done
for m in $FIXED; do
  $PY scripts/run.py eval --method "$m" --seed 0
done

$PY scripts/run.py analyze
$PY scripts/p1_tables.py --res results/p1 --out results/p1/tables
$PY scripts/p1_figures.py --res results/p1 --out results/p1/figures
