#!/bin/bash
# List every unreadable Zarr chunk behind the masked blocks of the Cloud Botany isentropic run.
#   sbatch scripts/cloud_botany_find_chunks.sbatch
# Rebuilds nan_report.csv from all EXP_*/nan_log.csv, then reads the first 16 bytes of every chunk
# at the affected levels of each masked block (about 100 small requests per block) and writes
# bad_chunks.csv next to the report (the full report table goes to nan_report.txt).
#SBATCH --job-name=cb_chunks
#SBATCH --partition=shared
#SBATCH --account=mh1569
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=cb_chunks_%j.log

set -euo pipefail
REPO=$HOME/Isentropic_Analysis
OUT=/scratch/${USER:0:1}/$USER/ISO/Cloud_Botany

eval "$(micromamba shell hook --shell bash)"
micromamba activate analysis
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

python -u "$REPO/scripts/cloud_botany_nan_report.py" "$OUT" > "$OUT/nan_report.txt"
python -u "$REPO/scripts/cloud_botany_find_chunks.py" "$OUT/nan_report.csv" --max-blocks 0
