#!/bin/bash
# Isentropic integrals of the Cloud Botany ensemble on Levante, one array task per member.
#   sbatch scripts/cloud_botany_iso.sbatch              # all members that did not crash
#   sbatch --array=0 scripts/cloud_botany_iso.sbatch    # member 1 only, as a first test
# Output: $OUT/EXP_<member>/CB_ISO_EXP_<member>_TIME_<time>.nc; a resubmitted task skips the
# times already written. Members 7, 15, 38, 39 and 47 crashed and are left out of MEMBERS.
# Sized on member 1 (2026-10-31): 60 snapshots in 2 h 24 min, peak RSS 9.6 GB.
#SBATCH --job-name=cb_iso
#SBATCH --partition=shared
#SBATCH --account=mh1569
#SBATCH --array=0-97%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=cb_iso_%A_%a.log

set -euo pipefail
MEMBERS=($(seq 1 103 | grep -vxE '7|15|38|39|47'))
REPO=$HOME/Isentropic_Analysis
OUT=/scratch/${USER:0:1}/$USER/ISO/Cloud_Botany

eval "$(micromamba shell hook --shell bash)"
micromamba activate analysis
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
python -c "import intake, intake_xarray, jinja2, zarr, aiohttp"  # fail early if a reader is missing

python -u "$REPO/scripts/cloud_botany_iso.py" "${MEMBERS[$SLURM_ARRAY_TASK_ID]}" :: "$OUT"
