#!/bin/bash
# Isentropic integrals of the SAM RCE runs on Torch, one array task per run.
#   sbatch scripts/sam_iso_integrals.sbatch
# Output: $OUT/iso_sums_<run directory>.nc, read by notebooks/isentropic_diagnostics.ipynb
#SBATCH --job-name=iso_sums
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=iso_sums_%A_%a.log
##SBATCH --account=YOUR_ACCOUNT

RUN_DIRS=(/scratch/op13/SAM-Torch/RCE_REF /scratch/op13/SAM-Torch/RCE_REF_TKE_M2005)
RUN_NAMES=(RCE_REF_300K RCE_REF_TKE_M2005_300K)
OUT=/scratch/op13/SAM-Torch/iso_sums

module purge
source /scratch/op13/home/env/iso/bin/activate
mkdir -p "$OUT"

i=$SLURM_ARRAY_TASK_ID
python ~/python/Isentropic_Analysis/scripts/sam_iso_integrals.py "${RUN_DIRS[$i]}" "${RUN_NAMES[$i]}" \
    -o "$OUT/iso_sums_$(basename "${RUN_DIRS[$i]}").nc"
