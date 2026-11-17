#!/bin/bash
# Redo the Cloud Botany snapshots that had unreadable chunks, reading through DKRZ's reference file
# with the re-created chunks (cloud_botany_iso.py --refs). One array task per member, its times
# taken from the masked blocks of nan_report.csv. Output goes to a separate directory, so the
# original files are kept; cloud_botany_merge.py --redo-dir then prefers the redone files.
#   bash scripts/cloud_botany_redo.sbatch --list          # write and show redo_list.txt (login node)
#   sbatch --array=1-<N>%4 scripts/cloud_botany_redo.sbatch   # N = lines in redo_list.txt
#SBATCH --job-name=cb_redo
#SBATCH --partition=shared
#SBATCH --account=mh1569
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=cb_redo_%A_%a.log

set -euo pipefail
REPO=$HOME/Isentropic_Analysis
OUT=/scratch/${USER:0:1}/$USER/ISO/Cloud_Botany
REDO=${OUT}_redo
LIST=$REDO/redo_list.txt

eval "$(micromamba shell hook --shell bash)"
micromamba activate analysis
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

if [[ "${1:-}" == "--list" ]]; then
    mkdir -p "$REDO"
    python - "$OUT/nan_report.csv" > "$LIST" <<'PY'
import sys
import pandas as pd
r = pd.read_csv(sys.argv[1])
r = r[r.status == "masked"]
for m, g in r.groupby("member"):
    times = sorted({t for a, b in zip(g.first_index, g.last_index) for t in range(a, b + 1)})
    print(m, ",".join(map(str, times)))
PY
    echo "$(wc -l < "$LIST") members in $LIST:"; cat "$LIST"
    exit 0
fi

python -c "import fsspec, pyarrow, zarr"  # fail early if a reader is missing
read -r MEMBER TIMES < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
python -u "$REPO/scripts/cloud_botany_iso.py" "$MEMBER" "$TIMES" "$REDO" --refs
