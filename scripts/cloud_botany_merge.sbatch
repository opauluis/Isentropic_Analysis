#!/bin/bash
# Merge the per-time Cloud Botany isentropic output into one file per member on /work.
#   sbatch scripts/cloud_botany_merge.sbatch            # every member found on scratch
# Writes /work/mh1569/$USER/cloud_botany_iso/CB_ISO_EXP_<member>.nc (about 7-8 GB in total before
# compression) plus each member's nan_log.csv; each file is read back and checked against the
# inputs. The scratch files are not deleted.
#SBATCH --job-name=cb_merge
#SBATCH --partition=shared
#SBATCH --account=mh1569
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=cb_merge_%j.log

set -euo pipefail
REPO=$HOME/Isentropic_Analysis

eval "$(micromamba shell hook --shell bash)"
micromamba activate analysis
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

python -u "$REPO/scripts/cloud_botany_merge.py"
