#!/bin/bash
# Tile one slide per SLURM array task. Submit from the repository root, passing the cohort arguments:
#
#   mkdir -p logs
#   N=$(.venv/bin/rdmap slides -q -c cohorts.toml --cohort my-cohort | wc -l)
#   sbatch --array=1-$N scripts/slurm/tile.sh -c cohorts.toml --cohort my-cohort
#
# (or `--wsi-dir /path/to/slides` instead of `-c ... --cohort ...`). Adjust partition, time and memory to your
# cluster. Flat (non-pyramidal) TIFFs are read into memory in full: allow about 3x their uncompressed size.
#
#SBATCH --job-name=rdmap-tile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rdmap-tile-%A_%a.out

set -euo pipefail
set -a; [ ! -f .env ] || . ./.env; set +a

SLIDE_ID=$(.venv/bin/rdmap slides -q "$@" | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "Tiling slide ${SLIDE_ID} (task ${SLURM_ARRAY_TASK_ID})"
.venv/bin/rdmap tile "$@" --slide-id "${SLIDE_ID}" --progress log
