#!/bin/bash
# Extract UNI-2 and CONCH features for one slide per SLURM array task (run tile.sh first). Submit from the
# repository root, passing the cohort arguments:
#
#   mkdir -p logs
#   N=$(.venv/bin/rdmap slides -q -c cohorts.toml --cohort my-cohort | wc -l)
#   sbatch --array=1-$N scripts/slurm/embed.sh -c cohorts.toml --cohort my-cohort
#
# Add `--backbone uni2` or `--backbone conch` to run only one backbone. Adjust partition, GPU type, time and
# memory to your cluster. Model weights are downloaded once to MODELS_DIR (needs HF_TOKEN, see README).
#
#SBATCH --job-name=rdmap-embed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/rdmap-embed-%A_%a.out

set -euo pipefail
set -a; [ ! -f .env ] || . ./.env; set +a

SLIDE_ID=$(.venv/bin/rdmap slides -q "$@" | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "Embedding slide ${SLIDE_ID} (task ${SLURM_ARRAY_TASK_ID})"
.venv/bin/rdmap embed "$@" --slide-id "${SLIDE_ID}" --progress log
