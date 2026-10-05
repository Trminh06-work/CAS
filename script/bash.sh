#!/bin/bash

#SBATCH --job-name=AS_sweep             # baseline / CAS / LAS adaptive scalarisation sweep
#SBATCH --array=0-659                   # one task per line of `run_experiment.py --list` (see job.conf)
#SBATCH --ntasks=1                      # We are planning to run one task
#SBATCH --nodes=1                       # We request the use of 1 compute node for this task
#SBATCH --cpus-per-task=8               # We request the use of 2 CPU's for this single task
#SBATCH --partition=Virtual             # vCPU's only - no model in this sweep needs a GPU
#SBATCH --mem=4G                        # adjust if a task is killed for memory
#SBATCH --time=08:00:00                 # LAS refits the interpolant every iteration, so allow room
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Runs one (model, problem, n_dim, n_obj, seed) of the grid in job.conf.
#
#   sbatch bash.sh                      submit the whole array
#   ./bash.sh 7                         run task 7 here, for testing, without SLURM
#
# Set --array above to 0 .. (count - 1), where count comes from
#   python run_experiment.py --conf job.conf --count

set -eo pipefail        # not -u: ~/.bashrc often reads variables that are unset in a batch shell

cd "$(dirname "${BASH_SOURCE[0]}")"
T1=$(date +%s)
mkdir -p logs results

CONF="job.conf"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [[ -z "$TASK_ID" ]]; then
  echo "No task index: submit with sbatch, or pass one as ./bash.sh <index>" >&2
  exit 2
fi

module purge
source ~/.bashrc
conda activate jupyter_env

# one task already has the node's CPUs; let BLAS use them without oversubscribing
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

ENV_INFO="logs/env_info_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.txt"

{
  echo "Task run at $(date)"
  echo "Host: $(hostname -s)"
  echo "PWD: $(pwd)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
  echo "Task: $(python run_experiment.py --conf "$CONF" --list | sed -n "$((TASK_ID + 1))p")"
  echo "Python executable:"
  which python
  module list
} > "$ENV_INFO" 2>&1

python run_experiment.py --conf "$CONF" --index "$TASK_ID"

T2=$(date +%s)
echo "Elapsed: $((T2 - T1)) seconds" >> "$ENV_INFO"
