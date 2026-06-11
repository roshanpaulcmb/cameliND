#!/bin/bash
#SBATCH --job-name=8DM9extend
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err

#SBATCH --nodes=1
#SBATCH -p dept_gpu
#SBATCH -C L40|A100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

#SBATCH --mail-user=rop174@pitt.edu
#SBATCH --mail-type=END,FAIL

# Continue a single run to a NEW, larger total step target (e.g. 100 ns -> 150 ns).
# Pass the run dir and the new --steps target. The new target is written into the
# dir's params.json, so later restarts resume toward it.
#
#   sbatch extend.slurm simulation/8DM9_1/rank0003/run1 75000000

eval "$(conda shell.bash hook)"
conda activate cameliND

set -euo pipefail

DIR="${1:?usage: sbatch extend.slurm <run_dir> <new_total_steps>}"
STEPS="${2:?usage: sbatch extend.slurm <run_dir> <new_total_steps>}"
echo "Extending ${DIR} to ${STEPS} steps"

python src/simulate.py --extend --dir "${DIR}" --steps "${STEPS}"
