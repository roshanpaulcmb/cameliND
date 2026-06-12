#!/bin/bash
#SBATCH --job-name=8DM9restart
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err

#SBATCH --nodes=1
#SBATCH -p dept_gpu
#SBATCH -C L40|A100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

#SBATCH --mail-user=rop174@pitt.edu
#SBATCH --mail-type=END,FAIL

# Resume a single production run from its directory. The dir fully describes the
# run (params.json + system.xml + solvated.pdb + state.xml), so nothing else is
# needed. state.xml is portable, so this can land on any GPU/partition.
#
#   sbatch restart.slurm simulation/8DM9_1/rank0003/run1

eval "$(conda shell.bash hook)"
conda activate cameliND

set -euo pipefail

DIR="${1:?usage: sbatch restart.slurm <run_dir>}"
echo "Restarting ${DIR}"

python -m cameliND restart --dir "${DIR}"
