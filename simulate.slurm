#!/bin/bash
#SBATCH --job-name=8DM9sim
#SBATCH --output=./slurm/%x_%A_%a.out
#SBATCH --error=./slurm/%x_%A_%a.err

#SBATCH --nodes=1
#SBATCH -p dept_gpu
#SBATCH -C L40|A100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --array=1-10

#SBATCH --mail-user=rop174@pitt.edu
#SBATCH --mail-type=END,FAIL

eval "$(conda shell.bash hook)"
conda activate cameliND

set -euo pipefail

DESIGN_DIR=workbench/8DM9_1/merged/final_ranked_designs/final_100_designs

# Array index 1-10 -> rank0001 .. rank0010 (suffix varies, so glob the prefix).
RANK=rank$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")
CIF=$(ls "${DESIGN_DIR}/${RANK}"_*.cif)   # errors if 0 or >1 match
echo "Simulating ${RANK}: ${CIF}"

# Per-design/run output dir so trajectories/logs don't collide. simulate.py
# creates it and writes every artifact inside it.
OUT="simulation/8DM9_1/${RANK}/run1"

python src/simulate.py --pdb "${CIF}" --dir "${OUT}"
