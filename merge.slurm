#!/bin/bash
#SBATCH --job-name=8DM9merge
#SBATCH --output=./slurm/%x_%j.out
#SBATCH --error=./slurm/%x_%j.err

#SBATCH --nodes=1
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

#SBATCH --mail-user=rop174@pitt.edu
#SBATCH --mail-type=END,FAIL

eval "$(conda shell.bash hook)"
conda activate boltzgen

boltzgen merge \
  workbench/8DM9_1/run_0 workbench/8DM9_1/run_1 workbench/8DM9_1/run_2 \
  workbench/8DM9_1/run_3 workbench/8DM9_1/run_4 workbench/8DM9_1/run_5 \
  workbench/8DM9_1/run_6 workbench/8DM9_1/run_7 workbench/8DM9_1/run_8 \
  workbench/8DM9_1/run_9 \
  --output workbench/8DM9_1/merged

boltzgen run yamls/8DM9/8DM9_new.yaml \
  --output workbench/8DM9_1/merged \
  --protocol nanobody-anything \
  --steps filtering \
  --budget 100
