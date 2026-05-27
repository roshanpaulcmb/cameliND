#!/bin/bash 
#SBATCH --job-name=8DM9gen
#SBATCH --output=./slurm/%x_%A_%a.out
#SBATCH --error=./slurm/%x_%A_%a.err

#SBATCH --nodes=1
#SBATCH -p dept_gpu
#SBATCH -C L40|A100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --array=0-9

#SBATCH --mail-user=rop174@pitt.edu
#SBATCH --mail-type=END,FAIL

eval "$(conda shell.bash hook)"
conda activate boltzgen

boltzgen run yamls/8DM9/8DM9_new.yaml \
	--output workbench/8DM9_1/run_${SLURM_ARRAY_TASK_ID} \
	--protocol nanobody-anything \
	--num_designs 1000 \
	--reuse \
	--steps design inverse_folding folding analysis
