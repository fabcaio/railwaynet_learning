#!/bin/bash
#SBATCH --job-name=job_optimal
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --account=research-me-dcsc
#SBATCH --output=outputs/job_optimal.out
module load miniconda3
conda activate RL_env2
# N, hidden_size, lr, state_opt, final_temp, seed, job_idx 
srun python -u rail_optimal.py > logs/job_optimal.log
conda deactivate