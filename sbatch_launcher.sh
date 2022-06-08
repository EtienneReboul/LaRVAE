#!/bin/bash

#SBATCH --time=00:06:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/retienne/projects/def-jeromew/retienne/TransVAE/logs/docker/docker_process_%A.out
#SBATCH --error=/home/retienne/projects/def-jeromew/retienne/TransVAE/logs/docker/docker_process_%A.err
#SBATCH --job-name=docker 
#SBATCH --mem=256M
#SBATCH --cpus-per-task=1
#SBATCH --array=0-2999
#module purge 
#module restore whole_canonical_modules
python cbas/docker.py $SLURM_ARRAY_TASK_ID 30000 --cores 1 --server cedar
