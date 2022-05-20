#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --job-name=docker
#SBATCH --output=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/docker_%A.out
#SBATCH --error=/home/retienne/projects/def-jeromew/retienne/TransVAE/cbas/logs/docker_%A.err
#SBATCH --mem=256M
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --array=0-2999
python cbas/docker.py $SLURM_ARRAY_TASK_ID 30000 --server $1 --exhaustiveness $2 --name $3 --cores $4 --oracle $5 --target $6