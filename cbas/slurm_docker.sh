#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --job-name=docker
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/cbas/logs/docker_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/cbas/logs/docker_%A.err
#SBATCH --mem=512M
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --array=0-2999
python cbas/docker.py $SLURM_ARRAY_TASK_ID 30000 --server $1 --exhaustiveness $2 --name $3 --cores $4 --oracle $5 --target $6