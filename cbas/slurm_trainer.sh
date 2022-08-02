#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --time=00:40:00
#SBATCH --job-name=trainer
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/cbas/logs/trainer_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/cbas/logs/trainer_%A.err
#SBATCH --cpus-per-task=12#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
python cbas/trainer.py --iteration $1 --name $2  --quantile $3