#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --time=00:45:00
#SBATCH --job-name=sampler
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/cbas/logs/sampler_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/cbas/logs/sampler_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
python cbas/sampler.py  --model $1 --model_ckpt $2 --sample_mode $3 --name $4 --n_samples $5 --cores $6 --iteration $7 
