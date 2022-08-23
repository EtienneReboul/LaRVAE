#!/bin/bash

#SBATCH --time=12:00:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/validity_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/validity_%A.err
#SBATCH --job-name=Validity_00_55
#SBATCH --mem=80000M
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
module restore LaRVAE_py_3810
python validity.py --model checkpoints/050_50_00_AdjVAE_lat55.ckpt --data data/moses_test.txt --num_samples 30000 --log_name 50_vae_00_lat55
