#!/bin/bash

#SBATCH --time=1:00:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/stability_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/stability_%A.err
#SBATCH --job-name=stability
#SBATCH --mem=80000M
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
module restore LaRVAE_py_3810_new
python check_stability.py --model checkpoints/100_100_NoAdj_no_hex_new_weights.ckpt --numsample 10000 --mol_encoding selfies --hex
