#!/bin/bash

#SBATCH --time=1:00:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/entropy_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/entropy_%A.err
#SBATCH --job-name=Entropy
#SBATCH --mem=80000M
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
module restore LaRVAE_py_3810_new
python entropy_evolution.py --model checkpoints/100_100_NoAdj_no_hex_new_weights.ckpt --max_epoch 100 --numsample 10000 --data data/moses_test_no_hex.txt --mol_encoding selfies --hex
