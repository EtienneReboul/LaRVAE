#!/bin/bash

#SBATCH --time=1-12:00:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/startTrainer_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/startTrainer_%A.err
#SBATCH --job-name=lat48
#SBATCH --mem=80000M
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
module restore LaRVAE_py_3810
python scripts1/train.py --model rnnattn --d_latent 48 --data_source custom --train_mols_path data/moses_train.txt --test_mols_path data/moses_test.txt --vocab_path data/moses_char_dict.pkl --char_weights_path data/moses_char_weights.npy --epochs 70  --save_name 70_008_AdjVAE_lat48 --adj_matrix --adj_weight 0.08