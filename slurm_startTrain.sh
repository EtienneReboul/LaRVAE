#!/bin/bash

#SBATCH --time=0-0:30:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/logs/startTrainer_%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/SelfiesToFingerprints/LaRVAE/logs/startTrainer_%A.err
#SBATCH --job-name=RNNAttn
#SBATCH --mem=20000M
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1

python scripts1/train.py --model rnnattn   --data_source custom --train_mols_path data/moses_train.txt --test_mols_path data/moses_test.txt --vocab_path data/moses_char_dict.pkl --char_weights_path data/moses_char_weights.npy --epochs 20  --save_name New_code_VAE_2