#!/bin/bash

#SBATCH --time=3:00:00  # time limit (D-HH:MM:ss)
#SBATCH --output=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/quality%A.out
#SBATCH --error=/home/zwefers/projects/def-jeromew/zwefers/git_version/LaRVAE/logs/quality%A.err
#SBATCH --job-name=quality_no_hex
#SBATCH --mem=80000M
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
module restore LaRVAE_py_3810_new
python sampledQuality.py --samples samples/sampled_selfies_no_overload.csv --trainset_dict samples/moses_canon_smile_dict.pkl --sample_counter samples/no_overload_counter.pkl --savename samples/no_overload_selfies_quality.csv
