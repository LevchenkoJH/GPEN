#!/bin/bash

#SBATCH --job-name=gpen_training
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --error=inference-%j.err
#SBATCH --output=inference-%j.log

conda deactivate
deactivate
source venv/bin/activate

python train_simple.py

