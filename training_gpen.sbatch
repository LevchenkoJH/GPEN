#!/bin/bash

#SBATCH --job-name=gpen_training
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --error=training-%j.err
#SBATCH --output=training-%j.log

source venv/bin/activate

python TEST.py

