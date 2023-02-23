#!/bin/bash

#SBATCH --job-name=gpen_training
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --error=inference-%j.err
#SBATCH --output=inference-%j.log



python3 -m venv env

source venv/bin/activate

which python

python3 -m pip install -r requirements.txt

python train_simple.py

