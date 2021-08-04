#!/bin/bash
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL   
module add cuda/10.0
module add cudnn/7-cuda-10.0


python3 train_baseline.py eng.pdtb.pdtb
python3 train_baseline.py rus.rst.rrt
python3 train_baseline.py tur.pdtb.tdb