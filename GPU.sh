#!/bin/bash
#SBATCH --job-name=UniSurf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bme_gpu
#SBATCH -x bme_gpu[01,02,09]
#SBATCH -t 2:00:00

python -u inference.py --data_path ./Sample --excel_path ./test.xlsx --surf_hemi left --model_path ./weights