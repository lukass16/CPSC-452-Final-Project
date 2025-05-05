#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=trainjobout.ipynb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=4:00:00



module load CUDA
module load cuDNN
module load miniconda
conda activate basenv
papermill pointnet.ipynb output.ipynb
