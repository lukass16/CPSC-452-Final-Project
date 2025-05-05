#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=trainjobout.ipynb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=1:30:00



module load CUDA
module load cuDNN
module load miniconda
conda activate basenv
papermill pointnet_sdfvae.ipynb output_vae.ipynb
