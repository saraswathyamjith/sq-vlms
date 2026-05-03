#!/bin/bash
#SBATCH --job-name=dl-clevr
#SBATCH --partition=mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

module load miniforge
source activate sq-vlm
cd ~/self-questioning-vlms

python download_clevr.py
