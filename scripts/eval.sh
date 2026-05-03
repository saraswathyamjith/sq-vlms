#!/bin/bash
#SBATCH --job-name=sq-eval
#SBATCH --partition=mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
cd ~/self-questioning-vlms
source scripts/env.sh

mkdir -p output results
python evaluate.py "$@"
