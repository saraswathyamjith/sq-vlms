#!/bin/bash
#SBATCH --job-name=eval-base-clevr
#SBATCH --partition=mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load miniforge cuda/12.9.1
source activate sq-vlm
source .env
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd ~/self-questioning-vlms

echo "Job ID: $SLURM_JOB_ID"

python evaluate.py \
    --checkpoint Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset clevr \
    --prompt direct \
    --max_samples 500 \
    --output output/eval_clevr_base_direct.json

python evaluate.py \
    --checkpoint Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset clevr \
    --prompt sq \
    --max_samples 500 \
    --output output/eval_clevr_base_sq.json
