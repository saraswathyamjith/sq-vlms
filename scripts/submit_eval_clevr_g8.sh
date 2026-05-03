#!/bin/bash
#SBATCH --job-name=eval-clevr-g8
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

SQ_CKPT=$(ls -d output/self-questioning-clevr-g8/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
DIRECT_CKPT=$(ls -d output/direct-rl-clevr-g8/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

echo "SQ checkpoint: $SQ_CKPT"
echo "Direct checkpoint: $DIRECT_CKPT"

if [ -z "$SQ_CKPT" ]; then
    echo "ERROR: No SQ checkpoint found"
    exit 1
fi
if [ -z "$DIRECT_CKPT" ]; then
    echo "ERROR: No Direct checkpoint found"
    exit 1
fi

python evaluate.py \
    --checkpoint "$SQ_CKPT" \
    --dataset clevr \
    --prompt direct \
    --max_samples 500 \
    --output output/eval_clevr_sq_g8_direct.json

python evaluate.py \
    --checkpoint "$SQ_CKPT" \
    --dataset clevr \
    --prompt sq \
    --max_samples 500 \
    --output output/eval_clevr_sq_g8_sq.json

python evaluate.py \
    --checkpoint "$DIRECT_CKPT" \
    --dataset clevr \
    --prompt direct \
    --max_samples 500 \
    --output output/eval_clevr_direct_g8.json
