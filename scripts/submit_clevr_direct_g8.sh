#!/bin/bash
#SBATCH --job-name=clevr-dir-g8
#SBATCH --partition=mit_normal_gpu
#SBATCH -G l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load miniforge cuda/12.9.1
source activate sq-vlm
source .env
export WANDB_PROJECT=question-vlms
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes:  $SLURM_JOB_NODELIST"
nvidia-smi --list-gpus

accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 1 \
    train.py --dataset clevr --config configs/grpo_clevr_direct_g8.yaml --prompt direct
