#!/bin/bash
module load miniforge cuda/12.9.1
source activate sq-vlm
source .env 2>/dev/null
export WANDB_PROJECT=question-vlms
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --list-gpus 2>/dev/null
