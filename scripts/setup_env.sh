#!/bin/bash
module load miniforge

mamba create -n sq-vlm python=3.11 -y
source activate sq-vlm

pip install \
    "torch>=2.1.0" \
    "transformers>=4.45.0" \
    "trl>=0.14.0" \
    "accelerate>=0.34.0" \
    datasets \
    Pillow \
    pyyaml \
    qwen-vl-utils \
    wandb
