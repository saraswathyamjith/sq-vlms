#!/bin/bash
#SBATCH --job-name=sq-train
#SBATCH --partition=mit_normal_gpu
#SBATCH -G l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
cd ~/self-questioning-vlms
source scripts/env.sh

if [[ "${1:-}" == "--baseline" ]]; then
    shift
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes 1 \
        train_baseline.py "$@"

elif [[ "${1:-}" == "--merge-and-continue" ]]; then
    shift
    LORA_CKPT="$1"; shift
    MERGED_DIR="$1"; shift

    python -c "
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
base = AutoModelForImageTextToText.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, '${LORA_CKPT}')
model = model.merge_and_unload()
model.save_pretrained('${MERGED_DIR}')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
processor.save_pretrained('${MERGED_DIR}')
print('Merged model saved to ${MERGED_DIR}')
"
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes 1 \
        train.py --model "${MERGED_DIR}" "$@"

else
    accelerate launch \
        --config_file accelerate_config.yaml \
        --num_processes 1 \
        train.py "$@"
fi
