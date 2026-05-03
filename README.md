# SQ-VLMs: Self-Questioning Vision-Language Models

GRPO-based reinforcement learning that teaches vision-language models to decompose visual questions into sub-questions before answering. The model learns to generate intermediate reasoning steps (sub-questions and answers) that improve final answer accuracy on visual QA tasks.

**Base model:** Qwen2.5-VL-3B-Instruct (fine-tuned with LoRA)

## Repository Structure

```
├── train.py                 # SQ + GRPO training (main)
├── train_baseline.py        # Baseline training (no self-questioning)
├── evaluate.py              # Evaluation on test sets
├── download_clevr.py        # Download and cache CLEVR dataset
├── accelerate_config.yaml   # Accelerate distributed training config
├── requirements.txt         # Python dependencies
├── src/
│   ├── data.py              # Dataset loading (CLEVR, A-OKVQA, GQA)
│   ├── data_baseline.py     # Baseline dataset loading
│   ├── prompts.py           # Self-questioning system prompt
│   ├── prompts_baseline.py  # Direct answering prompt
│   ├── rewards.py           # Reward functions (SQ + direct)
│   └── rewards_baseline.py  # Baseline reward function
├── configs/                 # GRPO training configs (YAML)
├── scripts/                 # SLURM job submission scripts
└── figures/                 # Plotting and visualization
```

## Setup

```bash
conda create -n sq-vlm python=3.11
conda activate sq-vlm
pip install -r requirements.txt
pip install peft
```

**Note:** vLLM requires CUDA and must be installed on a GPU node. On a SLURM cluster, install from within a GPU job rather than a login node.

## Training

Training uses [TRL](https://github.com/huggingface/trl)'s GRPO trainer with a vLLM generation backend. Two prompt modes are available:

- **Self-questioning (SQ):** The model generates sub-questions, answers them, then gives a final answer.
- **Direct:** The model answers the question directly (baseline comparison).

```bash
# Self-questioning on CLEVR
accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset clevr --config configs/grpo_clevr_g8.yaml --prompt sq

# Direct prompt baseline on CLEVR
accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset clevr --config configs/grpo_clevr_direct_g8.yaml --prompt direct

# A-OKVQA
accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset aokvqa --config configs/grpo_aokvqa.yaml --prompt sq
```

On a SLURM cluster, use the provided submission scripts:

```bash
sbatch scripts/train.sh --dataset clevr --config configs/grpo_clevr_g8.yaml --prompt sq
```

Training requires 2 GPUs (1 for training, 1 for vLLM generation). Checkpoints and logs are saved to the `output_dir` specified in the config.

## Evaluation

```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint \
    --dataset clevr \
    --prompt sq \
    --max_samples 500
```

Arguments:
- `--checkpoint` — Path to a LoRA checkpoint directory or merged model
- `--dataset` — `clevr`, `aokvqa`, or `gqa`
- `--prompt` — `sq` or `direct` (must match training mode)
- `--max_samples` — Number of test examples to evaluate
- `--output` — Optional path to save results JSON

Reports overall accuracy, self-questioning format rate, and per-question-type breakdown (for CLEVR).

## Datasets

| Dataset | Source | Task |
|---------|--------|------|
| [CLEVR](https://huggingface.co/datasets/HuggingFaceM4/clevr) | HuggingFaceM4/clevr | Compositional visual reasoning |
| [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) | HuggingFaceM4/A-OKVQA | Knowledge-based VQA |
| [GQA](https://huggingface.co/datasets/lmms-lab/GQA) | lmms-lab/GQA | Compositional VQA (eval only) |

To pre-download CLEVR (recommended before training):

```bash
python download_clevr.py
```
