import argparse

import pandas as pd
import wandb
import yaml
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.rewards import reward_fn, reward_fn_direct

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


class CompletionLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "_last_completions"):
            return
        completions_data = model._last_completions
        if completions_data:
            df = pd.DataFrame(completions_data)
            wandb.log(
                {"completions": wandb.Table(dataframe=df)},
                step=state.global_step,
            )
            model._last_completions = []


class LoggingGRPOTrainer(GRPOTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        if hasattr(self, "_metrics") and self._metrics:
            if not hasattr(model, "_last_completions"):
                model._last_completions = []
            completion_log = {}
            for k, v in self._metrics.items():
                if isinstance(v, (int, float, str)):
                    completion_log[k] = v
            if completion_log:
                model._last_completions.append(completion_log)

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="clevr", choices=["clevr", "aokvqa"])
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument("--prompt", type=str, default="sq", choices=["sq", "direct"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    training_args = GRPOConfig(
        **config_dict,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
        log_completions=True,
    )

    if args.prompt == "direct":
        from src.data import load_dataset_direct
        train_dataset = load_dataset_direct(args.dataset, "train")
    elif args.dataset == "clevr":
        from src.data import load_clevr_dataset
        train_dataset = load_clevr_dataset("train")
    else:
        from src.data import load_aokvqa_dataset
        train_dataset = load_aokvqa_dataset("train")

    selected_reward = reward_fn_direct if args.prompt == "direct" else reward_fn

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_name = args.model if args.model else MODEL_NAME
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[selected_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
