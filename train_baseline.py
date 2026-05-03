import yaml
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.data_baseline import load_clevr_dataset_baseline
from src.rewards_baseline import reward_fn_baseline

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
CONFIG_PATH = "configs/grpo_baseline.yaml"


def main():
    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)

    training_args = GRPOConfig(
        **config_dict,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
        log_completions=True,
    )

    train_dataset = load_clevr_dataset_baseline("train")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[reward_fn_baseline],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
