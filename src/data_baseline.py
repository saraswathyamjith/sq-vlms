from datasets import load_dataset, Dataset

from src.prompts_baseline import build_prompt_baseline


def load_clevr_dataset_baseline(split: str, max_examples: int = 10_000) -> Dataset:
    ds = load_dataset("HuggingFaceM4/clevr", "default", split=split, trust_remote_code=True)

    if max_examples and len(ds) > max_examples:
        print(f"Subsampling {max_examples} from {len(ds)} examples")
        ds = ds.select(range(max_examples))

    def transform(example):
        return {
            "prompt": build_prompt_baseline(example["question"]),
            "image": example["image"],
            "solution": str(example["answer"]),
        }

    print(f"Processing {len(ds)} examples...")
    ds = ds.map(transform, remove_columns=ds.column_names)
    print(f"Done. {len(ds)} examples ready.")
    return ds
