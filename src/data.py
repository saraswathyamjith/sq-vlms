import ast
from collections import Counter

from datasets import load_dataset, Dataset

from src.prompts import build_prompt
from src.prompts_baseline import build_prompt_baseline


def load_clevr_dataset(split: str, max_examples: int = 10_000) -> Dataset:
    ds = load_dataset("HuggingFaceM4/clevr", "default", split=split, trust_remote_code=True)

    if max_examples and len(ds) > max_examples:
        print(f"Subsampling {max_examples} from {len(ds)} examples")
        ds = ds.select(range(max_examples))

    def transform(example):
        return {
            "prompt": build_prompt(example["question"]),
            "image": example["image"],
            "solution": str(example["answer"]),
        }

    print(f"Processing {len(ds)} examples...")
    ds = ds.map(transform, remove_columns=ds.column_names)
    print(f"Done. {len(ds)} examples ready.")
    return ds


def load_gqa_dataset(max_examples: int = 500) -> Dataset:
    instructions = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_instructions", split="testdev"
    )
    images_ds = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_images", split="testdev"
    )

    print("Building GQA image lookup...")
    image_lookup = {row["id"]: row["image"] for row in images_ds}

    if max_examples and len(instructions) > max_examples:
        print(f"Subsampling {max_examples} from {len(instructions)} instructions")
        instructions = instructions.select(range(max_examples))

    def transform(example):
        image = image_lookup.get(example["imageId"])
        return {
            "prompt": build_prompt(example["question"]),
            "image": image,
            "solution": example["answer"],
            "_has_image": image is not None,
        }

    print(f"Processing {len(instructions)} GQA examples...")
    ds = instructions.map(transform, remove_columns=instructions.column_names)
    ds = ds.filter(lambda x: x["_has_image"])
    ds = ds.remove_columns(["_has_image"])
    print(f"Done. {len(ds)} GQA examples ready.")
    return ds


def load_aokvqa_dataset(split: str, max_examples: int = 10_000) -> Dataset:
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split, trust_remote_code=True)

    if max_examples and len(ds) > max_examples:
        print(f"Subsampling {max_examples} from {len(ds)} examples")
        ds = ds.select(range(max_examples))

    def transform(example):
        answers = example["direct_answers"]
        if isinstance(answers, str):
            answers = ast.literal_eval(answers)
        best_answer = Counter(answers).most_common(1)[0][0]
        return {
            "prompt": build_prompt(example["question"]),
            "image": example["image"],
            "solution": str(best_answer),
        }

    print(f"Processing {len(ds)} A-OKVQA examples...")
    ds = ds.map(transform, remove_columns=ds.column_names)
    print(f"Done. {len(ds)} examples ready.")
    return ds


def load_dataset_direct(dataset_name: str, split: str, max_examples: int = 10_000) -> Dataset:
    if dataset_name == "aokvqa":
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split, trust_remote_code=True)
    else:
        ds = load_dataset("HuggingFaceM4/clevr", "default", split=split, trust_remote_code=True)

    if max_examples and len(ds) > max_examples:
        print(f"Subsampling {max_examples} from {len(ds)} examples")
        ds = ds.select(range(max_examples))

    def transform(example):
        question = example["question"]
        if dataset_name == "aokvqa":
            answers = example["direct_answers"]
            if isinstance(answers, str):
                answers = ast.literal_eval(answers)
            solution = Counter(answers).most_common(1)[0][0]
        else:
            solution = str(example["answer"])
        return {
            "prompt": build_prompt_baseline(question),
            "image": example["image"],
            "solution": str(solution),
        }

    print(f"Processing {len(ds)} {dataset_name} examples (direct prompt)...")
    ds = ds.map(transform, remove_columns=ds.column_names)
    print(f"Done. {len(ds)} examples ready.")
    return ds
