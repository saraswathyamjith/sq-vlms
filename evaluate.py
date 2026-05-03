import argparse
import json
import os
import re
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.data import load_clevr_dataset
from src.prompts import SYSTEM_PROMPT
from src.prompts_baseline import SYSTEM_PROMPT_BASELINE


def parse_final_answer(text: str) -> str | None:
    matches = re.findall(r"Final Answer\s*:\s*(.+)", text)
    if not matches:
        matches = re.findall(r"(?<!\d\s)Answer\s*:\s*(.+)", text)
    if not matches:
        raw = text.strip().split("\n")[0].strip()
        return raw if raw else None
    return matches[-1].strip()


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def check_correct(prediction: str, solution: str) -> bool:
    pred_norm = normalize(prediction)
    sol_norm = normalize(solution)
    if not sol_norm:
        return False
    if pred_norm == sol_norm:
        return True
    return sol_norm in pred_norm


def has_sub_questions(text: str) -> bool:
    return bool(
        re.search(r"Sub-question\s+\d+\s*:", text)
        and re.search(r"Answer\s+\d+\s*:", text)
    )


def classify_clevr_question(question: str) -> str:
    q = question.lower()
    if q.startswith("how many"):
        return "count"
    if "the same" in q or "more" in q or "fewer" in q or "greater" in q or "less" in q:
        return "compare"
    if q.startswith("is there") or q.startswith("are there") or q.startswith("does") or q.startswith("do "):
        return "exist"
    return "query"


@torch.no_grad()
def run_inference(model, processor, dataset, system_prompt, max_samples: int | None = None):
    results = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in range(n):
        example = dataset[i]
        question = example["prompt"][1]["content"]
        image = example["image"]
        solution = example["solution"]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        parsed = parse_final_answer(output_text)
        correct = (
            parsed is not None
            and check_correct(parsed, solution)
        )

        results.append(
            {
                "question": question,
                "solution": solution,
                "prediction": parsed,
                "output": output_text,
                "correct": correct,
                "has_sub_questions": has_sub_questions(output_text),
            }
        )

        if (i + 1) % 50 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i + 1}/{n}] running accuracy: {acc:.4f}")

    return results


def report_clevr(results: list[dict]):
    by_type = defaultdict(list)
    for r in results:
        qtype = classify_clevr_question(r["question"])
        by_type[qtype].append(r["correct"])

    total_correct = sum(r["correct"] for r in results)
    print(f"\nOverall accuracy: {total_correct / len(results):.4f}  ({total_correct}/{len(results)})")
    format_rate = sum(r["has_sub_questions"] for r in results) / len(results)
    print(f"Self-questioning rate: {format_rate:.4f}")

    for qtype in sorted(by_type):
        vals = by_type[qtype]
        acc = sum(vals) / len(vals)
        print(f"  {qtype:>10s}: {acc:.4f}  ({sum(vals)}/{len(vals)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="sq", choices=["sq", "direct"])
    parser.add_argument("--dataset", type=str, default="clevr", choices=["clevr", "gqa", "aokvqa"])
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...")

    is_lora = os.path.exists(os.path.join(args.checkpoint, "adapter_config.json"))

    if is_lora:
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    system_prompt = SYSTEM_PROMPT if args.prompt == "sq" else SYSTEM_PROMPT_BASELINE
    print(f"Using prompt style: {args.prompt}")

    if args.dataset == "clevr":
        print("Loading CLEVR validation set ...")
        ds = load_clevr_dataset("validation", max_examples=args.max_samples)
    elif args.dataset == "aokvqa":
        print("Loading A-OKVQA validation set ...")
        from src.data import load_aokvqa_dataset
        ds = load_aokvqa_dataset("validation", max_examples=args.max_samples)
    else:
        print("Loading GQA testdev set ...")
        from src.data import load_gqa_dataset
        ds = load_gqa_dataset()
        if args.max_samples and len(ds) > args.max_samples:
            ds = ds.select(range(args.max_samples))

    print(f"Running inference on {len(ds)} {args.dataset.upper()} examples ...")
    results = run_inference(
        model, processor, ds, system_prompt, max_samples=args.max_samples
    )
    report_clevr(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
