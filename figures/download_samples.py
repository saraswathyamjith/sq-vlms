import os
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading A-OKVQA...")
aokvqa = load_dataset("HuggingFaceM4/A-OKVQA", split="train")

for i in range(5):
    ex = aokvqa[i]
    img = ex["image"]
    path = os.path.join(OUT_DIR, f"aokvqa_{i}.png")
    img.save(path)
    print(f"  [{i}] Saved {path}")
    print(f"      Q: {ex['question']}")
    print(f"      A: {ex['direct_answers']}")
    print()

print(f"Done — samples in {OUT_DIR}/")
