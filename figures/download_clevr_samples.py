import os
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading CLEVR...")
clevr = load_dataset("HuggingFaceM4/clevr", "default", split="validation", trust_remote_code=True)

for i in range(5):
    ex = clevr[i]
    img = ex["image"]
    path = os.path.join(OUT_DIR, f"clevr_{i}.png")
    img.save(path)
    print(f"  [{i}] Saved {path}")
    print(f"      Q: {ex['question']}")
    print(f"      A: {ex['answer']}")
    print()

print(f"Done -- samples in {OUT_DIR}/")
