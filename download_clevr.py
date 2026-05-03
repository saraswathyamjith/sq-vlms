import os
import time

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["FSSPEC_HTTP_TIMEOUT"] = "600"

from datasets import load_dataset

MAX_RETRIES = 5

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"Attempt {attempt}/{MAX_RETRIES}...")
        ds = load_dataset("HuggingFaceM4/clevr", "default", split="train", trust_remote_code=True)
        print(f"Success! {len(ds)} examples cached.")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            print("Retrying in 10 seconds...")
            time.sleep(10)
        else:
            print("All attempts failed.")
            raise
