"""Pre-download model weights to the network volume cache."""

import os
import sys

# Set cache dirs before any HF imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

# Load HF token from parent .env
from dotenv import load_dotenv
load_dotenv("/workspace/.env")

from huggingface_hub import snapshot_download

MODELS_TO_DOWNLOAD = [
    "google/gemma-2-9b-it",
    "google/gemma-2-9b",
    "EssentialAI/rnj-1-instruct",
    "EssentialAI/rnj-1",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
    "Qwen/Qwen3.5-27B",  # used for both reasoning and non-reasoning
]

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Gated models (Gemma) may fail.")

    for repo_id in MODELS_TO_DOWNLOAD:
        print(f"\n{'='*60}")
        print(f"Downloading: {repo_id}")
        print(f"{'='*60}")
        try:
            path = snapshot_download(
                repo_id,
                token=token,
                ignore_patterns=["*.gguf", "*.ggml", "*.bin", "original/*"],
            )
            print(f"  -> Cached at: {path}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            continue

    print("\nAll downloads complete.")

if __name__ == "__main__":
    main()
