"""Download the instruction-tuned Gemma to MODEL_PATH_IT (./models/gemma-4-E4B-it),
matching the base model's local layout. Used by the B3/B4 instruct baselines.

Gated repo: accept the license at huggingface.co/google/gemma-4-E4B-it and be
logged in (`huggingface-cli login` or HF_TOKEN) before running.

    uv run python download_it.py
"""
from huggingface_hub import snapshot_download

from config import MODEL_PATH_IT

REPO_ID = "google/gemma-4-E4B-it"

path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=MODEL_PATH_IT,
    # transformers + bnb only need the safetensors shards, configs, tokenizer, and
    # any custom modeling *.py -- skip GGUF / original consolidated checkpoints.
    ignore_patterns=["*.gguf", "*.pth", "consolidated*", "original/*"],
)
print(f"downloaded to: {path}")
