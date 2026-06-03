"""Shared configuration constants for the lyric-generation project.

Paths are relative to the `src/` directory (where the notebooks run).
"""

from pathlib import Path

MODEL_PATH = "./models/gemma-4-E4B"
CLF_PATH = "./classifier_output/best_model"

ADAPTERS_DIR = Path("adapters")
DATA_DIR = Path("data")
FIGURES_DIR = Path("../report/figures")

# Fixed training/inference prompt. The lyrics follow immediately after.
PROMPT = "Write song lyrics.\n\n"

# Default sampling config for lyric generation (matches the original 06_evaluation).
GEN_KWARGS = dict(
    max_new_tokens=512,
    min_new_tokens=200,
    temperature=0.9,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)

# Held-out / target artists, in lineup order.
ARTISTS = ["Gojira", "Tool", "Death", "Meshuggah", "Opeth"]
