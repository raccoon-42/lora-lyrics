"""Shared configuration constants for the lyric-generation project.

Paths are relative to the `src/` directory (where the notebooks run).
"""

from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = "./models/gemma-4-E4B"
CLF_PATH = "./artifacts/classifier/best_model"

ADAPTERS_DIR = Path("artifacts/adapters")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")          # cached numeric results (e.g. baselines.json)
FIGURES_DIR = Path("../report/figures")

# Fixed training/inference prompt. The lyrics follow immediately after.
PROMPT = "Write song lyrics.\n\n"

# Held-out / target artists, in lineup order.
ARTISTS = ["Gojira", "Tool", "Death", "Meshuggah", "Opeth"]

_KIND_LABEL = {"lora": "LoRA", "dora": "DoRA"}


@dataclass(frozen=True)
class Adapter:
    """One trained adapter. `name`/`path`/`label` are derived so the naming
    convention (matching `train_adapter` in 05_train_adapters) lives in one place."""

    artist: str
    kind: str = "lora"      # "lora" | "dora"
    rank: int = 8
    sw: bool = False        # style-weighted loss

    @property
    def artist_slug(self):
        return self.artist.lower().replace(" ", "_")

    @property
    def variant(self):
        suffix = "_sw" if self.sw else ""
        return f"{self.kind}_r{self.rank}{suffix}"

    @property
    def name(self):
        return f"{self.artist_slug}_{self.variant}"

    @property
    def path(self):
        return ADAPTERS_DIR / self.name

    @property
    def result_relpath(self):
        """Eval-cache path under results/adapters/, grouped by artist: gojira/lora_r8.json."""
        return Path(self.artist_slug) / f"{self.variant}.json"

    @property
    def label(self):
        sw = " SW" if self.sw else ""
        return f"{self.artist} {_KIND_LABEL[self.kind]} r={self.rank}{sw}"


def adapter_registry():
    """Canonical list of adapters the project trains -- single source of truth for
    05_train_adapters (the plan), 06_evaluation, and 08_perplexity.

    Main set: LoRA + DoRA r=8 for every artist. Plus Gojira-only ablation extras
    (rank sweep + style-weighted)."""
    specs = []
    for artist in ARTISTS:
        specs.append(Adapter(artist, "lora", 8))
        specs.append(Adapter(artist, "dora", 8))
    specs += [
        Adapter("Gojira", "lora", 4),
        Adapter("Gojira", "lora", 16),
        Adapter("Gojira", "lora", 8, sw=True),
    ]
    return specs


def main_adapters():
    """The per-artist headline adapter (LoRA r=8, no SW) -- one per artist. Used
    for the perplexity diagonal and any 'one adapter per artist' view."""
    return [a for a in adapter_registry() if a.kind == "lora" and a.rank == 8 and not a.sw]
