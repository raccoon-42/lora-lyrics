"""Shared configuration constants for the lyric-generation project.

Paths are relative to the `src/` directory (where the notebooks run).
"""

from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = "./models/gemma-4-E4B"
MODEL_PATH_IT = "./models/gemma-4-E4B-it"   # instruction-tuned variant, for B3/B4 instruct baselines
CLF_PATH = "./artifacts/classifier/best_model"

ADAPTERS_DIR = Path("artifacts/adapters")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")          # cached numeric results (e.g. baselines.json)
FIGURES_DIR = Path("../report/figures")

# Fixed training/inference prompt. The lyrics follow immediately after.
PROMPT = "Write song lyrics.\n\n"

# Held-out / target artists, in lineup order.
ARTISTS = ["Gojira", "Tool", "Death", "Mastodon", "Opeth"]

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

    Main set: LoRA + DoRA r=8 for every artist. Plus ablation extras: rank sweep
    on Gojira/Tool/Mastodon (picked by hypothesis: Gojira reference, Tool =
    under-committed/headroom, Mastodon = largest dataset) + style-weighted."""
    specs = []
    for artist in ARTISTS:
        specs.append(Adapter(artist, "lora", 8))
        specs.append(Adapter(artist, "dora", 8))
    specs += [
        Adapter("Gojira", "lora", 4),
        Adapter("Gojira", "lora", 16),
        Adapter("Tool", "lora", 4),
        Adapter("Tool", "lora", 16),
        Adapter("Mastodon", "lora", 4),
        Adapter("Mastodon", "lora", 16),
        Adapter("Gojira", "lora", 8, sw=True),
        Adapter("Tool", "lora", 8, sw=True),
        Adapter("Death", "lora", 8, sw=True),
        Adapter("Mastodon", "lora", 8, sw=True),
        Adapter("Opeth", "lora", 8, sw=True),
    ]
    return specs


def main_adapters():
    """The per-artist headline adapter (LoRA r=8, no SW) -- one per artist. Used
    for the perplexity diagonal and any 'one adapter per artist' view."""
    return [a for a in adapter_registry() if a.kind == "lora" and a.rank == 8 and not a.sw]


def blend_pair_key(src_a, src_b):
    """Cache/dir key for a blend of two source-adapter names. Keeps the artist
    plus the sw flag (drops lora_r8) so plain and SW pairs don't collide --
    e.g. gojira_tool (plain) vs gojira_sw_tool_sw (SW). Single source of truth
    shared by evaluate.py (writer) and 07_blend.ipynb (reader)."""
    def tag(name):
        artist = name.split("_")[0]
        return f"{artist}_sw" if name.endswith("_sw") else artist
    return f"{tag(src_a)}_{tag(src_b)}"
