"""Item-set definition for the LLM-judge module (the per-project adapter).

Single source of truth for the 14 configs the judge scores:
  5 SW adapters + 5 few-shot-it baselines + 4 SW blends @ alpha=0.50
= 140 items (10 cached samples each). Both the diversity metric and the
future judge runner import item_set_configs() so they agree on the
population. Swapping projects means rewriting only this file.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from config import ARTISTS, RESULTS_DIR

BLEND_ANCHOR = "Gojira"
BLEND_PARTNERS = ["Tool", "Death", "Opeth", "Mastodon"]
BLEND_ALPHA = "a0.50"   # even mix: both artists maximally co-present


def _slug(artist):
    return artist.lower().replace(" ", "_")


@dataclass(frozen=True)
class ItemConfig:
    config_id: str       # stable id, e.g. adapter_gojira_sw
    kind: str            # "adapter" | "it" | "blend"
    item_type: str       # "single" | "blend" (drives which judge criterion later)
    targets: tuple       # (artist,) | (anchor, partner)
    path: Path

    def samples(self):
        return json.loads(self.path.read_text())["samples"]


def item_set_configs():
    cfgs = []
    for a in ARTISTS:
        cfgs.append(ItemConfig(
            f"adapter_{_slug(a)}_sw", "adapter", "single", (a,),
            RESULTS_DIR / "adapters" / _slug(a) / "lora_r8_sw.json"))
    for a in ARTISTS:
        cfgs.append(ItemConfig(
            f"it_{_slug(a)}_fewshot", "it", "single", (a,),
            RESULTS_DIR / "baselines" / _slug(a) / "few_shot_it.json"))
    for partner in BLEND_PARTNERS:
        pair = f"{_slug(BLEND_ANCHOR)}_sw_{_slug(partner)}_sw"
        cfgs.append(ItemConfig(
            f"blend_{_slug(BLEND_ANCHOR)}_{_slug(partner)}", "blend", "blend",
            (BLEND_ANCHOR, partner), RESULTS_DIR / "blends" / pair / f"{BLEND_ALPHA}.json"))
    return cfgs
