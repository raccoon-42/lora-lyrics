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


# Which rubric criteria each item_type is scored on (mirrors rubric.yaml
# applies_to). "presence" is expanded per target artist by the runner.
CRITERIA_BY_TYPE = {
    "single": ["style_match", "coherence"],
    "blend": ["presence", "coherence"],
}

ITEMS_PATH = RESULTS_DIR / "judge" / "items.jsonl"


def standardized_items():
    """Yield the 140 standardized items: {id, text, metadata}.

    All 10 cached samples per config are used (10/config is the locked set, so
    no subsampling), with IDs pinned by index for reproducibility across judges
    and any future human rater. metadata carries everything the runner needs to
    dispatch criteria and fill rubric placeholders.
    """
    for cfg in item_set_configs():
        criteria = CRITERIA_BY_TYPE[cfg.item_type]
        for i, text in enumerate(cfg.samples()):
            yield {
                "id": f"{cfg.config_id}__{i:02d}",
                "text": text,
                "metadata": {
                    "config_id": cfg.config_id,
                    "kind": cfg.kind,
                    "item_type": cfg.item_type,
                    "targets": list(cfg.targets),
                    "criteria": criteria,
                },
            }


def export_items(path=ITEMS_PATH):
    items = list(standardized_items())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path, len(items)


if __name__ == "__main__":
    path, n = export_items()
    print(f"wrote {n} items -> {path}")
