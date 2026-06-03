"""Evaluate baselines + every trained adapter, caching results under results/.

Run before the figures notebook (06_evaluation):

    uv run python evaluate.py

Loads the base model + classifier once, then writes one JSON per result, grouped
by artist:
  - baselines: results/baselines/<artist>/{zero_shot,few_shot}.json
  - adapters:  results/adapters/<artist>/<variant>.json

Baselines recompute only when their spec (n_samples + prompt) changes; adapters
recompute only when their weights are newer than the cached entry. Untouched
results are skipped (no GPU generation).
"""
import json
import random

import pandas as pd

from config import ARTISTS, DATA_DIR, RESULTS_DIR, adapter_registry
from classifier.classify import classify
from classifier.model import load_classifier
from evaluation.metrics import evaluate_adapter
from generation.generate import generate_samples
from generation.model import load_base_model

N_SAMPLES = 10
FEWSHOT_EXAMPLES = 3
FEWSHOT_SEED = 42

BASELINES_DIR = RESULTS_DIR / "baselines"
ADAPTERS_DIR = RESULTS_DIR / "adapters"


def _weights_mtime(path):
    # Newest file under the adapter dir -- bumps whenever the adapter is retrained.
    return max(f.stat().st_mtime for f in path.rglob("*") if f.is_file())


def _cache_baseline(model, tokenizer, clf, artist, method, prompt):
    # Spec-based guard: baselines have no weight file, so recompute only when
    # n_samples or the (deterministic) prompt changes.
    cache_file = BASELINES_DIR / artist.lower().replace(" ", "_") / f"{method}.json"
    spec = {"n_samples": N_SAMPLES, "prompt": prompt}
    if cache_file.exists() and json.load(open(cache_file)).get("spec") == spec:
        print(f"cached {artist} {method}: up to date")
        return

    print(f"\n=== {artist} {method} ===")
    samples = generate_samples(model, tokenizer, prompt, N_SAMPLES)
    df = pd.DataFrame([classify(clf, text) for text in samples])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"target": artist, "method": method, "samples": samples,
                   "df": df.to_dict(orient="list"), "spec": spec}, f, indent=2)
    print(f"  mean {df[artist].mean():.4f} +/- {df[artist].std():.4f}")


def run_baselines(model, tokenizer, clf):
    # B1 -- zero-shot: artist name in prompt, no examples.
    for artist in ARTISTS:
        prompt = f"Write song lyrics in the style of {artist}.\n\n"
        _cache_baseline(model, tokenizer, clf, artist, "zero_shot", prompt)

    # B2 -- few-shot: artist name + FEWSHOT_EXAMPLES in-context examples. Fixed
    # seed so the example selection (and thus the prompt) is stable across runs,
    # which keeps the spec cache reliable.
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    random.seed(FEWSHOT_SEED)
    for artist in ARTISTS:
        lyrics = train_df[train_df["artist"] == artist]["lyrics"].tolist()
        examples = random.sample(lyrics, FEWSHOT_EXAMPLES)
        prompt = f"Write song lyrics in the style of {artist}.\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n{ex}\n\n"
        prompt += f"Now write new song lyrics in the style of {artist}:\n\n"
        _cache_baseline(model, tokenizer, clf, artist, "few_shot", prompt)


def run_adapters(base_model, tokenizer, clf):
    for a in adapter_registry():
        if not a.path.exists():
            print(f"skip {a.label}: {a.path} not found")
            continue

        mtime = _weights_mtime(a.path)
        cache_file = ADAPTERS_DIR / a.result_relpath
        if cache_file.exists() and json.load(open(cache_file)).get("mtime") == mtime:
            print(f"cached {a.label}: up to date")
            continue

        print(f"\n=== {a.label} ===")
        samples, df = evaluate_adapter(base_model, tokenizer, clf, a.path)
        # Store everything JSON-native; the Adapter object is re-attached from the
        # registry on load, so it never needs serializing.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"target": a.artist, "samples": samples,
                       "df": df.to_dict(orient="list"), "mtime": mtime}, f, indent=2)
        print(f"  Target-artist mean: {df[a.artist].mean():.4f} "
              f"+/- {df[a.artist].std():.4f}  (wrote {a.result_relpath})")


def main():
    base_model, tokenizer = load_base_model()
    clf = load_classifier()

    print("\n##### BASELINES #####")
    run_baselines(base_model, tokenizer, clf)

    print("\n##### ADAPTERS #####")
    run_adapters(base_model, tokenizer, clf)


if __name__ == "__main__":
    main()
