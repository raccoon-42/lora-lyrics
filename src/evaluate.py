"""Evaluate baselines + every trained adapter, caching results under results/.

Run before the figures notebook (06_evaluation):

    uv run python evaluate.py

Loads the base model + classifier once, then writes one JSON per result, grouped
by artist:
  - baselines: results/baselines/<artist>/{zero_shot,few_shot}.json
  - adapters:  results/adapters/<artist>/<variant>.json
  - blends:    results/blends/<pair>/a<alpha>.json

Baselines recompute only when their spec (n_samples + prompt) changes; adapters
recompute only when their weights are newer than the cached entry; blends
recompute only when their spec (n_samples + alpha + source adapter mtimes)
changes. Untouched results are skipped (no GPU generation).

Pass --force to ignore all caches and recompute everything, or one of
--force-baselines / --force-adapters / --force-blends to recompute just that group.
"""
import json
import random

import pandas as pd
import torch

from config import ARTISTS, DATA_DIR, RESULTS_DIR, ADAPTERS_DIR as WEIGHTS_DIR, adapter_registry
from classifier.classify import classify
from classifier.model import load_classifier
from evaluation.blend import blend_adapters
from evaluation.metrics import evaluate_adapter
from generation.generate import generate_samples
from generation.model import load_base_model

N_SAMPLES = 10
FEWSHOT_EXAMPLES = 3
FEWSHOT_SEED = 42
GEN_SEED = 0          # reset before each adapter/blend eval -> paired sampling noise across adapters

BLEND_PAIRS = [("gojira_lora_r8", "tool_lora_r8")]
BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 1 = pure first source

BASELINES_DIR = RESULTS_DIR / "baselines"
ADAPTERS_DIR = RESULTS_DIR / "adapters"
BLENDS_DIR = RESULTS_DIR / "blends"


def _weights_mtime(path):
    # Newest file under the adapter dir -- bumps whenever the adapter is retrained.
    return max(f.stat().st_mtime for f in path.rglob("*") if f.is_file())


def _cache_baseline(model, tokenizer, clf, artist, method, prompt, force=False):
    # Spec-based guard: baselines have no weight file, so recompute only when
    # n_samples or the (deterministic) prompt changes. force=True ignores the cache.
    cache_file = BASELINES_DIR / artist.lower().replace(" ", "_") / f"{method}.json"
    spec = {"n_samples": N_SAMPLES, "prompt": prompt}
    if not force and cache_file.exists() and json.load(open(cache_file)).get("spec") == spec:
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


def run_baselines(model, tokenizer, clf, force=False):
    # B1 -- zero-shot: artist name in prompt, no examples.
    for artist in ARTISTS:
        prompt = f"Write song lyrics in the style of {artist}.\n\n"
        _cache_baseline(model, tokenizer, clf, artist, "zero_shot", prompt, force)

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
        _cache_baseline(model, tokenizer, clf, artist, "few_shot", prompt, force)


def run_adapters(base_model, tokenizer, clf, force=False):
    for a in adapter_registry():
        if not a.path.exists():
            print(f"skip {a.label}: {a.path} not found")
            continue

        mtime = _weights_mtime(a.path)
        cache_file = ADAPTERS_DIR / a.result_relpath
        if not force and cache_file.exists() and json.load(open(cache_file)).get("mtime") == mtime:
            print(f"cached {a.label}: up to date")
            continue

        print(f"\n=== {a.label} ===")
        torch.manual_seed(GEN_SEED)   # same RNG state per adapter -> paired A/B (e.g. plain vs SW)
        samples, df = evaluate_adapter(base_model, tokenizer, clf, a.path)
        # Store everything JSON-native; the Adapter object is re-attached from the
        # registry on load, so it never needs serializing.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"target": a.artist, "samples": samples,
                       "df": df.to_dict(orient="list"), "mtime": mtime}, f, indent=2)
        print(f"  Target-artist mean: {df[a.artist].mean():.4f} "
              f"+/- {df[a.artist].std():.4f}  (wrote {a.result_relpath})")


def run_blends(base_model, tokenizer, clf, force=False):
    # Blends have no weight file of their own -- they're rebuilt (CPU) from two
    # source adapters each time. Spec-based guard: recompute only when n_samples,
    # the alpha, the source names, or the SOURCE adapters' mtimes change. The
    # source adapters are never retrained here; only the derived blend is rebuilt.
    for src_a, src_b in BLEND_PAIRS:
        pa, pb = WEIGHTS_DIR / src_a, WEIGHTS_DIR / src_b
        if not (pa.exists() and pb.exists()):
            print(f"skip blend {src_a}+{src_b}: source adapter missing")
            continue

        pair = f"{src_a.split('_')[0]}_{src_b.split('_')[0]}"   # e.g. gojira_tool
        src_mtimes = [_weights_mtime(pa), _weights_mtime(pb)]
        for alpha in BLEND_ALPHAS:
            cache_file = BLENDS_DIR / pair / f"a{alpha:.2f}.json"
            spec = {"n_samples": N_SAMPLES, "src_a": src_a, "src_b": src_b,
                    "alpha": alpha, "src_mtimes": src_mtimes}
            if not force and cache_file.exists() and json.load(open(cache_file)).get("spec") == spec:
                print(f"cached blend {pair} a={alpha:.2f}: up to date")
                continue

            print(f"\n=== blend {pair} a={alpha:.2f} ===")
            blend_name = blend_adapters(src_a, src_b, alpha)   # CPU: (re)writes rank-2r adapter
            torch.manual_seed(GEN_SEED)
            samples, df = evaluate_adapter(base_model, tokenizer, clf, WEIGHTS_DIR / blend_name)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"src_a": src_a, "src_b": src_b, "alpha": alpha,
                           "samples": samples, "df": df.to_dict(orient="list"),
                           "spec": spec}, f, indent=2)
            print(f"  Gojira={df['Gojira'].mean():.4f}  Tool={df['Tool'].mean():.4f}  "
                  f"(wrote {pair}/a{alpha:.2f}.json)")


def main(force_baselines=False, force_adapters=False, force_blends=False):
    base_model, tokenizer = load_base_model()
    clf = load_classifier()

    print("\n##### BASELINES #####")
    run_baselines(base_model, tokenizer, clf, force=force_baselines)

    print("\n##### ADAPTERS #####")
    run_adapters(base_model, tokenizer, clf, force=force_adapters)

    print("\n##### BLENDS #####")
    run_blends(base_model, tokenizer, clf, force=force_blends)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate baselines + adapters + blends, caching under results/.")
    p.add_argument("--force", action="store_true",
                   help="ignore all caches and recompute everything")
    p.add_argument("--force-baselines", action="store_true", help="recompute baselines only")
    p.add_argument("--force-adapters", action="store_true", help="recompute adapters only")
    p.add_argument("--force-blends", action="store_true", help="recompute blends only")
    args = p.parse_args()

    main(force_baselines=args.force or args.force_baselines,
         force_adapters=args.force or args.force_adapters,
         force_blends=args.force or args.force_blends)
