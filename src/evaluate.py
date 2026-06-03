"""Evaluate every trained adapter and cache per-adapter results.

Run before the figures notebook (06_evaluation):

    uv run python evaluate.py

Registry-driven: each adapter in config.adapter_registry() that exists on disk
is generated + classified, and its result written to results/adapters/<name>.json.
An adapter is re-evaluated only when its weights are newer than the cached entry,
so a retrain auto-refreshes while untouched adapters are skipped (no GPU work).
"""
import json

from config import RESULTS_DIR, adapter_registry
from classifier.model import load_classifier
from evaluation.metrics import evaluate_adapter
from generation.model import load_base_model

CACHE_DIR = RESULTS_DIR / "adapters"


def _weights_mtime(path):
    # Newest file under the adapter dir -- bumps whenever the adapter is retrained.
    return max(f.stat().st_mtime for f in path.rglob("*") if f.is_file())


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_model, tokenizer = load_base_model()
    clf = load_classifier()

    for a in adapter_registry():
        if not a.path.exists():
            print(f"skip {a.label}: {a.path} not found")
            continue

        mtime = _weights_mtime(a.path)
        cache_file = CACHE_DIR / f"{a.name}.json"
        if cache_file.exists() and json.load(open(cache_file)).get("mtime") == mtime:
            print(f"cached {a.label}: up to date")
            continue

        print(f"\n=== {a.label} ===")
        samples, df = evaluate_adapter(base_model, tokenizer, clf, a.path)
        # Store everything JSON-native; the Adapter object is re-attached from the
        # registry on load, so it never needs serializing.
        with open(cache_file, "w") as f:
            json.dump({"target": a.artist, "samples": samples,
                       "df": df.to_dict(orient="list"), "mtime": mtime}, f, indent=2)
        print(f"  Target-artist mean: {df[a.artist].mean():.4f} "
              f"+/- {df[a.artist].std():.4f}  (wrote {cache_file.name})")


if __name__ == "__main__":
    main()
