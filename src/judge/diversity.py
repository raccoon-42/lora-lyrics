"""Cross-sample diversity metric for the judge item set.

For each config, embeds its 10 cached samples and measures self-similarity:
mean pairwise cosine sim over the n(n-1)/2 distinct sample pairs. High =
collapsed (e.g. -it few-shot shares an opening sentence); low = diverse.
Complements attribution (classifier/embedding ask "which artist"; this asks
"are the 10 outputs the same text 10 times").

Same encoder + normalized space as embedding_eval.py, so numbers compare.
From src/:
    uv run python -m judge.diversity
"""

import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import RESULTS_DIR
from judge.items import item_set_configs

ENCODER = "all-MiniLM-L6-v2"   # match embedding_eval.py
CACHE_DIR = "./hf_cache"


def pairwise_sims(emb):
    """Cosine sims for the n(n-1)/2 distinct pairs (upper triangle, no
    diagonal). emb rows are L2-normalized, so the dot product IS cosine."""
    S = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    return S[iu]


def main():
    enc = SentenceTransformer(ENCODER, cache_folder=CACHE_DIR)
    rows = []
    for cfg in item_set_configs():
        samples = cfg.samples()
        if len(samples) < 2:
            print(f"skip {cfg.config_id}: <2 samples")
            continue
        emb = enc.encode(samples, normalize_embeddings=True)
        sims = pairwise_sims(emb)
        rows.append({
            "config_id": cfg.config_id,
            "kind": cfg.kind,
            "item_type": cfg.item_type,
            "targets": list(cfg.targets),
            "n_samples": len(samples),
            "n_pairs": int(sims.size),
            "mean_pairwise_sim": float(sims.mean()),
            "std_pairwise_sim": float(sims.std()),
            "min_pairwise_sim": float(sims.min()),
            "max_pairwise_sim": float(sims.max()),
            "diversity": float(1.0 - sims.mean()),  # 1 - mean sim; higher = more varied
        })

    table = pd.DataFrame(rows).sort_values("mean_pairwise_sim", ascending=False)
    print("\nIntra-config diversity (most collapsed first):")
    cols = ["config_id", "kind", "n_samples", "mean_pairwise_sim", "std_pairwise_sim", "diversity"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = RESULTS_DIR / "diversity.json"
    out.write_text(json.dumps({"encoder": ENCODER, "rows": rows}, indent=2))
    print(f"\nSummary -> {out}")


if __name__ == "__main__":
    main()
