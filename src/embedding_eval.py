"""Embedding-space evaluation: classifier-free attribution + 2D map.

Embeds real artist lyrics (train.csv `clean`) and generated samples (the
cached results/ JSONs) with a sentence encoder, then measures where the
generated lyrics sit relative to each artist's real-lyric region:

- centroid cosine similarity + softmax pseudo-prob over the 5 artists
- kNN purity (fraction of the k nearest REAL lyrics by the target artist) --
  robust to multi-mode catalogs (Death early-gore vs late-philosophical)
- max similarity to ANY centroid -- lets baselines express "none of the
  above" (OOD junk floats far from every cluster), which the forced 1-of-5
  classifier cannot

The 2D scatter is ILLUSTRATION ONLY (projection distorts distances); all
metrics are computed in the full embedding space.

Runs on CPU (MacBook-friendly; no CUDA needed). Run from src/:
    uv run python embedding_eval.py [--tsne]
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from config import ARTISTS, FIGURES_DIR, RESULTS_DIR

ENCODER = "all-MiniLM-L6-v2"
CACHE_DIR = "./hf_cache"
TEMP = 0.1   # softmax temperature over centroid sims; arbitrary scale, ordering is what matters
K = 5        # kNN neighbours among real lyrics

# Which generated sets to score (skipped silently when the cache is missing).
ADAPTER_VARIANTS = ["lora_r8", "lora_r8_sw"]
BASELINE_METHODS = ["zero_shot", "few_shot", "zero_shot_it", "few_shot_it"]


def _slug(artist):
    return artist.lower().replace(" ", "_")


def load_generated():
    """Yield (artist, variant, samples) for every cached generated set."""
    sets = []
    for artist in ARTISTS:
        s = _slug(artist)
        for v in ADAPTER_VARIANTS:
            p = RESULTS_DIR / "adapters" / s / f"{v}.json"
            if p.exists():
                sets.append((artist, v, json.loads(p.read_text())["samples"]))
        for m in BASELINE_METHODS:
            p = RESULTS_DIR / "baselines" / s / f"{m}.json"
            if p.exists():
                sets.append((artist, m, json.loads(p.read_text())["samples"]))
    return sets


def main(use_tsne=False):
    df = pd.read_csv("data/train.csv")
    enc = SentenceTransformer(ENCODER, cache_folder=CACHE_DIR)

    print(f"Embedding {len(df)} real lyrics ...")
    real_emb = enc.encode(df["clean"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    real_artists = df["artist"].to_numpy()

    # Per-artist centroids (mean of normalized song vectors, renormalized).
    C = np.stack([real_emb[real_artists == a].mean(axis=0) for a in ARTISTS])
    C /= np.linalg.norm(C, axis=1, keepdims=True)

    rows, gen_sets = [], []
    for artist, variant, samples in load_generated():
        emb = enc.encode(samples, normalize_embeddings=True)
        sims = emb @ C.T                                   # (n, 5) cosine sims
        probs = np.exp(sims / TEMP)
        probs /= probs.sum(axis=1, keepdims=True)
        ti = ARTISTS.index(artist)

        nn = np.argsort(-(emb @ real_emb.T), axis=1)[:, :K]  # k nearest real songs
        purity = (real_artists[nn] == artist).mean(axis=1)

        rows.append({
            "artist": artist, "variant": variant,
            "sim_target": float(sims[:, ti].mean()),
            "pseudo_prob": float(probs[:, ti].mean()),
            "knn_purity": float(purity.mean()),
            "max_sim_any": float(sims.max(axis=1).mean()),
            "argmax": ARTISTS[int(np.bincount(sims.argmax(axis=1), minlength=len(ARTISTS)).argmax())],
        })
        gen_sets.append((artist, variant, emb))

    table = pd.DataFrame(rows)
    print("\nFull-dimensional metrics (means over samples):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = RESULTS_DIR / "embedding" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"encoder": ENCODER, "temp": TEMP, "k": K, "rows": rows}, indent=2))
    print(f"\nSummary -> {out}")

    plot_map(real_emb, real_artists, C, gen_sets, use_tsne)


def plot_map(real_emb, real_artists, C, gen_sets, use_tsne):
    """2D scatter: real songs (light), centroids (stars), generated samples.
    Illustration only -- distances are distorted by the projection."""
    gen_all = np.vstack([e for _, _, e in gen_sets])
    if use_tsne:
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, random_state=42, init="pca").fit_transform(
            np.vstack([real_emb, C, gen_all]))
        method = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(real_emb)  # map = real-lyric space; gen projected into it
        xy = pca.transform(np.vstack([real_emb, C, gen_all]))
        method = "PCA"
    real_xy, cent_xy, gen_xy = np.split(xy, [len(real_emb), len(real_emb) + len(C)])

    colors = {a: plt.cm.tab10(i) for i, a in enumerate(ARTISTS)}
    adapter_marker = {"lora_r8": ("^", "plain"), "lora_r8_sw": ("s", "SW")}

    fig, ax = plt.subplots(figsize=(9, 7))
    for a in ARTISTS:
        m = real_artists == a
        ax.scatter(*real_xy[m].T, color=colors[a], s=14, alpha=0.25, label=f"{a} (real)")
    ax.scatter(*cent_xy.T, c=[colors[a] for a in ARTISTS], marker="*", s=320,
               edgecolors="black", linewidths=0.8, zorder=5)

    seen = set()
    offset = 0
    for artist, variant, emb in gen_sets:
        pts = gen_xy[offset:offset + len(emb)]
        offset += len(emb)
        if variant in adapter_marker:
            marker, tag = adapter_marker[variant]
            label = f"generated ({tag})" if tag not in seen else None
            seen.add(tag)
            ax.scatter(*pts.T, color=colors[artist], marker=marker, s=48,
                       edgecolors="black", linewidths=0.5, label=label, zorder=4)
        else:  # baselines: gray, not artist-colored -- they're junk regardless of target
            label = "baseline" if "baseline" not in seen else None
            seen.add("baseline")
            ax.scatter(*pts.T, color="gray", marker="x", s=36, alpha=0.7, label=label, zorder=3)

    ax.set_title(f"Lyrics in sentence-embedding space ({ENCODER}, {method})\n"
                 "2D projection for illustration only -- metrics computed in full dims")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "embedding_map.pdf"
    fig.savefig(out)
    print(f"Figure -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsne", action="store_true", help="t-SNE instead of PCA for the 2D map")
    args = ap.parse_args()
    main(use_tsne=args.tsne)
