"""Evaluation metrics in one import surface.

Three independent axes on "does this adapter capture the artist":
  - attribution   -- classifier says the output reads like the artist (06)
  - perplexity    -- adapter assigns high likelihood to the artist's real lyrics (08)
  - distinctive-token coverage -- output actually uses the artist's rare,
    style-carrying vocabulary, the tokens style-weighted loss up-weights (09)

`evaluate_adapter` is the cross-pipeline driver: it attaches an adapter, samples
lyrics (generation side), and scores them (classifier side). The perplexity
helpers are re-exported here so notebooks have one import for all of evaluation.
"""

import pandas as pd
import torch
from peft import PeftModel

from classifier.classify import classify
from generation.generate import generate_lyrics

from .perplexity import (  # noqa: F401  (re-export)
    corpus_perplexity,
    perplexity_matrix,
    plot_perplexity_matrix,
)


def evaluate_adapter(base_model, tokenizer, clf, adapter_path, n_samples=10, verbose=True):
    """Attach `adapter_path`, generate `n_samples` lyrics, and classify each.

    Returns (samples, DataFrame of per-sample attribution probabilities).
    """
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    samples, rows = [], []
    for i in range(n_samples):
        text = generate_lyrics(model, tokenizer)
        samples.append(text)
        probs = classify(clf, text)
        rows.append(probs)
        if verbose:
            top = max(probs, key=probs.get)
            print(f"  Sample {i + 1}: {top} ({probs[top]:.3f})")
    model.unload()
    return samples, pd.DataFrame(rows)


def attribution_stats(df, target):
    """(mean, std) of the target-artist attribution over a sample DataFrame."""
    return df[target].mean(), df[target].std()


def distinctive_tokens(weights, tokenizer, n=40, min_len=3):
    """Set of an artist's top-`n` distinctive tokens, lowercased and filtered to
    alphabetic tokens of length >= `min_len` (drops subword fragments and
    punctuation). `weights` come from `generation.style_loss.build_style_weights`."""
    _, idx = torch.topk(weights, n)
    toks = (tokenizer.decode([i]).strip() for i in idx)
    return {t.lower() for t in toks if t.isalpha() and len(t) >= min_len}


def token_occurrences(samples, tokens):
    """Total occurrences of `tokens` across `samples` (counts repeats -- sensitive
    to degeneration/looping)."""
    return sum(s.lower().count(t) for s in samples for t in tokens)


def token_types(samples, tokens):
    """Mean number of *distinct* `tokens` present per sample (ignores repeats --
    measures vocabulary breadth, not loop count)."""
    if not samples:
        return 0.0
    return sum(sum(t in s.lower() for t in tokens) for s in samples) / len(samples)
