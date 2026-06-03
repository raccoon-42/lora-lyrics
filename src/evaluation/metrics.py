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

import re

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


def distinctive_tokens(weights, tokenizer, n=40, min_len=3, drop_stopwords=False):
    """Set of an artist's top-`n` distinctive tokens, lowercased and filtered to
    alphabetic tokens of length >= `min_len` (drops subword fragments and
    punctuation). `weights` come from `generation.style_loss.build_style_weights`.

    `drop_stopwords` additionally removes English function words (sklearn's list).
    This is a per-artist knob: token-level TF-IDF cleanly captures lexically-concrete
    styles (e.g. Gojira: mountains/stars/ocean) but for register-based styles
    (e.g. Tool) the top tokens are diluted by function words, so filtering gives a
    fairer view of distinctive *content* vocabulary."""
    _, idx = torch.topk(weights, n)
    toks = (tokenizer.decode([i]).strip() for i in idx)
    out = {t.lower() for t in toks if t.isalpha() and len(t) >= min_len}
    if drop_stopwords:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        out -= set(ENGLISH_STOP_WORDS)
    return out


def _word_patterns(tokens):
    # Whole-word matching: substring matching overcounts badly (e.g. the subword
    # fragment "uck" matches inside every "fuck", "low" inside "below"). Word
    # boundaries make such fragments inert and stop short tokens matching inside
    # unrelated words.
    return [re.compile(rf"\b{re.escape(t)}\b") for t in tokens]


def token_occurrences(samples, tokens):
    """Total whole-word occurrences of `tokens` across `samples` (counts repeats --
    sensitive to degeneration/looping)."""
    pats = _word_patterns(tokens)
    return sum(len(p.findall(s.lower())) for s in samples for p in pats)


def token_types(samples, tokens):
    """Mean number of *distinct* `tokens` present (as whole words) per sample
    (ignores repeats -- measures vocabulary breadth, not loop count)."""
    if not samples:
        return 0.0
    pats = _word_patterns(tokens)
    return sum(sum(bool(p.search(s.lower())) for p in pats) for s in samples) / len(samples)
