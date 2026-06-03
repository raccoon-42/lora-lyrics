"""Style-weighted cross-entropy for artist-conditional adapter training.

Plain next-token CE weights every token equally, so the gradient is dominated by
function words ("the", "and") and the rare, style-carrying tokens ("whale",
"spiral") barely move the model. Here we up-weight each token's loss by how
*distinctive* it is to the target artist, measured with token-level TF-IDF over
the per-artist corpora.

Everything in this module is CPU-only. The returned loss closure is passed to
`trl.SFTTrainer(..., compute_loss_func=...)`; only the training run itself needs
the GPU.

Notebook usage
--------------
    from generation.style_loss import build_style_weights, top_tokens, make_style_loss_func

    w = build_style_weights("Gojira", train_df, tokenizer)   # text_col="clean"
    top_tokens(w, tokenizer, n=30)                            # sanity check
    loss_fn = make_style_loss_func(w)
    # then: SFTTrainer(..., compute_loss_func=loss_fn)
"""

import math
from collections import Counter

import torch
import torch.nn.functional as F


def _artist_token_counts(train_df, tokenizer, text_col, artist_col):
    """Token-id Counter per artist over their full corpus."""
    counts = {}
    for artist in sorted(train_df[artist_col].unique()):
        texts = train_df.loc[train_df[artist_col] == artist, text_col].tolist()
        c = Counter()
        for t in texts:
            c.update(tokenizer(str(t), add_special_tokens=False)["input_ids"])
        counts[artist] = c
    return counts


def build_style_weights(
    target_artist,
    train_df,
    tokenizer,
    text_col="clean",
    artist_col="artist",
    smoothing=1.0,
    max_weight=10.0,
    min_count=3,
):
    """Per-token loss weights for `target_artist`, shape (vocab_size,).

    weight(t) = (smoothing + bonus(t)) / (smoothing + 1)

    where bonus(t) is the token's TF-IDF for this artist, normalized so its
    frequency-weighted mean is 1 and clamped to `max_weight`. This construction
    keeps the frequency-weighted mean of the weights at ~1, so the overall loss
    scale (and effective learning rate) stays comparable to standard CE.

    - Distinctive tokens (high TF-IDF) -> weight up to (smoothing+max_weight)/(smoothing+1).
    - Ubiquitous tokens (TF-IDF 0, appear for every artist) -> floor smoothing/(smoothing+1).
    - `smoothing` is the knob: large -> all weights approach 1 (no effect);
      small -> common tokens approach 0 (text degenerates). Default 1.0 -> floor 0.5.
    - `min_count` guards against hapax noise: a token must occur at least this
      many times in the target corpus to earn a high weight; rarer tokens are
      left at the floor. TF-IDF maxes out one-off words (proper nouns, typos),
      so without this the top weights are dominated by noise.
    """
    counts = _artist_token_counts(train_df, tokenizer, text_col, artist_col)
    if target_artist not in counts:
        raise ValueError(f"{target_artist!r} not in {list(counts)}")

    n_artists = len(counts)
    doc_freq = Counter()
    for c in counts.values():
        doc_freq.update(c.keys())   # +1 per artist that uses the token

    target = counts[target_artist]
    total = sum(target.values())

    # raw TF-IDF and the frequency-weighted mean used to normalize it.
    # Tokens below `min_count` stay at the floor (treated as non-distinctive),
    # which keeps one-off hapaxes out of the high-weight tail.
    raw, mean_raw = {}, 0.0
    for tok, cnt in target.items():
        if cnt < min_count:
            continue
        tf = cnt / total
        idf = math.log(n_artists / doc_freq[tok])   # 0 when token used by all artists
        raw[tok] = tf * idf
        mean_raw += tf * raw[tok]

    floor = smoothing / (smoothing + 1.0)
    weights = torch.full((len(tokenizer),), floor, dtype=torch.float32)
    if mean_raw > 0:
        for tok, r in raw.items():
            bonus = min(r / mean_raw, max_weight)
            weights[tok] = (smoothing + bonus) / (smoothing + 1.0)
    return weights


def top_tokens(weights, tokenizer, n=30):
    """Print the n highest-weighted tokens -- a sanity check that the weights
    actually pick up the artist's distinctive vocabulary."""
    vals, idx = torch.topk(weights, n)
    rows = [(tokenizer.decode([i]).strip(), round(v.item(), 3)) for v, i in zip(vals, idx)]
    width = max((len(t) for t, _ in rows), default=0)
    for tok, w in rows:
        print(f"  {tok:<{width}}  {w}")
    return rows


def make_style_loss_func(weights):
    """Build a `compute_loss_func` for SFTTrainer that applies `weights` to the
    per-token cross-entropy. Signature matches transformers:
    (outputs, labels, num_items_in_batch=None)."""
    weights = weights.detach().clone()

    def style_loss_func(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        # standard causal shift: predict token t+1 from position t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
        flat_labels = shift_labels.view(-1)

        ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
        mask = flat_labels != -100
        tok_w = weights.to(flat_labels.device)[flat_labels.clamp(min=0)]
        weighted = ce * tok_w * mask

        if num_items_in_batch is not None:   # HF normalizes by token count across accumulation
            return weighted.sum() / num_items_in_batch
        return weighted.sum() / mask.sum().clamp(min=1)

    return style_loss_func
