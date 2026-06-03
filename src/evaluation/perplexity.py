"""Held-out perplexity evaluation for artist-conditional adapters.

Quantitative complement to the classifier attribution: does an artist's adapter
assign higher likelihood (lower perplexity) to *that* artist's held-out lyrics
than the base model and than the other artists' adapters? Arrange the answer as
a cross-artist matrix -- if an adapter specialized, its column-minimum sits on
the diagonal (lowest perplexity on its own artist).

Perplexity is measured on the lyric continuation only: the fixed training prompt
("Write song lyrics.\\n\\n") is masked, matching how the adapters were trained.
NLL is summed over all tokens and exponentiated once (corpus perplexity), which
is more stable than averaging per-song perplexity on a small held-out set.

The forward passes need the GPU (4-bit base). Notebook usage:

    from evaluation.perplexity import perplexity_matrix, plot_perplexity_matrix

    adapter_map = {
        "base":      None,
        "gojira":    "adapters/gojira_lora_r8",
        "tool":      "adapters/tool_lora_r8",
    }
    ppl = perplexity_matrix(base_model, tokenizer, adapter_map, eval_df)
    plot_perplexity_matrix(ppl)
"""

import math

import torch

from config import PROMPT


@torch.no_grad()
def corpus_perplexity(model, tokenizer, lyrics, prompt=PROMPT, max_length=512):
    """Token-level perplexity of a list of lyric strings, conditioned on `prompt`.

    Prompt tokens are masked to -100 so only the continuation contributes. Returns
    exp(sum_NLL / num_tokens) over the whole corpus.
    """
    prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    total_nll, total_tok = 0.0, 0
    for text in lyrics:
        enc = tokenizer(
            prompt + str(text), return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100
        n_tok = int((labels[:, 1:] != -100).sum())   # HF's loss denominator (shifted targets)
        if n_tok == 0:
            continue
        out = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels
        )
        total_nll += out.loss.item() * n_tok           # loss is mean NLL -> recover the sum
        total_tok += n_tok
    if total_tok == 0:
        return float("nan")
    return math.exp(total_nll / total_tok)


def perplexity_matrix(
    base_model, tokenizer, adapter_map, eval_df,
    artist_col="artist", text_col="clean", max_length=512,
):
    """Cross-artist perplexity matrix.

    `adapter_map`: dict name -> adapter path (str/Path), or None for the base model.
    Returns a DataFrame with adapter names as rows and artists as columns; each
    cell is corpus perplexity (lower = better fit). The column-min should land on
    the matching adapter if it specialized.
    """
    import pandas as pd
    from peft import PeftModel

    artists = sorted(eval_df[artist_col].unique())
    lyrics_by_artist = {
        a: eval_df.loc[eval_df[artist_col] == a, text_col].tolist() for a in artists
    }

    rows = {}
    for name, path in adapter_map.items():
        model = base_model if path is None else PeftModel.from_pretrained(base_model, str(path))
        model.eval()
        rows[name] = {
            a: corpus_perplexity(model, tokenizer, lyrics_by_artist[a], max_length=max_length)
            for a in artists
        }
        if path is not None:
            model.unload()
        print(name + ": " + "  ".join(f"{a}={rows[name][a]:.1f}" for a in artists))

    return pd.DataFrame(rows).T[artists]


def plot_perplexity_matrix(ppl_df, save_path="../report/figures/perplexity_matrix.pdf"):
    """Heatmap of the perplexity matrix; the per-column minimum is boxed."""
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    data = ppl_df.values
    fig, ax = plt.subplots(figsize=(1.4 * data.shape[1] + 2, 0.8 * data.shape[0] + 2))
    im = ax.imshow(data, cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(ppl_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(ppl_df.index)
    ax.set_xlabel("Held-out artist (lyrics being scored)")
    ax.set_ylabel("Adapter")
    ax.set_title("Held-out Perplexity (lower = better fit)", fontweight="bold")

    col_min = np.nanargmin(data, axis=0)
    for j, i in enumerate(col_min):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=2))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                    color="white", fontsize=9)

    fig.colorbar(im, ax=ax, label="perplexity")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
