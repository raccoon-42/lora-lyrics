# Artist-Conditional Lyric Generation with QLoRA Adapter Blending

CENG 467 Natural Language Understanding and Generation -- Term Project  
Izmir Institute of Technology, Spring 2026

**Report (Overleaf, view-only):** https://www.overleaf.com/read/sfkjmpsgnxhs#ecadbb  
The compiled report is also included as `report/final_report.pdf`.

## Overview

Per-artist QLoRA adapters on Gemma 4 E4B for style-conditional lyric generation, with inference-time adapter blending for style interpolation. Evaluation uses four independent instruments — a RoBERTa-based artist-attribution classifier (87.3% accuracy, macro-F1 0.86 over five artists), a cross-artist perplexity matrix, distinctive-token lexical metrics, and a classifier-free sentence-embedding analysis — and the central finding is that they disagree.

## Repository Structure

```
├── report/                      # LNCS report (LaTeX source + figures)
│   ├── final_report.tex          # Final report
│   ├── appendix_samples.tex      # Generated lyric samples (appendix)
│   ├── progress_report.tex       # Mid-term progress report
│   ├── presentation.tex          # Beamer slide deck
│   ├── references.bib
│   └── figures/
├── src/
│   ├── 01_inspect.ipynb          # Dataset exploration
│   ├── 02_preprocess.ipynb       # Artist selection, cleaning, train/eval split
│   ├── 03_classifier.ipynb       # RoBERTa classifier training + report figures
│   ├── 04_baselines.ipynb        # Display-only: zero-shot / few-shot results
│   ├── 05_train_adapters.ipynb   # QLoRA adapter training (per-artist + ablations)
│   ├── 06_evaluation.ipynb       # Display-only: adapter attribution + figures
│   ├── 07_blend.ipynb            # Adapter blending (CPU build) + display-only results
│   ├── 08_perplexity.ipynb       # Cross-artist perplexity matrix (GPU)
│   ├── 09_sw_compare.ipynb       # Display-only: plain vs style-weighted adapter
│   ├── evaluate.py               # Single GPU entry point: caches all eval results
│   ├── embedding_eval.py         # Classifier-free embedding attribution + 2D maps (CPU)
│   ├── experiment_clf_epochs.py  # Classifier epoch sweep (adopted epoch-5 of 10)
│   ├── gen_appendix.py           # Generate appendix_samples.tex from cached results
│   ├── config.py                 # Shared paths, constants, Adapter registry
│   ├── generation/               # Base model, data, adapter training, generation
│   ├── classifier/               # RoBERTa model, data, training, classify
│   ├── evaluation/               # Attribution metrics, perplexity, blending
│   ├── figures/                  # Figure/plot generation scripts (run as `python -m figures.<name>`)
│   │   ├── regen_report_figures.py  # Training-curve, method-comparison + rank-ablation figures (CPU)
│   │   ├── regen_clf_figures.py     # Confusion matrix + report from the adopted classifier
│   │   └── clf_report.py            # Per-class report + confusion PNGs for any classifier dir
│   ├── judge/                    # Exportable LLM-as-judge eval (rubric, runner, agreement)
│   ├── artifacts/                # Trained weights (adapters/ + classifier/)
│   ├── results/                  # Cached eval results (JSON, tracked in git)
│   ├── pyproject.toml            # Python dependencies
│   └── uv.lock
```

All GPU evaluation compute lives in `evaluate.py` (loads the base model + classifier once, then caches baselines, adapter, and blend results under `results/`). The `04`/`06`/`07`/`09` notebooks are **display-only**: they read those caches to build tables and figures, no model load.

## Reproducing Results

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with >= 16 GB VRAM for QLoRA training (trained on RTX 5090, 32 GB)
- Classifier training can run on CPU or MPS (Apple Silicon)

### 1. Setup

```bash
cd src
uv sync
```

This installs all dependencies from `pyproject.toml` into a local `.venv`.

### 2. Dataset Preparation

Run notebooks in order:

```bash
# Activate the environment
source .venv/bin/activate

# 1. Explore the dataset
jupyter notebook 01_inspect.ipynb

# 2. Select artists, clean lyrics, create train/eval split
jupyter notebook 02_preprocess.ipynb
# Outputs: data/train.csv, data/eval.csv
```

The dataset (`genius-lyrics-cleaned`) is downloaded automatically from Hugging Face on first run. Cleaning strips section headers (`[Verse]`, `[Chorus]`), drops lyrics under 100 characters, and NFKC-normalizes the text (folds scraped Unicode junk such as exotic whitespace and homoglyphs while preserving line breaks).

### 3. Classifier Training

```bash
jupyter notebook 03_classifier.ipynb
# Outputs: artifacts/classifier/best_model/
```

Trains a RoBERTa-base artist-attribution classifier (5 classes, ~84% accuracy). Runs on CPU/MPS in ~5 minutes.

### 4. QLoRA Adapter Training (requires CUDA GPU)

```bash
jupyter notebook 05_train_adapters.ipynb
# Outputs: artifacts/adapters/<artist>_<method>_r<rank>/
```

Downloads Gemma 4 E4B on first run (requires Hugging Face authentication with access to the model). Uses `train_adapter(model, tokenizer, train_df, artist, r=8, use_dora=False, ...)` to train adapters with configurable rank, LoRA/DoRA, and optional style-weighted loss. Trains the main per-artist set plus the rank/DoRA/style-weighted ablations.

`train_adapter` skips any adapter whose weights already exist on disk, so re-running the notebook only trains what is missing. To retrain an existing adapter, pass `overwrite=True` (or delete its directory under `artifacts/adapters/`) — this is what forces a rebuild after the training data or preprocessing changes.

### 5. Evaluation compute (requires CUDA GPU)

```bash
uv run python evaluate.py
```

Single GPU entry point. Loads the base model + classifier once, then generates and classifies for baselines (zero-shot + few-shot), every registered adapter, and the blend sweep, caching one JSON per result under `results/`. Re-runs are incremental:

- baselines recompute only when their spec (n_samples + prompt) changes
- adapters recompute only when their weights are newer than the cached entry
- blends recompute only when their spec (n_samples + alpha + source mtimes) changes

Use `--force` to ignore all caches, or `--force-baselines` / `--force-adapters` / `--force-blends` to refresh one group (e.g. after changing the generation seed).

The cached JSON under `results/` (including the generated sample text in each entry) is committed to the repository, so the display-only notebooks, figures, and report can be reproduced on a machine without a GPU.

### 6. Figures and analysis (CPU, display-only)

```bash
jupyter notebook 04_baselines.ipynb    # baseline tables
jupyter notebook 06_evaluation.ipynb   # adapter attribution + comparison figures
jupyter notebook 07_blend.ipynb        # blend build/validation + interpolation figure
jupyter notebook 09_sw_compare.ipynb   # plain vs style-weighted comparison
```

These read the caches written by `evaluate.py`; no model is loaded. The cross-artist perplexity matrix is computed separately on GPU:

```bash
jupyter notebook 08_perplexity.ipynb
```

## Evaluation

No single automatic metric captures "style," so the project evaluates adapters on three independent axes. The central finding is that they disagree, which is why all three are reported.

- **Attribution** — classifier probability assigned to the target artist (`06_evaluation.ipynb`). Measures distributional fidelity, but is gameable: it rewards surface-marker density, not coherence.
- **Perplexity** — cross-artist perplexity matrix on held-out lyrics (`08_perplexity.ipynb`). Each artist's own adapter wins its own column (full specialization); read down columns, not across rows.
- **Lexical coverage** — distinctive-token *types* per sample (breadth) vs. *occurrences* / distinct-n (repetition), in `09_sw_compare.ipynb`. Types is the lexical-richness metric; occurrences is reported only as a degeneration signal, because repetition inflates it.

### Style-Weighted Loss

`generation/style_loss.py` reweights the cross-entropy loss by token-level TF-IDF (each artist corpus treated as one document), up-weighting artist-distinctive tokens during training. Enabling it produces the `*_lora_r8_sw` adapters.

Its effect is **sign-dependent on attribution**: for already-saturated adapters (Gojira, Death, Opeth) it *lowers* attribution while adding genuine lexical breadth, whereas for under-committed adapters (Tool, Mastodon) it *raises* attribution. The Tool gain, however, is a repetition artifact — its lexical-type breadth stays flat while occurrences multiply. This plain-vs-style-weighted contrast is the concrete evidence that attribution, lexical breadth, and degeneration are independent axes; `09_sw_compare.ipynb` builds the comparison.

## Key Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Gemma 4 E4B (dense, ~4B params) |
| Quantization | 4-bit NF4, double quant, bfloat16 |
| LoRA rank | r=8, alpha=16, dropout=0.1 |
| Target modules | Attention (q,k,v,o) + MLP (gate,up,down) |
| Training | 3 epochs, effective batch 4, lr 2e-4, cosine |
| Classifier | RoBERTa-base, 5 epochs, lr 2e-5 |

## Artists

| Artist | Songs | Style |
|--------|-------|-------|
| Death | 116 | Technical, philosophical |
| Gojira | 86 | Environmental, philosophical |
| Mastodon | 154 | Sludge/prog, Southern groove |
| Opeth | 118 | Poetic, melancholic |
| Tool | 74 | Cryptic, progressive |

All five are progressive/cerebral metal acts of the same broad genre, so the classifier is forced to learn artist identity rather than genre.
