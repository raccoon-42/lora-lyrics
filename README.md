# Artist-Conditional Lyric Generation with QLoRA Adapter Blending

CENG 467 Natural Language Understanding and Generation -- Term Project  
Izmir Institute of Technology, Spring 2026

## Overview

Per-artist QLoRA adapters on Gemma 4 E4B for style-conditional lyric generation, with inference-time adapter blending for style interpolation. A RoBERTa-based artist-attribution classifier (92.2% accuracy) serves as the evaluation instrument.

## Repository Structure

```
├── report/                  # LNCS report (LaTeX source + figures)
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── src/
│   ├── 01_inspect.ipynb      # Dataset exploration
│   ├── 02_preprocess.ipynb   # Artist selection, cleaning, train/eval split
│   ├── 03_classifier.ipynb   # RoBERTa classifier training + report figures
│   ├── 04_baselines.ipynb    # Zero-shot and few-shot baselines
│   ├── 05_train.ipynb        # QLoRA adapter training (per-artist)
│   ├── 06_evaluation.ipynb   # Generate lyrics + classify with RoBERTa
│   ├── pyproject.toml        # Python dependencies
│   └── uv.lock
```

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

The dataset (`genius-lyrics-cleaned`) is downloaded automatically from Hugging Face on first run.

### 3. Classifier Training

```bash
jupyter notebook 03_classifier.ipynb
# Outputs: classifier_output/best_model/
```

Trains a RoBERTa-base artist-attribution classifier (5 classes, 92.2% accuracy). Runs on CPU/MPS in ~5 minutes.

### 4. Baselines (requires CUDA GPU)

```bash
jupyter notebook 04_baselines.ipynb
# Runs zero-shot and few-shot generation for all artists, classifies output
```

### 5. QLoRA Adapter Training (requires CUDA GPU)

```bash
jupyter notebook 05_train.ipynb
# Outputs: adapters/<artist>_<method>_r<rank>/
```

Downloads Gemma 4 E4B on first run (requires Hugging Face authentication with access to the model). Uses `train_adapter(artist, r=8, use_dora=False)` to train adapters with configurable rank and LoRA/DoRA.

### 6. Evaluation (requires CUDA GPU)

```bash
jupyter notebook 06_evaluation.ipynb
# Generates lyrics from trained adapters and classifies with RoBERTa
```

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
| Meshuggah | 116 | Mechanical, abstract |
| Opeth | 118 | Poetic, melancholic |
| Tool | 74 | Cryptic, progressive |
