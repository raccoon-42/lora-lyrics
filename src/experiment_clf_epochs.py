"""Classifier epoch-sweep experiment: train to N epochs, watch per-epoch loss.

Writes to artifacts/classifier_e{N}/ -- does NOT touch the production
classifier at artifacts/classifier/best_model.

Run from src/:  uv run python experiment_clf_epochs.py [--epochs 10]
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, set_seed

from classifier.data import LyricsDataset
from classifier.train import train_classifier

SEED = 42
MODEL_NAME = "roberta-base"
CACHE_DIR = "./hf_cache"


def main(epochs):
    set_seed(SEED)

    train_df = pd.read_csv("data/train.csv")
    eval_df = pd.read_csv("data/eval.csv")
    label2id = {name: i for i, name in enumerate(sorted(train_df["artist"].unique()))}
    id2label = {i: name for name, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    train_dataset = LyricsDataset(train_df, tokenizer, label2id)
    eval_dataset = LyricsDataset(eval_df, tokenizer, label2id)

    output_dir = f"./artifacts/classifier_e{epochs}"
    trainer = train_classifier(
        train_dataset, eval_dataset, label2id, id2label,
        model_name=MODEL_NAME, cache_dir=CACHE_DIR,
        output_dir=output_dir, epochs=epochs, seed=SEED, overwrite=True,
    )

    hist = pd.DataFrame(trainer.state.log_history)
    ev = hist.dropna(subset=["eval_loss"])[["epoch", "eval_loss", "eval_accuracy"]]
    tr = hist.dropna(subset=["loss"])[["epoch", "loss"]]

    print("\nPer-epoch eval:")
    print(ev.to_string(index=False))
    best = ev.loc[ev["eval_accuracy"].idxmax()]
    print(f"\nBest epoch: {best['epoch']:.0f}  acc={best['eval_accuracy']:.3f}  eval_loss={best['eval_loss']:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(tr["epoch"], tr["loss"], label="train loss", alpha=0.7)
    ax1.plot(ev["epoch"], ev["eval_loss"], "o-", label="eval loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()
    ax2.plot(ev["epoch"], ev["eval_accuracy"], "o-", color="green")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("eval accuracy")
    fig.suptitle(f"classifier {epochs}-epoch run")
    fig.tight_layout()
    out = f"{output_dir}/curves.png"
    fig.savefig(out, dpi=150)
    print(f"Curves -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    main(args.epochs)
