"""Regenerate the deck/report classifier figures from the ADOPTED classifier.

Run from src/ on the GPU box:  uv run python -m figures.regen_clf_figures

Loads the adopted checkpoint (config.CLF_PATH = e10-ep5, acc ~0.873), runs
inference on the held-out eval set, and writes two artifacts into report/figures/:
  - confusion_matrix.pdf       (seaborn, matches 03_classifier.ipynb cell 11)
  - classification_report.txt  (per-class precision/recall/F1 + macro/weighted avg)

Inference only -- nothing is trained, and the classifier folder is not touched.
This exists because 03_classifier.ipynb draws its confusion matrix from the model
it just trained (a fresh ~0.84 run), NOT from the adopted e10-ep5 checkpoint that
actually scores generations. Run this whenever the adopted classifier changes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from classifier.data import LyricsDataset
from config import CLF_PATH

MODEL_NAME = "roberta-base"
CACHE_DIR = "./hf_cache"
FIGURES = Path("../report/figures")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(CLF_PATH)
    label2id = {k: int(v) for k, v in model.config.label2id.items()}
    artists = sorted(label2id, key=label2id.get)  # axis order from the model itself

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    eval_ds = LyricsDataset(pd.read_csv("data/eval.csv"), tokenizer, label2id)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/clf_report", report_to="none", per_device_eval_batch_size=8
        ),
    )
    out = trainer.predict(eval_ds)
    y_pred, y_true = out.predictions.argmax(-1), out.label_ids

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=artists, digits=2)
    print(f"Model: {CLF_PATH}   accuracy {acc:.3f} ({(y_pred == y_true).sum()}/{len(y_true)})")
    print(report)

    report_path = FIGURES / "classification_report.txt"
    report_path.write_text(f"Model: {CLF_PATH}\nAccuracy: {acc:.4f}\n\n{report}")
    print(f"saved -> {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=artists, yticklabels=artists, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = FIGURES / "confusion_matrix.pdf"
    plt.savefig(cm_path, bbox_inches="tight")
    print(f"saved -> {cm_path}")


if __name__ == "__main__":
    main()
