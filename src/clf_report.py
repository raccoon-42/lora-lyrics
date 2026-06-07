"""Per-class report + confusion matrix PNGs for a trained classifier.

Run from src/:  uv run python clf_report.py artifacts/classifier_e10

The argument is a classifier folder; the model inside is resolved in order:
a model dir itself (config.json), a best_model/ child (production layout),
or the best checkpoint recorded in the latest checkpoint's trainer_state.json
(epoch-sweep layout). Inference only -- nothing is trained or overwritten.
PNGs are saved into the given folder.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from classifier.data import LyricsDataset
from classifier.report import save_clf_report

MODEL_NAME = "roberta-base"
CACHE_DIR = "./hf_cache"


def resolve_model_dir(clf_dir):
    """Locate the model weights inside a classifier folder."""
    if (clf_dir / "config.json").exists():
        return clf_dir
    if (clf_dir / "best_model" / "config.json").exists():
        return clf_dir / "best_model"
    ckpts = sorted(clf_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
        return Path(state["best_model_checkpoint"])
    raise SystemExit(f"no model found under {clf_dir} (config.json / best_model / checkpoint-*)")


def main(clf_dir):
    clf_dir = Path(clf_dir)
    model_dir = resolve_model_dir(clf_dir)
    state_file = model_dir / "trainer_state.json"
    epoch = json.loads(state_file.read_text()).get("epoch") if state_file.exists() else None
    print(f"Model: {model_dir}" + (f"  (epoch {epoch:.0f})" if epoch else ""))

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    label2id = {k: int(v) for k, v in model.config.label2id.items()}
    names = sorted(label2id, key=label2id.get)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    eval_ds = LyricsDataset(pd.read_csv("data/eval.csv"), tokenizer, label2id)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/clf_report", report_to="none", per_device_eval_batch_size=8
        ),
    )
    out = trainer.predict(eval_ds)
    preds = out.predictions.argmax(-1)

    for path in save_clf_report(out.label_ids, preds, names, clf_dir):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clf_dir", help="classifier folder, e.g. artifacts/classifier or artifacts/classifier_e10")
    main(ap.parse_args().clf_dir)
