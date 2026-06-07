"""Per-class report + confusion matrix for a saved classifier checkpoint.

Run from src/:  uv run python clf_report.py [artifacts/classifier_e10/checkpoint-NNN]

With no argument, resolves the best checkpoint recorded by the epoch-sweep
experiment (artifacts/classifier_e10). Inference only -- nothing is trained
or overwritten.
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
SWEEP_DIR = Path("./artifacts/classifier_e10")


def best_checkpoint(sweep_dir=SWEEP_DIR):
    """Best checkpoint recorded in the latest checkpoint's trainer_state.json."""
    ckpts = sorted(sweep_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {sweep_dir} -- pass a checkpoint path")
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    return state["best_model_checkpoint"]


def main(ckpt):
    ckpt = Path(ckpt or best_checkpoint())
    state_file = ckpt / "trainer_state.json"
    epoch = json.loads(state_file.read_text()).get("epoch") if state_file.exists() else None
    print(f"Checkpoint: {ckpt}" + (f"  (epoch {epoch:.0f})" if epoch else ""))

    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
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

    for path in save_clf_report(out.label_ids, preds, names, ckpt.parent):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "ckpt", nargs="?", default=None,
        help="checkpoint dir (default: best checkpoint of artifacts/classifier_e10)",
    )
    main(ap.parse_args().ckpt)
