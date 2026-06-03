"""Fine-tune RoBERTa for 5-class artist attribution.

Returns the trained `Trainer` so the notebook can still pull `predict`,
`state.log_history`, and `save_model` for its figures and checkpoint.

Re-running the training cell when the classifier already exists on disk loads
it instead of retraining (pass `overwrite=True` to force). The loaded path
returns an eval-only Trainer with `state.log_history` recovered from the latest
checkpoint, so the downstream `predict` and training-curve cells behave the same
whether the model was just trained or loaded.
"""

import json
from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainerState,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}


def _latest_log_history(output_dir):
    """Recover `log_history` from the highest-numbered checkpoint's trainer_state."""
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    for ckpt in reversed(ckpts):
        state = ckpt / "trainer_state.json"
        if state.exists():
            return json.loads(state.read_text()).get("log_history", [])
    return []


def _load_trained_classifier(eval_dataset, output_dir, model_dir):
    """Build an eval-only Trainer around an already-trained model, with the
    training log history reattached so the figure cells still work."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.state = TrainerState()
    trainer.state.log_history = _latest_log_history(output_dir)
    return trainer


def train_classifier(
    train_dataset, eval_dataset, label2id, id2label,
    model_name="roberta-base", cache_dir="./hf_cache",
    output_dir="./artifacts/classifier", epochs=5, lr=2e-5,
    weight_decay=0.01, seed=42, overwrite=False,
):
    """Fine-tune the model and return the trained Trainer -- or, if a saved
    classifier already exists under `output_dir/best_model`, load it instead."""
    best_model = Path(output_dir) / "best_model"
    if best_model.exists() and not overwrite:
        print(f"[load] classifier exists, not retraining: {best_model} (overwrite=True to force)")
        return _load_trained_classifier(eval_dataset, output_dir, best_model)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        cache_dir=cache_dir,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        weight_decay=weight_decay,
        learning_rate=lr,
        warmup_ratio=0.1,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
