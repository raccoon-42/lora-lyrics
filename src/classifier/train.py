"""Fine-tune RoBERTa for 5-class artist attribution.

Returns the trained `Trainer` so the notebook can still pull `predict`,
`state.log_history`, and `save_model` for its figures and checkpoint.
"""

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}


def train_classifier(
    train_dataset, eval_dataset, label2id, id2label,
    model_name="roberta-base", cache_dir="./hf_cache",
    output_dir="./classifier_output", epochs=5, lr=2e-5,
    weight_decay=0.01, seed=42,
):
    """Build the model + Trainer, fine-tune, and return the trained Trainer."""
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
