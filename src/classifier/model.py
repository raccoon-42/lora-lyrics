"""Loader for the trained RoBERTa artist-attribution classifier."""

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CLF_PATH


@dataclass
class Classifier:
    """Bundle of the attribution classifier and its id->label map."""

    model: object
    tokenizer: object
    labels: dict


def load_classifier(clf_path=CLF_PATH):
    """Return a `Classifier` for the trained RoBERTa attribution model."""
    model = AutoModelForSequenceClassification.from_pretrained(clf_path)
    tokenizer = AutoTokenizer.from_pretrained(clf_path)
    model.eval()
    return Classifier(model=model, tokenizer=tokenizer, labels=model.config.id2label)
