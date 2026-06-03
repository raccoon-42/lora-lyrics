"""Model loaders: the 4-bit base LM and the artist-attribution classifier."""

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import CLF_PATH, MODEL_PATH


def bnb_config():
    """4-bit NF4 + double quantization, bfloat16 compute."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(model_path=MODEL_PATH):
    """Return (model, tokenizer) for the 4-bit base LM, pad token set to eos."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config(), device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


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
