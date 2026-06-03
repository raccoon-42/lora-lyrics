"""Lyric generation, attribution scoring, and per-adapter evaluation."""

import pandas as pd
import torch
from peft import PeftModel

from config import GEN_KWARGS, PROMPT


def generate_lyrics(model, tokenizer, prompt=PROMPT, **gen_kwargs):
    """Sample lyrics from `model`. `gen_kwargs` overrides the defaults in GEN_KWARGS."""
    kwargs = {**GEN_KWARGS, **gen_kwargs}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, **kwargs)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


@torch.no_grad()
def classify(clf, text):
    """Return {artist: probability} from the attribution classifier."""
    enc = clf.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    logits = clf.model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {clf.labels[j]: probs[j].item() for j in range(len(clf.labels))}


def evaluate_adapter(base_model, tokenizer, clf, adapter_path, n_samples=10, verbose=True):
    """Attach `adapter_path`, generate `n_samples` lyrics, and classify each.

    Returns (samples, DataFrame of per-sample attribution probabilities).
    """
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    samples, rows = [], []
    for i in range(n_samples):
        text = generate_lyrics(model, tokenizer)
        samples.append(text)
        probs = classify(clf, text)
        rows.append(probs)
        if verbose:
            top = max(probs, key=probs.get)
            print(f"  Sample {i + 1}: {top} ({probs[top]:.3f})")
    model.unload()
    return samples, pd.DataFrame(rows)
