"""Score text with the artist-attribution classifier."""

import torch


@torch.no_grad()
def classify(clf, text):
    """Return {artist: probability} from the attribution classifier."""
    enc = clf.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    logits = clf.model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {clf.labels[j]: probs[j].item() for j in range(len(clf.labels))}
