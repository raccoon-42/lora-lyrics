"""Torch dataset for the RoBERTa attribution classifier.

Unlike the generation side (which hands `trl.SFTTrainer` a raw-text
`datasets.Dataset`), HF `Trainer` for sequence classification wants a map-style
`torch.utils.data.Dataset` that yields already-tokenized tensors + integer
labels. Hence a subclass rather than `datasets.Dataset`.
"""

import torch
from torch.utils.data import Dataset


class LyricsDataset(Dataset):
    """Tokenized lyrics + artist labels for sequence classification."""

    def __init__(self, df, tokenizer, label2id, max_length=512, text_col="clean", artist_col="artist"):
        self.encodings = tokenizer(
            df[text_col].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([label2id[a] for a in df[artist_col]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
