"""Training-data loading and per-artist dataset construction.

The adapters train on the prompt-prefixed lyric (`PROMPT + clean`), so the
`text` column built here matches exactly what the model sees at generation time
(`config.PROMPT`). Keeping this in one place means training and inference can't
drift apart.
"""

import pandas as pd
from datasets import Dataset

from config import DATA_DIR, PROMPT


def load_train_df(path=None, text_col="clean"):
    """Load training lyrics and add the prompt-prefixed `text` column SFTTrainer
    trains on. Defaults to `data/train.csv`."""
    path = path or (DATA_DIR / "train.csv")
    df = pd.read_csv(path)
    df["text"] = PROMPT + df[text_col].astype(str)
    return df


def artist_dataset(train_df, artist, artist_col="artist"):
    """HF `Dataset` of one artist's prompt-prefixed lyrics (just the `text` column)."""
    rows = train_df[train_df[artist_col] == artist].reset_index(drop=True)
    return Dataset.from_pandas(rows[["text"]])
