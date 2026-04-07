from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from src.config import LABEL2ID, ID2LABEL


@dataclass
class DatasetBundle:
    dataset: DatasetDict
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]


VALID_LABEL_STRINGS = {"negative", "neutral", "positive"}


def normalize_label(value):
    """Normalize user labels from string/int to 0/1/2."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in VALID_LABEL_STRINGS:
            return LABEL2ID[value]
        if value.isdigit():
            value = int(value)
        else:
            raise ValueError(f"Unsupported string label: {value}")

    if isinstance(value, (int, float)):
        value = int(value)
        if value in ID2LABEL:
            return value

    raise ValueError(f"Unsupported label value: {value}")


def load_tweet_eval_3class() -> DatasetBundle:
    ds = load_dataset("tweet_eval", "sentiment")
    # tweet_eval sentiment uses 0=negative, 1=neutral, 2=positive
    return DatasetBundle(
        dataset=DatasetDict(
            train=ds["train"],
            validation=ds["validation"],
            test=ds["test"],
        ),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def _read_csv_to_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, label")

    df = df[["text", "label"]].dropna().copy()
    df["label"] = df["label"].apply(normalize_label)
    return Dataset.from_pandas(df, preserve_index=False)


def load_local_csv_3class(train_file: str, valid_file: str, test_file: str) -> DatasetBundle:
    return DatasetBundle(
        dataset=DatasetDict(
            train=_read_csv_to_dataset(train_file),
            validation=_read_csv_to_dataset(valid_file),
            test=_read_csv_to_dataset(test_file),
        ),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def load_sentiment140_binary_local(csv_path: str) -> pd.DataFrame:
    """
    Minimal loader for a common Sentiment140 CSV export.
    Expected fields usually include target and text; the exact column layout may vary.

    Output labels:
      0 -> negative
      2 -> positive  (later remapped to 0/2 if you want a binary baseline)

    This helper is only for baseline comparison because Sentiment140 is naturally binary.
    """
    df = pd.read_csv(csv_path, encoding="latin-1", header=None)
    if df.shape[1] < 6:
        raise ValueError("Unexpected Sentiment140 format.")

    df = df.iloc[:, [0, 5]].copy()
    df.columns = ["target", "text"]
    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: "negative", 4: "positive"})
    return df[["text", "label"]]


def tokenize_batch(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
