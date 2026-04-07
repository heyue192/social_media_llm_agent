from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from src.config import ID2LABEL


LABEL_ORDER = ["negative", "neutral", "positive"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--save_dir", type=str, default="outputs/eval")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_ds = ds.map(tokenize_fn, batched=True)

    tokenized_ds = tokenized_ds.remove_columns(["text"])

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # 如果这里报错，就改成 tokenizer=tokenizer
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    preds_output = trainer.predict(tokenized_ds)
    logits = preds_output.predictions
    y_pred_ids = np.argmax(logits, axis=-1)
    y_true_ids = np.array(ds["label"])

    y_true = [ID2LABEL[int(x)] for x in y_true_ids]
    y_pred = [ID2LABEL[int(x)] for x in y_pred_ids]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=4)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    metrics_path = os.path.join(args.save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(cmap="Blues")
    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {save_path}")


if __name__ == "__main__":
    main()