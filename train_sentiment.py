from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import CONFIG
from src.data_utils import (
    load_local_csv_3class,
    load_tweet_eval_3class,
    tokenize_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_source", type=str, default="tweet_eval", choices=["tweet_eval", "local_csv"])
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--valid_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--model_name", type=str, default=CONFIG.sentiment_model_name)
    parser.add_argument("--output_dir", type=str, default="outputs/sentiment_model")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_source == "tweet_eval":
        bundle = load_tweet_eval_3class()
    else:
        if not (args.train_file and args.valid_file and args.test_file):
            raise ValueError("For local_csv, you must provide --train_file --valid_file --test_file")
        bundle = load_local_csv_3class(args.train_file, args.valid_file, args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized = bundle.dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, CONFIG.max_input_length),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=bundle.num_labels,
        id2label=bundle.id2label,
        label2id=bundle.label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,   
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"])
    print("Test metrics:", test_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()