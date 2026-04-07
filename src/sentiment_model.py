from __future__ import annotations

from typing import Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from src.config import ID2LABEL, LABEL2ID


class SentimentClassifier:
    def __init__(self, model_dir: str, device: int = -1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def predict(self, text: str) -> Dict[str, float | str]:
        outputs = self.pipe(text, truncation=True, top_k=None)

        # 兼容不同 transformers 版本的返回格式
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
            outputs = outputs[0]

        scores = {}
        for item in outputs:
            raw_label = item["label"]
            if raw_label.startswith("LABEL_"):
                label_id = int(raw_label.split("_")[-1])
            else:
                label_id = LABEL2ID.get(raw_label.lower(), 1)

            label_name = ID2LABEL[label_id]
            scores[label_name] = float(item["score"])

        pred_label = max(scores, key=scores.get)
        return {
            "label": pred_label,
            "scores": scores,
        }