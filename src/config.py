from dataclasses import dataclass
from pathlib import Path
import os


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass
class ProjectConfig:
    sentiment_model_name: str = os.getenv("SENTIMENT_MODEL_NAME", "distilbert-base-uncased")
    qwen_model_name: str = os.getenv("QWEN_MODEL_PATH", "C:\models\Qwen2.5-1.5B-Instruct\Qwen\Qwen2___5-1___5B-Instruct")
    max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "128"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "96"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.45"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    knowledge_path: str = os.getenv(
        "KNOWLEDGE_PATH",
        str(Path("knowledge") / "combined_knowledge.txt"),
    )


CONFIG = ProjectConfig()