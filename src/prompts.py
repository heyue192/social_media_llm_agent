from __future__ import annotations

from pathlib import Path


def load_response_prompt() -> str:
    prompt_path = Path("prompts") / "response_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")