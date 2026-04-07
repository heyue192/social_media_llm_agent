from __future__ import annotations
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_generated_text(text: str) -> str:
    text = text.strip()
    prefixes = [
        "最终的回复：",
        "最终回复：",
        "Final reply:",
        "Response:",
        "回复：",
        "答复：",
        "输出：",
        "下面是回复：",
        "以下是回复："
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    text = text.replace("---", " ").strip()
    text = re.sub(r"\n{2,}", "\n", text).strip()

    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if parts:
        text = parts[0]

    text = re.sub(r"#\S+", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text.strip()


class QwenResponseGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 48,
        temperature: float = 0.2,
        top_p: float = 0.8,
        device: int = -1,
    ):
        model_path = Path(model_name).resolve()
        model_path_fixed = str(model_path).replace("\\", "/")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_fixed,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_fixed,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

        if device >= 0 and torch.cuda.is_available():
            self.model = self.model.to(f"cuda:{device}")

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful and concise social media assistant."},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            model_inputs = self.tokenizer(prompt, return_tensors="pt").input_ids

        model_inputs = model_inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.08,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][model_inputs.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        cleaned = clean_generated_text(text)
        return cleaned if cleaned else text.strip()