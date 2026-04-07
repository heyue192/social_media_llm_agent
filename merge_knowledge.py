from __future__ import annotations

from pathlib import Path


def main():
    knowledge_dir = Path("knowledge")
    policy_path = knowledge_dir / "brand_policy.txt"
    examples_path = knowledge_dir / "empathetic_examples.txt"
    merged_path = knowledge_dir / "combined_knowledge.txt"

    if not policy_path.exists():
        policy_text = (
            "Brand tone guidelines:\n"
            "- Be polite and calm.\n"
            "- Be concise.\n"
            "- Acknowledge user feelings when negative emotion is present.\n"
            "- Do not make promises you cannot verify.\n"
            "- Avoid rude, sarcastic, or overly formal wording.\n"
        )
        policy_path.write_text(policy_text, encoding="utf-8")

    policy = policy_path.read_text(encoding="utf-8").strip()
    examples = examples_path.read_text(encoding="utf-8").strip()

    merged = policy + "\n\n" + examples
    merged_path.write_text(merged, encoding="utf-8")

    print(f"Saved merged knowledge to {merged_path}")


if __name__ == "__main__":
    main()