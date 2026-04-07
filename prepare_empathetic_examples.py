from __future__ import annotations

import os
import random
from collections import defaultdict

from datasets import load_from_disk, load_dataset


def clean_text(s: str) -> str:
    if s is None:
        return ""
    return (
        s.replace("_comma_", ",")
        .replace("_period_", ".")
        .replace("_question_", "?")
        .replace("_exclamation_", "!")
        .replace(" ' ", "'")
        .strip()
    )


def main():
    random.seed(42)
    os.makedirs("knowledge", exist_ok=True)

    # 如果你之前 save_to_disk 了，就优先读本地
    local_path = "./data/empathetic_dialogues"
    if os.path.exists(local_path):
        ds = load_from_disk(local_path)
    else:
        ds = load_dataset("facebook/empathetic_dialogues")

    rows = ds["train"]

    conversations = defaultdict(list)
    for row in rows:
        conv_id = row["conv_id"]
        utterance_idx = int(row["utterance_idx"])
        utterance = clean_text(row["utterance"])
        conversations[conv_id].append((utterance_idx, utterance))

    pairs = []
    for conv_id, items in conversations.items():
        items = sorted(items, key=lambda x: x[0])

        for i in range(len(items) - 1):
            user_post = items[i][1]
            reply = items[i + 1][1]

            if len(user_post.split()) < 4 or len(reply.split()) < 3:
                continue
            if len(user_post.split()) > 40 or len(reply.split()) > 40:
                continue

            pairs.append((user_post, reply))

    random.shuffle(pairs)
    pairs = pairs[:300]

    out_path = "knowledge/empathetic_examples.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (user_post, reply) in enumerate(pairs, start=1):
            block = (
                f"[Example {idx}]\n"
                f"User post: {user_post}\n"
                f"Empathetic reply: {reply}\n\n"
            )
            f.write(block)

    print(f"Saved {len(pairs)} examples to {out_path}")


if __name__ == "__main__":
    main()