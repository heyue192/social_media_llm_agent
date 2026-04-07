from __future__ import annotations

import argparse
import json

from src.agent_chain import SocialMediaAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    agent = SocialMediaAgent(classifier_dir=args.classifier_dir)

    print("Type a social media post. Enter 'quit' to stop.")
    while True:
        post = input("\nPost> ").strip()
        if post.lower() in {"quit", "exit"}:
            break
        result = agent.run(post)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
