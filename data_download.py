from datasets import load_dataset

tweet_eval = load_dataset("cardiffnlp/tweet_eval", "sentiment")

empathetic = load_dataset(
    "facebook/empathetic_dialogues",
    trust_remote_code=True
)

tweet_eval.save_to_disk("./data/tweet_eval_sentiment")
empathetic.save_to_disk("./data/empathetic_dialogues")
