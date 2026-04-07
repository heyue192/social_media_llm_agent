from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, knowledge_path: str):
        path = Path(knowledge_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")

        raw_text = path.read_text(encoding="utf-8")
        self.chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 2) -> List[RetrievedChunk]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        indices = scores.argsort()[::-1][:top_k]
        return [RetrievedChunk(text=self.chunks[i], score=float(scores[i])) for i in indices]
