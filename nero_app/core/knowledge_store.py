from __future__ import annotations

import math
import re
from collections import Counter

from nero_app.core.schema import HistoricalMatch, MacroEvent


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


class LocalKnowledgeStore:
    def __init__(self, events: list[MacroEvent]) -> None:
        self.events = events
        self._documents = [_tokenize(f"{event.title} {event.narrative} {' '.join(event.tags)}") for event in events]
        self._idf = self._build_idf(self._documents)
        self._vectors = [self._vectorize(tokens) for tokens in self._documents]

    def search(self, query: str, asset: str, limit: int = 5) -> list[HistoricalMatch]:
        query_vector = self._vectorize(_tokenize(query))
        scored = [(_cosine(query_vector, vector), event) for vector, event in zip(self._vectors, self.events)]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            HistoricalMatch(
                event_id=event.event_id,
                event_date=event.event_date,
                title=event.title,
                similarity=round(float(similarity), 4),
                forward_bias=float(event.asset_bias.get(asset, 0.0)),
                tags=event.tags,
            )
            for similarity, event in scored[:limit]
        ]

    def _build_idf(self, documents: list[list[str]]) -> dict[str, float]:
        document_count = len(documents)
        frequencies = Counter(token for doc in documents for token in set(doc))
        return {
            token: math.log((document_count + 1) / (frequency + 1)) + 1
            for token, frequency in frequencies.items()
        }

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {token: count * self._idf.get(token, 1.0) for token, count in counts.items()}


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left).intersection(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))
