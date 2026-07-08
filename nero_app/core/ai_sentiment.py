from __future__ import annotations

from dataclasses import dataclass
import json
import re

import requests

from nero_app.core.news_feed import NewsItem


POSITIVE_WORDS = {"surge", "jump", "rise", "gain", "record high", "breakout", "rally", "strong", "bullish", "improves"}
NEGATIVE_WORDS = {"selloff", "crash", "drop", "collapse", "weak", "bearish", "falls", "risk", "caution", "slumps"}


@dataclass(frozen=True)
class SentimentResult:
    overall_sentiment: str
    sentiment_score: int
    summary: str
    source: str


def analyze_news_sentiment(
    headlines: list[NewsItem],
    asset: str,
    gemini_api_key: str = "",
    timeout_seconds: int = 20,
) -> SentimentResult:
    if not headlines:
        return SentimentResult("Neutral", 0, "No headlines were available for sentiment analysis.", "local")
    if gemini_api_key.strip():
        try:
            return _analyze_with_gemini(headlines, asset, gemini_api_key.strip(), timeout_seconds)
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError):
            return _local_sentiment(headlines, asset, source="local fallback after Gemini error")
    return _local_sentiment(headlines, asset, source="local")


def _analyze_with_gemini(
    headlines: list[NewsItem],
    asset: str,
    api_key: str,
    timeout_seconds: int,
) -> SentimentResult:
    news_text = "\n".join(
        f"[{item.source}] [Tags: {', '.join(item.tags) or 'None'}] {item.title}\nLink: {item.link}"
        for item in headlines[:12]
    )
    prompt = f"""
You are an expert quantitative financial analyst. Analyze these recent market news headlines for {asset}.
Return strict JSON only with keys:
- overall_sentiment: one of Bullish, Bearish, Neutral
- sentiment_score: integer from -10 to 10
- summary: brief 2-3 sentence explanation

News:
{news_text}
"""
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        params={"key": api_key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    cleaned = _strip_markdown_json(text)
    data = json.loads(cleaned)
    score = int(data.get("sentiment_score", 0))
    score = max(-10, min(10, score))
    sentiment = str(data.get("overall_sentiment", "Neutral")).title()
    if sentiment not in {"Bullish", "Bearish", "Neutral"}:
        sentiment = _label_from_score(score)
    return SentimentResult(sentiment, score, str(data.get("summary", "")), "Gemini")


def _local_sentiment(headlines: list[NewsItem], asset: str, source: str) -> SentimentResult:
    joined = " ".join(item.title.lower() for item in headlines)
    positive = sum(1 for word in POSITIVE_WORDS if word in joined)
    negative = sum(1 for word in NEGATIVE_WORDS if word in joined)
    score = max(-10, min(10, (positive - negative) * 2))
    sentiment = _label_from_score(score)
    tag_counts: dict[str, int] = {}
    for item in headlines:
        for tag in item.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top_tags = ", ".join(tag for tag, _ in sorted(tag_counts.items(), key=lambda pair: pair[1], reverse=True)[:4])
    summary = (
        f"Local sentiment for {asset} is {sentiment.lower()} with score {score}. "
        f"Dominant news tags: {top_tags or 'none'}."
    )
    return SentimentResult(sentiment, score, summary, source)


def _label_from_score(score: int) -> str:
    if score >= 3:
        return "Bullish"
    if score <= -3:
        return "Bearish"
    return "Neutral"


def _strip_markdown_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    if stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped.strip()
