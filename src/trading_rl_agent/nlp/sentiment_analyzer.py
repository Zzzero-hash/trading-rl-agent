"""
Sentiment analysis for financial text.

This module provides sentiment analysis capabilities for:
- News articles
- Social media posts
- Financial reports
- Market commentary
"""

import re
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    text: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    keywords: list[str]
    entities: list[str]


class SentimentAnalyzer:
    """Basic sentiment analyzer for financial text."""

    def __init__(self) -> None:
        """Initialize the sentiment analyzer."""
        # Simple financial sentiment keywords
        self.positive_keywords = {
            "bullish",
            "positive",
            "growth",
            "profit",
            "gain",
            "rise",
            "increase",
            "strong",
            "excellent",
            "outperform",
            "beat",
            "surge",
            "rally",
            "soar",
            "jump",
            "climb",
            "advance",
            "improve",
            "recovery",
            "bounce",
            "rebound",
        }

        self.negative_keywords = {
            "bearish",
            "negative",
            "decline",
            "loss",
            "drop",
            "fall",
            "decrease",
            "weak",
            "poor",
            "underperform",
            "miss",
            "plunge",
            "crash",
            "tumble",
            "slump",
            "dip",
            "retreat",
            "worsen",
            "recession",
            "correction",
            "selloff",
        }

        self.neutral_keywords = {
            "stable",
            "steady",
            "maintain",
            "hold",
            "neutral",
            "unchanged",
            "flat",
            "sideways",
            "consolidate",
            "range",
            "support",
            "resistance",
        }

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with analysis
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Count sentiment keywords
        positive_count = self._count_keywords(processed_text, self.positive_keywords)
        negative_count = self._count_keywords(processed_text, self.negative_keywords)
        neutral_count = self._count_keywords(processed_text, self.neutral_keywords)

        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count + neutral_count

        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
            confidence = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            confidence = total_sentiment_words / len(processed_text.split()) if processed_text else 0.0

            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

        # Extract keywords and entities
        keywords = self._extract_keywords(processed_text)
        entities = self._extract_entities(processed_text)

        return SentimentResult(
            text=text,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects
        """
        return [self.analyze_sentiment(text) for text in texts]

    def get_sentiment_summary(self, results: list[SentimentResult]) -> dict[str, float]:
        """Get summary statistics for sentiment analysis results.

        Args:
            results: List of SentimentResult objects

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "average_sentiment": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "average_confidence": 0.0,
            }

        sentiment_scores = [r.sentiment_score for r in results]
        confidence_scores = [r.confidence for r in results]

        label_counts: dict[str, int] = {}
        for result in results:
            label_counts[result.sentiment_label] = label_counts.get(result.sentiment_label, 0) + 1

        total_count = len(results)

        return {
            "average_sentiment": sum(sentiment_scores) / len(sentiment_scores),
            "positive_ratio": label_counts.get("positive", 0) / total_count,
            "negative_ratio": label_counts.get("negative", 0) / total_count,
            "neutral_ratio": label_counts.get("neutral", 0) / total_count,
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
        }

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis.

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        return re.sub(r"\s+", " ", text).strip()

    def _count_keywords(self, text: str, keywords: set) -> int:
        """Count occurrences of keywords in text.

        Args:
            text: Preprocessed text
            keywords: Set of keywords to count

        Returns:
            Number of keyword occurrences
        """
        words = text.split()
        count = 0
        for word in words:
            if word in keywords:
                count += 1
        return count

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract important keywords from text.

        Args:
            text: Preprocessed text

        Returns:
            List of extracted keywords
        """
        all_keywords = self.positive_keywords | self.negative_keywords | self.neutral_keywords
        words = text.split()
        keywords = [word for word in words if word in all_keywords]
        return list(set(keywords))  # Remove duplicates

    def _extract_entities(self, text: str) -> list[str]:
        """Extract financial entities from text.

        Args:
            text: Preprocessed text

        Returns:
            List of extracted entities
        """
        # Simple entity extraction - look for common financial terms
        financial_entities = {
            "stock",
            "bond",
            "currency",
            "commodity",
            "index",
            "etf",
            "fund",
            "market",
            "exchange",
            "trading",
            "investment",
            "portfolio",
            "asset",
        }

        words = text.split()
        entities = [word for word in words if word in financial_entities]
        return list(set(entities))  # Remove duplicates
