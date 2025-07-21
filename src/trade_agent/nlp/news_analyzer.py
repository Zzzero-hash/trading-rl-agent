"""
News analysis for financial markets.

This module provides news analysis capabilities for:
- News sentiment analysis
- Market impact assessment
- Event detection
- News aggregation
"""

from dataclasses import dataclass
from datetime import datetime

from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .text_processor import ProcessedText, TextProcessor


@dataclass
class NewsArticle:
    """Represents a news article."""

    title: str
    content: str
    source: str
    published_at: datetime
    url: str | None = None
    author: str | None = None
    category: str | None = None


@dataclass
class NewsAnalysis:
    """Result of news analysis."""

    article: NewsArticle
    sentiment: SentimentResult
    processed_text: ProcessedText
    market_impact: float
    relevance_score: float
    entities: list[str]


class NewsAnalyzer:
    """Analyzer for financial news articles."""

    def __init__(self) -> None:
        """Initialize the news analyzer."""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.text_processor = TextProcessor()

        # Market impact keywords (high impact)
        self.high_impact_keywords = {
            "earnings",
            "revenue",
            "profit",
            "loss",
            "guidance",
            "forecast",
            "merger",
            "acquisition",
            "ipo",
            "bankruptcy",
            "regulation",
            "fed",
            "interest rate",
            "inflation",
            "recession",
            "crisis",
        }

        # Medium impact keywords
        self.medium_impact_keywords = {
            "product",
            "launch",
            "partnership",
            "expansion",
            "restructuring",
            "layoff",
            "hire",
            "ceo",
            "management",
            "board",
            "dividend",
            "buyback",
            "stock split",
            "analyst",
            "rating",
            "target",
        }

    def analyze_article(self, article: NewsArticle) -> NewsAnalysis:
        """Analyze a news article.

        Args:
            article: NewsArticle to analyze

        Returns:
            NewsAnalysis with analysis results
        """
        # Combine title and content for analysis
        full_text = f"{article.title}. {article.content}"

        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(full_text)

        # Process text
        processed_text = self.text_processor.process_text(full_text)

        # Calculate market impact
        market_impact = self._calculate_market_impact(full_text)

        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(article, processed_text)

        # Extract entities
        entities = self._extract_entities(full_text)

        return NewsAnalysis(
            article=article,
            sentiment=sentiment,
            processed_text=processed_text,
            market_impact=market_impact,
            relevance_score=relevance_score,
            entities=entities,
        )

    def analyze_batch(self, articles: list[NewsArticle]) -> list[NewsAnalysis]:
        """Analyze multiple news articles.

        Args:
            articles: List of NewsArticle objects

        Returns:
            List of NewsAnalysis objects
        """
        return [self.analyze_article(article) for article in articles]

    def get_market_sentiment(self, analyses: list[NewsAnalysis]) -> dict[str, float]:
        """Get overall market sentiment from news analyses.

        Args:
            analyses: List of NewsAnalysis objects

        Returns:
            Dictionary with market sentiment metrics
        """
        if not analyses:
            return {
                "overall_sentiment": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "average_impact": 0.0,
                "high_impact_count": 0,
            }

        # Calculate weighted sentiment (weighted by market impact)
        total_weight = sum(analysis.market_impact for analysis in analyses)

        if total_weight == 0:
            weighted_sentiment = 0.0
        else:
            weighted_sentiment = (
                sum(analysis.sentiment.sentiment_score * analysis.market_impact for analysis in analyses) / total_weight
            )

        # Calculate sentiment ratios
        sentiment_counts: dict[str, int] = {}
        for analysis in analyses:
            label = analysis.sentiment.sentiment_label
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

        total_count = len(analyses)

        # Count high impact articles
        high_impact_count = sum(1 for analysis in analyses if analysis.market_impact > 0.7)

        return {
            "overall_sentiment": weighted_sentiment,
            "positive_ratio": sentiment_counts.get("positive", 0) / total_count,
            "negative_ratio": sentiment_counts.get("negative", 0) / total_count,
            "neutral_ratio": sentiment_counts.get("neutral", 0) / total_count,
            "average_impact": sum(analysis.market_impact for analysis in analyses) / total_count,
            "high_impact_count": high_impact_count,
        }

    def filter_relevant_articles(self, analyses: list[NewsAnalysis], min_relevance: float = 0.5) -> list[NewsAnalysis]:
        """Filter articles by relevance score.

        Args:
            analyses: List of NewsAnalysis objects
            min_relevance: Minimum relevance score

        Returns:
            List of relevant NewsAnalysis objects
        """
        return [analysis for analysis in analyses if analysis.relevance_score >= min_relevance]

    def get_high_impact_articles(self, analyses: list[NewsAnalysis], min_impact: float = 0.7) -> list[NewsAnalysis]:
        """Get articles with high market impact.

        Args:
            analyses: List of NewsAnalysis objects
            min_impact: Minimum market impact threshold

        Returns:
            List of high impact NewsAnalysis objects
        """
        return [analysis for analysis in analyses if analysis.market_impact >= min_impact]

    def _calculate_market_impact(self, text: str) -> float:
        """Calculate market impact score for text.

        Args:
            text: Text to analyze

        Returns:
            Market impact score (0.0 to 1.0)
        """
        text_lower = text.lower()

        # Count impact keywords
        high_impact_count = sum(1 for keyword in self.high_impact_keywords if keyword in text_lower)
        medium_impact_count = sum(1 for keyword in self.medium_impact_keywords if keyword in text_lower)

        # Calculate impact score
        impact_score = (high_impact_count * 0.8 + medium_impact_count * 0.4) / 10.0

        # Cap at 1.0
        return min(impact_score, 1.0)

    def _calculate_relevance_score(self, article: NewsArticle, processed_text: ProcessedText) -> float:
        """Calculate relevance score for article.

        Args:
            article: NewsArticle object
            processed_text: ProcessedText object

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0

        # Source credibility
        credible_sources = {"reuters", "bloomberg", "wsj", "ft", "cnbc", "marketwatch"}
        if article.source.lower() in credible_sources:
            score += 0.3

        # Content length (longer articles tend to be more relevant)
        content_length = len(article.content)
        if content_length > 500:
            score += 0.2
        elif content_length > 200:
            score += 0.1

        # Financial terms in content
        financial_terms = {
            "stock",
            "market",
            "trading",
            "investment",
            "financial",
            "earnings",
        }
        token_set = set(processed_text.tokens)
        financial_term_count = len(token_set.intersection(financial_terms))
        score += min(financial_term_count * 0.1, 0.3)

        # Recent articles get higher score
        if article.published_at:
            days_old = (datetime.now() - article.published_at).days
            if days_old <= 1:
                score += 0.2
            elif days_old <= 7:
                score += 0.1

        return min(score, 1.0)

    def _extract_entities(self, text: str) -> list[str]:
        """Extract financial entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of extracted entities
        """
        # Simple entity extraction - look for company names, tickers, etc.
        entities = []

        # Look for ticker patterns (e.g., AAPL, GOOGL)
        import re

        ticker_pattern = r"\b[A-Z]{2,5}\b"
        tickers = re.findall(ticker_pattern, text)
        entities.extend(tickers)

        # Look for currency pairs (e.g., EUR/USD, GBP/USD)
        currency_pattern = r"\b[A-Z]{3}/[A-Z]{3}\b"
        currencies = re.findall(currency_pattern, text)
        entities.extend(currencies)

        # Look for common financial terms
        financial_terms = {
            "federal reserve",
            "fed",
            "ecb",
            "boj",
            "boe",
            "treasury",
            "s&p 500",
            "nasdaq",
            "dow jones",
            "ftse",
            "nikkei",
        }

        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower:
                entities.append(term)

        return list(set(entities))  # Remove duplicates
