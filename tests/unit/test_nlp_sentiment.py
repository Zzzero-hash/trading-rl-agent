"""
Tests for the NLP module.
"""

from datetime import datetime

from trade_agent.nlp import (
    NewsAnalysis,
    NewsAnalyzer,
    NewsArticle,
    ProcessedText,
    SentimentAnalyzer,
    SentimentResult,
    TextProcessor,
)


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""

    def test_initialization(self):
        """Test SentimentAnalyzer initialization."""
        analyzer = SentimentAnalyzer()

        assert len(analyzer.positive_keywords) > 0
        assert len(analyzer.negative_keywords) > 0
        assert len(analyzer.neutral_keywords) > 0

    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis with positive text."""
        analyzer = SentimentAnalyzer()

        result = analyzer.analyze_sentiment("The stock market is showing strong growth and positive momentum.")

        assert isinstance(result, SentimentResult)
        assert result.text == "The stock market is showing strong growth and positive momentum."
        assert result.sentiment_score > 0
        assert result.sentiment_label == "positive"
        assert result.confidence > 0
        assert "growth" in result.keywords
        assert "strong" in result.keywords

    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis with negative text."""
        analyzer = SentimentAnalyzer()

        result = analyzer.analyze_sentiment("The market is experiencing a significant decline and poor performance.")

        assert isinstance(result, SentimentResult)
        assert result.sentiment_score < 0
        assert result.sentiment_label == "negative"
        assert result.confidence > 0
        assert "decline" in result.keywords
        assert "poor" in result.keywords

    def test_analyze_sentiment_neutral(self):
        """Test sentiment analysis with neutral text."""
        analyzer = SentimentAnalyzer()

        result = analyzer.analyze_sentiment("The market remains stable with no significant changes.")

        assert isinstance(result, SentimentResult)
        assert abs(result.sentiment_score) < 0.1
        assert result.sentiment_label == "neutral"
        assert "stable" in result.keywords

    def test_analyze_sentiment_no_keywords(self):
        """Test sentiment analysis with text containing no sentiment keywords."""
        analyzer = SentimentAnalyzer()

        result = analyzer.analyze_sentiment("The weather is sunny today.")

        assert isinstance(result, SentimentResult)
        assert result.sentiment_score == 0.0
        assert result.sentiment_label == "neutral"
        assert result.confidence == 0.0
        assert len(result.keywords) == 0

    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        analyzer = SentimentAnalyzer()

        texts = [
            "Strong growth in the market.",
            "Significant decline in performance.",
            "Stable market conditions.",
        ]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[0].sentiment_label == "positive"
        assert results[1].sentiment_label == "negative"
        assert results[2].sentiment_label == "neutral"

    def test_get_sentiment_summary(self):
        """Test sentiment summary generation."""
        analyzer = SentimentAnalyzer()

        texts = [
            "Strong growth in the market.",
            "Significant decline in performance.",
            "Stable market conditions.",
            "Excellent performance results.",
        ]

        results = analyzer.analyze_batch(texts)
        summary = analyzer.get_sentiment_summary(results)

        assert "average_sentiment" in summary
        assert "positive_ratio" in summary
        assert "negative_ratio" in summary
        assert "neutral_ratio" in summary
        assert "average_confidence" in summary
        assert summary["positive_ratio"] == 0.5  # 2 out of 4
        assert summary["negative_ratio"] == 0.25  # 1 out of 4
        assert summary["neutral_ratio"] == 0.25  # 1 out of 4


class TestTextProcessor:
    """Test suite for TextProcessor."""

    def test_initialization(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor()

        assert len(processor.financial_abbreviations) > 0
        assert len(processor.stop_words) > 0

    def test_process_text(self):
        """Test text processing."""
        processor = TextProcessor()

        text = "The corp. reported strong earnings growth of 25%."
        result = processor.process_text(text)

        assert isinstance(result, ProcessedText)
        assert result.original_text == text
        assert "corporation" in result.cleaned_text  # Abbreviation expanded
        assert "earnings" in result.tokens
        assert "growth" in result.tokens
        assert result.features["has_numbers"] is True
        assert result.features["has_percentages"] is True

    def test_process_text_with_stop_words(self):
        """Test text processing with stop word removal."""
        processor = TextProcessor()

        text = "The market is showing strong growth and positive momentum."
        result = processor.process_text(text, remove_stop_words=True)

        # Stop words should be removed
        assert "the" not in result.tokens
        assert "is" not in result.tokens
        assert "and" not in result.tokens
        assert "showing" in result.tokens
        assert "strong" in result.tokens
        assert "growth" in result.tokens

    def test_process_text_without_stop_words(self):
        """Test text processing without stop word removal."""
        processor = TextProcessor()

        text = "The market is showing strong growth."
        result = processor.process_text(text, remove_stop_words=False)

        # Stop words should be included
        assert "the" in result.tokens
        assert "is" in result.tokens
        assert "showing" in result.tokens
        assert "strong" in result.tokens
        assert "growth" in result.tokens

    def test_clean_text(self):
        """Test text cleaning."""
        processor = TextProcessor()

        text = "The corp. reported earnings of $1.2B. Visit http://example.com for more info."
        cleaned = processor._clean_text(text)

        assert "corporation" in cleaned  # Abbreviation expanded
        assert "http://example.com" not in cleaned  # URL removed
        assert "$" in cleaned  # Currency symbol preserved
        assert "1.2b" in cleaned  # Numbers preserved (lowercase after processing)

    def test_extract_features(self):
        """Test feature extraction."""
        processor = TextProcessor()

        tokens = ["earnings", "growth", "25%", "$1.2B", "market"]
        features = processor._extract_features(tokens)

        assert features["token_count"] == 5
        assert features["unique_tokens"] == 5
        assert features["has_numbers"] is True
        assert features["has_currency"] is True
        assert features["has_percentages"] is True

    def test_process_batch(self):
        """Test batch text processing."""
        processor = TextProcessor()

        texts = [
            "Strong earnings growth.",
            "Market decline reported.",
            "Stable performance.",
        ]

        results = processor.process_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, ProcessedText) for r in results)
        assert "earnings" in results[0].tokens
        assert "decline" in results[1].tokens
        assert "stable" in results[2].tokens


class TestNewsAnalyzer:
    """Test suite for NewsAnalyzer."""

    def test_initialization(self):
        """Test NewsAnalyzer initialization."""
        analyzer = NewsAnalyzer()

        assert len(analyzer.high_impact_keywords) > 0
        assert len(analyzer.medium_impact_keywords) > 0
        assert analyzer.sentiment_analyzer is not None
        assert analyzer.text_processor is not None

    def test_analyze_article(self):
        """Test article analysis."""
        analyzer = NewsAnalyzer()

        article = NewsArticle(
            title="Company Reports Strong Earnings Growth",
            content="The company reported excellent quarterly earnings with 25% growth in revenue.",
            source="Reuters",
            published_at=datetime.now(),
        )

        analysis = analyzer.analyze_article(article)

        assert isinstance(analysis, NewsAnalysis)
        assert analysis.article == article
        assert analysis.sentiment.sentiment_label == "positive"
        assert analysis.market_impact > 0
        assert analysis.relevance_score > 0
        assert len(analysis.entities) >= 0

    def test_analyze_batch(self):
        """Test batch article analysis."""
        analyzer = NewsAnalyzer()

        articles = [
            NewsArticle(
                title="Strong Earnings Report",
                content="Company reports excellent quarterly results.",
                source="Bloomberg",
                published_at=datetime.now(),
            ),
            NewsArticle(
                title="Market Decline",
                content="Significant market downturn reported.",
                source="Reuters",
                published_at=datetime.now(),
            ),
        ]

        analyses = analyzer.analyze_batch(articles)

        assert len(analyses) == 2
        assert all(isinstance(a, NewsAnalysis) for a in analyses)
        assert analyses[0].sentiment.sentiment_label == "positive"
        assert analyses[1].sentiment.sentiment_label == "negative"

    def test_get_market_sentiment(self):
        """Test market sentiment calculation."""
        analyzer = NewsAnalyzer()

        articles = [
            NewsArticle(
                title="Positive News",
                content="Strong growth reported.",
                source="Bloomberg",
                published_at=datetime.now(),
            ),
            NewsArticle(
                title="Negative News",
                content="Decline in performance.",
                source="Reuters",
                published_at=datetime.now(),
            ),
            NewsArticle(
                title="Neutral News",
                content="Stable market conditions.",
                source="WSJ",
                published_at=datetime.now(),
            ),
        ]

        analyses = analyzer.analyze_batch(articles)
        sentiment = analyzer.get_market_sentiment(analyses)

        assert "overall_sentiment" in sentiment
        assert "positive_ratio" in sentiment
        assert "negative_ratio" in sentiment
        assert "neutral_ratio" in sentiment
        assert "average_impact" in sentiment
        assert "high_impact_count" in sentiment

    def test_filter_relevant_articles(self):
        """Test article filtering by relevance."""
        analyzer = NewsAnalyzer()

        articles = [
            NewsArticle(
                title="Financial News",
                content="Stock market trading shows strong performance with excellent earnings growth.",
                source="Bloomberg",
                published_at=datetime.now(),
            ),
            NewsArticle(
                title="Weather Report",
                content="Sunny weather expected today.",
                source="Weather Channel",
                published_at=datetime.now(),
            ),
        ]

        analyses = analyzer.analyze_batch(articles)
        relevant = analyzer.filter_relevant_articles(analyses, min_relevance=0.5)

        # First article should be more relevant (financial content)
        assert len(relevant) >= 1
        assert relevant[0].article.title == "Financial News"

    def test_get_high_impact_articles(self):
        """Test filtering high impact articles."""
        analyzer = NewsAnalyzer()

        articles = [
            NewsArticle(
                title="Earnings Report",
                content="Company reports strong earnings and revenue growth.",
                source="Bloomberg",
                published_at=datetime.now(),
            ),
            NewsArticle(
                title="Weather Update",
                content="Nice weather today.",
                source="Weather Channel",
                published_at=datetime.now(),
            ),
        ]

        analyses = analyzer.analyze_batch(articles)
        high_impact = analyzer.get_high_impact_articles(analyses, min_impact=0.5)

        # First article should have higher impact (earnings keywords)
        assert len(high_impact) >= 1
        assert high_impact[0].article.title == "Earnings Report"
