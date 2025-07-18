"""
Forex-specific sentiment analysis and data collection.

This module provides specialized sentiment analysis for forex trading,
including economic calendar events, central bank announcements,
and currency-specific news sources.
"""

import numpy as np
import pandas as pd

from .sentiment import SentimentAnalyzer, SentimentConfig


class ForexSentimentData:
    """Forex-specific sentiment data collection and analysis."""

    def __init__(self, config: SentimentConfig | None = None) -> None:
        self.config = config or SentimentConfig()
        self.analyzer = SentimentAnalyzer(self.config)

        # Forex-specific news sources and keywords
        self.forex_news_sources = [
            "forexfactory.com",
            "investing.com",
            "fxstreet.com",
            "dailyfx.com",
            "marketwatch.com",
        ]

        # Economic calendar events that affect forex
        self.economic_events = {
            "NFP": "Non-Farm Payrolls",
            "CPI": "Consumer Price Index",
            "GDP": "Gross Domestic Product",
            "FOMC": "Federal Open Market Committee",
            "ECB": "European Central Bank",
            "BOE": "Bank of England",
            "BOJ": "Bank of Japan",
            "RBA": "Reserve Bank of Australia",
            "BOC": "Bank of Canada",
            "SNB": "Swiss National Bank",
        }

        # Currency-specific keywords for better sentiment analysis
        self.currency_keywords = {
            "USD": ["dollar", "greenback", "buck", "federal reserve", "fed"],
            "EUR": ["euro", "european union", "ecb", "european central bank"],
            "GBP": ["pound", "sterling", "british", "boe", "bank of england"],
            "JPY": ["yen", "japanese", "boj", "bank of japan"],
            "AUD": ["australian dollar", "aussie", "rba", "reserve bank of australia"],
            "CAD": ["canadian dollar", "loonie", "boc", "bank of canada"],
            "CHF": ["swiss franc", "franc", "snb", "swiss national bank"],
            "NZD": ["new zealand dollar", "kiwi", "rbnz", "reserve bank of new zealand"],
        }

    def get_forex_sentiment(self, currency_pair: str, days_back: int = 7) -> dict[str, float]:
        """Get comprehensive forex sentiment for a currency pair."""

        # Parse currency pair
        base_currency, quote_currency = self._parse_currency_pair(currency_pair)

        # Get sentiment for both currencies
        base_sentiment = self.analyzer.get_symbol_sentiment(base_currency, days_back)
        quote_sentiment = self.analyzer.get_symbol_sentiment(quote_currency, days_back)

        # Calculate relative sentiment (base vs quote)
        relative_sentiment = base_sentiment - quote_sentiment

        # Get economic calendar sentiment
        economic_sentiment = self._get_economic_calendar_sentiment(currency_pair, days_back)

        # Get central bank sentiment
        central_bank_sentiment = self._get_central_bank_sentiment(currency_pair, days_back)

        return {
            "base_currency_sentiment": base_sentiment,
            "quote_currency_sentiment": quote_sentiment,
            "relative_sentiment": relative_sentiment,
            "economic_calendar_sentiment": economic_sentiment,
            "central_bank_sentiment": central_bank_sentiment,
            "aggregate_sentiment": (base_sentiment + relative_sentiment + economic_sentiment + central_bank_sentiment)
            / 4,
        }

    def get_forex_sentiment_features(self, currency_pairs: list[str], days_back: int = 7) -> pd.DataFrame:
        """Get sentiment features for multiple currency pairs."""
        features = []

        for pair in currency_pairs:
            sentiment_data = self.get_forex_sentiment(pair, days_back)

            features.append({"currency_pair": pair, **sentiment_data})

        return pd.DataFrame(features)

    def _parse_currency_pair(self, currency_pair: str) -> tuple[str, str]:
        """Parse currency pair into base and quote currencies."""
        if len(currency_pair) == 6:  # Standard format like "EURUSD"
            return currency_pair[:3], currency_pair[3:]
        if len(currency_pair) == 7 and currency_pair[3] == "/":  # Format like "EUR/USD"
            return currency_pair[:3], currency_pair[4:]
        # Try to extract from common formats
        if "USD" in currency_pair:
            if currency_pair.startswith("USD"):
                return "USD", currency_pair[3:]
            return currency_pair.replace("USD", ""), "USD"
        # Default parsing
        return currency_pair[:3], currency_pair[3:]

    def _get_economic_calendar_sentiment(self, currency_pair: str, days_back: int) -> float:
        """Get sentiment based on economic calendar events."""
        # This would integrate with economic calendar APIs
        # For now, return a mock sentiment based on recent events

        # Mock economic calendar sentiment
        # In a real implementation, this would:
        # 1. Fetch economic calendar data
        # 2. Analyze event outcomes vs expectations
        # 3. Calculate sentiment based on surprises

        import random

        return random.uniform(-0.5, 0.5)

    def _get_central_bank_sentiment(self, currency_pair: str, days_back: int) -> float:
        """Get sentiment based on central bank announcements and policy."""
        base_currency, quote_currency = self._parse_currency_pair(currency_pair)

        # Get central bank sentiment for both currencies
        base_cb_sentiment = self._get_currency_central_bank_sentiment(base_currency)
        quote_cb_sentiment = self._get_currency_central_bank_sentiment(quote_currency)

        # Return relative central bank sentiment
        return base_cb_sentiment - quote_cb_sentiment

    def _get_currency_central_bank_sentiment(self, currency: str) -> float:
        """Get central bank sentiment for a specific currency."""
        # Central bank keywords and their typical sentiment impact
        cb_keywords = {
            "USD": ["federal reserve", "fed", "fomc", "jerome powell"],
            "EUR": ["ecb", "european central bank", "christine lagarde"],
            "GBP": ["boe", "bank of england", "andrew bailey"],
            "JPY": ["boj", "bank of japan", "kazuo ueda"],
            "AUD": ["rba", "reserve bank of australia", "philip lowe"],
            "CAD": ["boc", "bank of canada", "tiff macklem"],
            "CHF": ["snb", "swiss national bank", "thomas jordan"],
            "NZD": ["rbnz", "reserve bank of new zealand", "adrian orr"],
        }

        # Get keywords for this currency
        keywords = cb_keywords.get(currency, [currency])

        # Search for recent news about this central bank
        # For now, return mock sentiment
        import random

        return random.uniform(-0.3, 0.3)

    def get_currency_correlation_sentiment(self, currencies: list[str], days_back: int = 7) -> pd.DataFrame:
        """Get sentiment correlation matrix for multiple currencies."""
        sentiment_data = {}

        for currency in currencies:
            sentiment_data[currency] = self.analyzer.get_symbol_sentiment(currency, days_back)

        # Create correlation matrix
        df = pd.DataFrame([sentiment_data])
        return df.corr()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of sentiment scores."""
        # Get sentiment data for major currencies
        major_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        sentiment_data = {}

        for currency in major_currencies:
            sentiment_data[currency] = self.analyzer.get_symbol_sentiment(currency, days_back=7)

        df = pd.DataFrame([sentiment_data])
        return df.corr()

    def get_risk_sentiment(
        self,
        safe_havens: list[str] | None = None,
        risk_currencies: list[str] | None = None,
    ) -> float:
        """Get overall risk sentiment based on safe haven vs risk currency sentiment."""
        if safe_havens is None:
            safe_havens = ["USD", "JPY", "CHF"]
        if risk_currencies is None:
            risk_currencies = ["AUD", "NZD", "CAD"]

        # Get average sentiment for safe haven currencies
        safe_haven_sentiment = np.mean(
            [self.analyzer.get_symbol_sentiment(currency, days_back=7) for currency in safe_havens],
        )

        # Get average sentiment for risk currencies
        risk_currency_sentiment = np.mean(
            [self.analyzer.get_symbol_sentiment(currency, days_back=7) for currency in risk_currencies],
        )

        # Risk sentiment: positive when risk currencies outperform safe havens
        return float(risk_currency_sentiment - safe_haven_sentiment)
