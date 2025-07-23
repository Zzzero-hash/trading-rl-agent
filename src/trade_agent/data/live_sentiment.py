"""
Live sentiment scraper for real-time trading.
"""

import asyncio
import random
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup

from .sentiment import SentimentData, SentimentProvider


class LiveSentimentScraper(SentimentProvider):
    """Custom live sentiment scraper for real-time trading."""

    def __init__(self, update_interval: int = 60) -> None:
        super().__init__()
        self.update_interval = update_interval
        self.session: aiohttp.ClientSession | None = None
        self.last_update: dict[str, datetime] = {}

        # Custom scraping sources
        self.news_sources = {
            "yahoo_finance": "https://finance.yahoo.com/quote/{symbol}/news",
            "marketwatch": "https://www.marketwatch.com/investing/stock/{symbol}",
            "seeking_alpha": "https://seekingalpha.com/symbol/{symbol}",
            "investing": "https://www.investing.com/equities/{symbol}",
            "reuters": "https://www.reuters.com/markets/companies/{symbol}",
        }

        # Social media sources (if you have API access)
        self.social_sources = {
            "twitter": "https://twitter.com/search?q={symbol}&src=typed_query&f=live",
            "reddit": "https://www.reddit.com/r/wallstreetbets/search/?q={symbol}&restrict_sr=1&t=day",
        }

    async def start_session(self) -> None:
        """Start async session for scraping."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )

    async def close_session(self) -> None:
        """Close async session."""
        if self.session:
            await self.session.close()

    async def get_live_sentiment(
        self,
        symbol: str,
        force_update: bool = False
    ) -> SentimentData:
        """
        Get live sentiment for a symbol.

        Args:
            symbol: Trading symbol
            force_update: Force update even if recently cached

        Returns:
            Latest sentiment data
        """

        # Check if we need to update
        if not force_update and self._should_use_cache(symbol):
            return self._get_cached_sentiment(symbol)

        await self.start_session()

        try:
            # Collect sentiment from multiple sources
            sentiment_scores = []

            # News sentiment
            news_sentiment = await self._scrape_news_sentiment(symbol)
            if news_sentiment:
                sentiment_scores.extend(news_sentiment)

            # Social sentiment
            social_sentiment = await self._scrape_social_sentiment(symbol)
            if social_sentiment:
                sentiment_scores.extend(social_sentiment)

            # Calculate aggregated sentiment
            if sentiment_scores:
                avg_score = sum(s.score for s in sentiment_scores) / len(sentiment_scores)
                avg_magnitude = sum(s.magnitude for s in sentiment_scores) / len(sentiment_scores)
                sources = list({s.source for s in sentiment_scores})
            else:
                avg_score = 0.0
                avg_magnitude = 0.0
                sources = ["no_data"]

            # Create sentiment data
            sentiment_data = SentimentData(
                symbol=symbol,
                score=avg_score,
                magnitude=avg_magnitude,
                timestamp=datetime.now(),
                source=",".join(sources),
                raw_data={"scores": sentiment_scores}
            )

            # Cache the result
            self._cache_sentiment(symbol, sentiment_data)

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Error getting live sentiment for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)

    async def _scrape_news_sentiment(self, symbol: str) -> list[SentimentData]:
        """Scrape news sentiment from multiple sources."""
        sentiment_data: list[SentimentData] = []

        if self.session is None:
            return sentiment_data

        for source_name, url_template in self.news_sources.items():
            try:
                url = url_template.format(symbol=symbol)
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        headlines = self._extract_headlines(html, source_name)

                        for headline in headlines[:5]:  # Limit to 5 headlines per source
                            score = self._analyze_text_sentiment(headline)
                            magnitude = self._calculate_magnitude(headline)

                            sentiment_data.append(SentimentData(
                                symbol=symbol,
                                score=score,
                                magnitude=magnitude,
                                timestamp=datetime.now(),
                                source=f"news_{source_name}",
                                raw_data={"headline": headline, "source": source_name}
                            ))

                        # Add small delay to be respectful
                        await asyncio.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                self.logger.debug(f"Failed to scrape {source_name} for {symbol}: {e}")
                continue

        return sentiment_data

    async def _scrape_social_sentiment(self, _symbol: str) -> list[SentimentData]:
        """Scrape social media sentiment."""
        # This is a placeholder - you'd need proper API access for social media
        # For now, return empty list
        return []

    def _extract_headlines(self, html: str, source: str) -> list[str]:
        """Extract headlines from HTML based on source."""
        soup = BeautifulSoup(html, "html.parser")
        headlines = []

        # Source-specific selectors
        selectors = {
            "yahoo_finance": ["h3", "h4", ".headline", ".title"],
            "marketwatch": ["h1", "h2", "h3", ".headline"],
            "seeking_alpha": ["h1", "h2", "h3", ".article-title"],
            "investing": ["h1", "h2", "h3", ".title"],
            "reuters": ["h1", "h2", "h3", ".article-heading"],
        }

        source_selectors = selectors.get(source, ["h1", "h2", "h3"])

        for selector in source_selectors:
            elements = soup.find_all(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 10 and len(text) < 200:
                    headlines.append(text)

        return headlines[:10]  # Limit to 10 headlines

    def _should_use_cache(self, symbol: str) -> bool:
        """Check if we should use cached data."""
        if symbol not in self.last_update:
            return False

        time_since_update = datetime.now() - self.last_update[symbol]
        return bool(time_since_update.seconds < self.update_interval)

    def _cache_sentiment(self, symbol: str, _sentiment_data: SentimentData) -> None:
        """Cache sentiment data."""
        self.last_update[symbol] = datetime.now()
        # You could also save to a file or database here

    def _get_cached_sentiment(self, symbol: str) -> SentimentData:
        """Get cached sentiment data."""
        # Return the last known sentiment
        return SentimentData(
            symbol=symbol,
            score=0.0,
            magnitude=0.0,
            timestamp=datetime.now(),
            source="cached",
            raw_data={"cached": True}
        )

    def _get_fallback_sentiment(self, symbol: str) -> SentimentData:
        """Get fallback sentiment when scraping fails."""
        return SentimentData(
            symbol=symbol,
            score=0.0,
            magnitude=0.0,
            timestamp=datetime.now(),
            source="fallback",
            raw_data={"fallback": True}
        )
