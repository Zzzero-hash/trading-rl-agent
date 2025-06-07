"""Sentiment Analysis Module for Trading RL Agent.

This module provides sentiment analysis capabilities for financial markets,
integrating news sentiment and social media sentiment to enhance trading decisions.
It supports real-time sentiment fetching and historical sentiment analysis.

Example usage:
>>> analyzer = SentimentAnalyzer()
>>> sentiment_score = analyzer.get_symbol_sentiment('AAPL')
>>> news_sentiment = analyzer.analyze_news_sentiment('AAPL', days_back=7)
"""

import logging
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import requests
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Global sentiment cache for backward compatibility
sentiment = {}

@dataclass
class SentimentData:
    """Structured sentiment data for a financial symbol."""
    symbol: str
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    magnitude: float  # 0.0 to 1.0 (confidence level)
    timestamp: datetime.datetime
    source: str  # 'news', 'social', 'analyst', etc.
    raw_data: Optional[Dict[str, Any]] = None


class SentimentProvider(ABC):
    """Abstract base class for sentiment data providers."""
    
    @abstractmethod
    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> List[SentimentData]:
        """Fetch sentiment data for a symbol."""
        pass


class NewsSentimentProvider(SentimentProvider):
    """News-based sentiment provider using financial news APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        
    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> List[SentimentData]:
        """Fetch news sentiment for a symbol."""
        if not self.api_key:
            # Return mock data for testing
            return self._get_mock_news_sentiment(symbol, days_back)
            
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days_back)
            
            params = {
                'q': f'"{symbol}" OR "{self._symbol_to_company(symbol)}"',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            sentiment_data = []
            
            for article in articles[:10]:  # Limit to 10 most relevant articles
                score = self._analyze_text_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                sentiment_data.append(SentimentData(
                    symbol=symbol,
                    score=score,
                    magnitude=0.7,  # Default confidence for news
                    timestamp=datetime.datetime.now(),
                    source='news',
                    raw_data=article
                ))
                
            return sentiment_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch news sentiment for {symbol}: {e}")
            return self._get_mock_news_sentiment(symbol, days_back)
    
    def _symbol_to_company(self, symbol: str) -> str:
        """Convert stock symbol to company name for better news search."""
        company_map = {
            'AAPL': 'Apple Inc',
            'GOOGL': 'Google Alphabet',
            'TSLA': 'Tesla',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'META': 'Meta Facebook',
            'NVDA': 'Nvidia'
        }
        return company_map.get(symbol, symbol)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple sentiment analysis using keyword matching."""
        positive_words = ['bullish', 'growth', 'profit', 'gain', 'up', 'rise', 'strong', 'beat', 'outperform']
        negative_words = ['bearish', 'loss', 'decline', 'down', 'fall', 'weak', 'miss', 'underperform', 'drop']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, sentiment_score))
    
    def _get_mock_news_sentiment(self, symbol: str, days_back: int) -> List[SentimentData]:
        """Generate mock news sentiment data for testing."""
        import random
        random.seed(hash(symbol))  # Consistent mock data per symbol
        
        sentiment_data = []
        for i in range(min(5, days_back)):
            score = random.uniform(-0.5, 0.8)  # Slightly positive bias
            sentiment_data.append(SentimentData(
                symbol=symbol,
                score=score,
                magnitude=random.uniform(0.5, 0.9),
                timestamp=datetime.datetime.now() - datetime.timedelta(days=i),
                source='news_mock',
                raw_data={'mock': True}
            ))
        return sentiment_data


class SocialSentimentProvider(SentimentProvider):
    """Social media sentiment provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
    def fetch_sentiment(self, symbol: str, days_back: int = 1) -> List[SentimentData]:
        """Fetch social media sentiment for a symbol."""
        # For now, return mock data - can be extended with Twitter/Reddit APIs
        return self._get_mock_social_sentiment(symbol, days_back)
    
    def _get_mock_social_sentiment(self, symbol: str, days_back: int) -> List[SentimentData]:
        """Generate mock social sentiment data."""
        import random
        random.seed(hash(symbol + 'social'))
        
        sentiment_data = []
        for i in range(min(3, days_back)):
            score = random.uniform(-0.8, 0.6)  # More volatile than news
            sentiment_data.append(SentimentData(
                symbol=symbol,
                score=score,
                magnitude=random.uniform(0.3, 0.8),
                timestamp=datetime.datetime.now() - datetime.timedelta(days=i),
                source='social_mock',
                raw_data={'mock': True}
            ))
        return sentiment_data


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    enable_news: bool = True
    enable_social: bool = True
    news_api_key: Optional[str] = None
    social_api_key: Optional[str] = None
    cache_duration_hours: int = 1
    sentiment_weight: float = 0.2  # Weight in final feature vector


class SentimentAnalyzer:
    """Main sentiment analysis coordinator."""
    
    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self.providers: List[SentimentProvider] = []
        self.sentiment_cache: Dict[str, List[SentimentData]] = {}
        
        if self.config.enable_news:
            self.providers.append(NewsSentimentProvider(self.config.news_api_key))
        if self.config.enable_social:
            self.providers.append(SocialSentimentProvider(self.config.social_api_key))
    
    def get_symbol_sentiment(self, symbol: str, days_back: int = 1) -> float:
        """Get aggregated sentiment score for a symbol."""
        sentiment_data = self.fetch_all_sentiment(symbol, days_back)
        
        if not sentiment_data:
            return 0.0
            
        # Weight by magnitude and recency
        total_weighted_score = 0.0
        total_weight = 0.0
        
        now = datetime.datetime.now()
        for data in sentiment_data:
            # Recency weight (more recent = higher weight)
            hours_old = (now - data.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1.0 - (hours_old / (days_back * 24)))
            
            weight = data.magnitude * recency_weight
            total_weighted_score += data.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def fetch_all_sentiment(self, symbol: str, days_back: int = 1) -> List[SentimentData]:
        """Fetch sentiment from all providers."""
        cache_key = f"{symbol}_{days_back}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached_data = self.sentiment_cache[cache_key]
            if cached_data and self._is_cache_valid(cached_data[0].timestamp):
                return cached_data
        
        # Fetch fresh data
        all_sentiment = []
        for provider in self.providers:
            try:
                sentiment_data = provider.fetch_sentiment(symbol, days_back)
                all_sentiment.extend(sentiment_data)
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")
        
        # Cache results
        self.sentiment_cache[cache_key] = all_sentiment
        
        # Update global sentiment dictionary for backward compatibility
        if all_sentiment:
            avg_score = sum(d.score for d in all_sentiment) / len(all_sentiment)
            avg_magnitude = sum(d.magnitude for d in all_sentiment) / len(all_sentiment)
            sentiment[symbol] = {
                'score': avg_score,
                'magnitude': avg_magnitude,
                'timestamp': datetime.datetime.now().isoformat(),
                'source': 'aggregated'
            }
        
        return all_sentiment
    
    def _is_cache_valid(self, timestamp: datetime.datetime) -> bool:
        """Check if cached data is still valid."""
        age_hours = (datetime.datetime.now() - timestamp).total_seconds() / 3600
        return age_hours < self.config.cache_duration_hours
    
    def get_sentiment_features(self, symbols: List[str], days_back: int = 1) -> Dict[str, float]:
        """Get sentiment features for multiple symbols suitable for ML models."""
        features = {}
        for symbol in symbols:
            sentiment_score = self.get_symbol_sentiment(symbol, days_back)
            features[f'sentiment_{symbol}'] = sentiment_score
            features[f'sentiment_{symbol}_abs'] = abs(sentiment_score)  # Magnitude feature
        return features
    
    def update_sentiment_cache(self, symbol: str, days_back: int = 1):
        """Manually update sentiment cache for a symbol."""
        self.fetch_all_sentiment(symbol, days_back)


# Default global analyzer instance
_default_analyzer = SentimentAnalyzer()

def get_sentiment_score(symbol: str, days_back: int = 1) -> float:
    """Convenience function to get sentiment score."""
    return _default_analyzer.get_symbol_sentiment(symbol, days_back)

def update_sentiment(symbol: str, days_back: int = 1):
    """Convenience function to update sentiment."""
    _default_analyzer.update_sentiment_cache(symbol, days_back)