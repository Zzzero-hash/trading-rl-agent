"""
Natural Language Processing module for the trading RL agent.

This module provides NLP capabilities for:
- Sentiment analysis of news and social media
- Text preprocessing and feature extraction
- Named entity recognition for financial entities
- Topic modeling for market analysis
"""

from .news_analyzer import NewsAnalysis, NewsAnalyzer, NewsArticle
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .text_processor import ProcessedText, TextProcessor

__all__ = [
    "NewsAnalysis",
    "NewsAnalyzer",
    "NewsArticle",
    "ProcessedText",
    "SentimentAnalyzer",
    "SentimentResult",
    "TextProcessor",
]
