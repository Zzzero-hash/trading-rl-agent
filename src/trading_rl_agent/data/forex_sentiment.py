"""
Asset-to-source mapping and Yahoo Finance scraping for forex pairs.
"""

import datetime
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class ForexSentimentData:
    pair: str
    score: float
    magnitude: float
    timestamp: datetime.datetime
    source: str
    raw_data: dict


def get_yahoo_finance_url_for_forex(pair: str) -> str:
    """Return Yahoo Finance news URL for a forex pair (e.g., 'EURUSD')."""
    # Yahoo uses 'EURUSD=X' for forex pairs
    return f"https://finance.yahoo.com/quote/{pair}=X/news?p={pair}=X"


def scrape_yahoo_finance_forex_headlines(
    pair: str,
    max_headlines: int = 15,
) -> list[str]:
    url = get_yahoo_finance_url_for_forex(pair)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    headlines = set()
    for tag in ["h3", "h2", "a"]:
        for item in soup.find_all(tag):
            text = item.get_text(strip=True)
            if text and len(text) > 10:
                headlines.add(text)
    if not headlines:
        raise RuntimeError(f"No headlines found for {pair} on Yahoo Finance.")
    return list(headlines)[:max_headlines]


def analyze_text_sentiment(text: str) -> float:
    """Analyze sentiment of a headline using VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return float(scores.get("compound", 0.0))


def get_forex_sentiment(pair: str) -> list[ForexSentimentData]:
    try:
        headlines = scrape_yahoo_finance_forex_headlines(pair)
    except Exception as e:
        now = datetime.datetime.now()
        return [
            ForexSentimentData(
                pair=pair,
                score=0.0,
                magnitude=0.0,
                timestamp=now,
                source="no_sentiment",
                raw_data={"error": str(e)},
            ),
        ]
    now = datetime.datetime.now()
    sentiment_data = []
    for i, headline in enumerate(headlines):
        score = analyze_text_sentiment(headline)
        sentiment_data.append(
            ForexSentimentData(
                pair=pair,
                score=score,
                magnitude=0.7,
                timestamp=now - datetime.timedelta(minutes=i * 10),
                source="yahoo_finance_scrape",
                raw_data={"headline": headline},
            ),
        )
    return sentiment_data


def get_all_forex_sentiment(pairs: list[str]) -> dict[str, list[ForexSentimentData]]:
    return {pair: get_forex_sentiment(pair) for pair in pairs}
