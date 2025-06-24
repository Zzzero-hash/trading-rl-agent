#!/usr/bin/env python3
"""
Build combined datasets from synthetic scenarios and real market data.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print(
        "Warning: 'datasets' package not found. HuggingFace sentiment will be skipped."
    )

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print(
        "Warning: 'vaderSentiment' package not found. "
        "Twitter sentiment will be skipped."
    )

import feedparser

# Ensure root dir in path for imports
root = Path(__file__).parent
sys.path.insert(0, str(root))

# Synthetic data generator
from generate_sample_data import (
    add_sentiment_features,
    add_technical_indicators,
    generate_labels,
    generate_sample_price_data,
)

# Historical data fetcher
from src.data.historical import fetch_historical_data


def generate_synthetic(symbols, days, volatility, scenarios_per_symbol):
    dfs = []
    for symbol in symbols:
        for i in range(scenarios_per_symbol):
            df = generate_sample_price_data(
                symbol=symbol,
                days=days,
                start_price=100 + i * 10,
                volatility=volatility,
            )
            df = add_technical_indicators(df)
            df = add_sentiment_features(df)
            df = generate_labels(df)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def fetch_real(symbols, start, end, timestep):
    dfs = []
    for symbol in symbols:
        df = fetch_historical_data(symbol, start, end, timestep)
        if df.empty:
            continue
        df = add_technical_indicators(df)
        df = add_sentiment_features(df)
        df = generate_labels(df)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def add_hf_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical sentiment from Hugging Face financial_sentiment dataset."""
    if not HAS_DATASETS:
        print("Warning: datasets package not available, skipping HF sentiment")
        df["hf_sentiment"] = 0.0
        return df

    # Load dataset once
    ds = load_dataset("financial_sentiment", split="train")
    hf_df = pd.DataFrame(list(ds))
    # Assume dataset has 'symbol', 'date', 'sentiment_score' fields
    hf_df = hf_df.rename(
        columns={"date": "timestamp", "sentiment_score": "hf_sentiment"}
    )
    hf_df["timestamp"] = pd.to_datetime(hf_df["timestamp"])
    # Merge and fill missing
    df = df.merge(
        hf_df[["timestamp", "symbol", "hf_sentiment"]],
        on=["timestamp", "symbol"],
        how="left",
    )
    df["hf_sentiment"] = df["hf_sentiment"].fillna(0.0)
    return df


def add_twitter_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch tweets via snscrape and analyze with VADER, per symbol per day."""
    if not HAS_VADER:
        print(
            "Warning: vaderSentiment package not available, skipping Twitter sentiment"
        )
        df["twitter_sentiment"] = 0.0
        return df

    analyzer = SentimentIntensityAnalyzer()
    records = []
    for symbol in df["symbol"].unique():
        for date in df["timestamp"].dt.date.unique():
            # Build snscrape command
            since = date.isoformat()
            until = (date + datetime.timedelta(days=1)).isoformat()
            query = f"{symbol} since:{since} until:{until}"
            try:
                output = subprocess.check_output(
                    ["snscrape", "--jsonl", f"twitter-search {query}"],
                    shell=False,
                    text=True,
                )
                tweets = [json.loads(line) for line in output.splitlines() if line]
                scores = [
                    analyzer.polarity_scores(t.get("content", ""))["compound"]
                    for t in tweets
                ]
                avg_score = sum(scores) / len(scores) if scores else 0.0
            except Exception:
                avg_score = 0.0
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp(date),
                    "twitter_sentiment": avg_score,
                }
            )
    tw_df = pd.DataFrame(records)
    df = df.merge(tw_df, on=["timestamp", "symbol"], how="left")
    df["twitter_sentiment"] = df["twitter_sentiment"].fillna(0.0)
    return df


# News RSS/HTML sentiment sources (symbol → list of feed URLs)
NEWS_FEEDS = {
    # e.g. 'AAPL': ['https://finance.yahoo.com/rss/headline?s=AAPL',
    #               'https://www.reuters.com/companies/AAPL.OQ?view=companyNews&format=xml'],
}


def add_news_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add news sentiment per symbol per day using RSS feeds and VADER analysis."""
    if not HAS_VADER:
        print("Warning: vaderSentiment package not available for news sentiment")
        df["news_sentiment"] = 0.0
        return df

    analyzer = SentimentIntensityAnalyzer()
    records = []

    for symbol in df["symbol"].unique():
        for date in df["timestamp"].dt.date.unique():
            sentiment_scores = []

            # Check if we have news feeds for this symbol
            if symbol in NEWS_FEEDS:
                for feed_url in NEWS_FEEDS[symbol]:
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries:
                            # Parse entry date - this is a simplified approach
                            if (
                                hasattr(entry, "published_parsed")
                                and entry.published_parsed
                            ):
                                entry_date = datetime.date(*entry.published_parsed[:3])
                                if entry_date == date:
                                    # Analyze sentiment of title and summary
                                    title = entry.get("title", "")
                                    summary = entry.get("summary", "")
                                    text = f"{title} {summary}"
                                    score = analyzer.polarity_scores(text)["compound"]
                                    sentiment_scores.append(score)
                    except Exception as e:
                        print(f"Error parsing feed {feed_url}: {e}")
                        continue

            # Average sentiment for the day
            avg_sentiment = (
                sum(sentiment_scores) / len(sentiment_scores)
                if sentiment_scores
                else 0.0
            )
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.Timestamp(date),
                    "news_sentiment": avg_sentiment,
                }
            )

    news_df = pd.DataFrame(records)
    df = df.merge(news_df, on=["symbol", "timestamp"], how="left")
    df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build combined dataset")
    parser.add_argument(
        "--symbols", nargs="+", default=["AAPL"], help="List of ticker symbols"
    )
    parser.add_argument(
        "--start", default="2020-01-01", help="Real data start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2023-01-01", help="Real data end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--timestep",
        choices=["day", "hour", "minute"],
        default="day",
        help="Data interval",
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Synthetic data days per scenario"
    )
    parser.add_argument(
        "--volatility", type=float, default=0.02, help="Synthetic volatility"
    )
    parser.add_argument(
        "--scenarios", type=int, default=5, help="Synthetic scenarios per symbol"
    )
    parser.add_argument(
        "--output", default="data/combined_data.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    print(f"Fetching real data for {args.symbols} from {args.start} to {args.end}...")
    real_df = fetch_real(args.symbols, args.start, args.end, args.timestep)
    print(f"Real data rows: {len(real_df)}")

    print(f"Generating synthetic data: {args.scenarios} scenarios per symbol...")
    synth_df = generate_synthetic(
        args.symbols, args.days, args.volatility, args.scenarios
    )
    print(f"Synthetic data rows: {len(synth_df)}")

    combined = pd.concat([real_df, synth_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    # Integrate additional sentiment sources
    print("Integrating Hugging Face financial_sentiment data...")
    combined = add_hf_sentiment(combined)
    print("Integrating Twitter sentiment via snscrape...")
    combined = add_twitter_sentiment(combined)
    print("Integrating News sentiment via RSS feeds...")
    combined = add_news_sentiment(combined)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Saved combined dataset to {args.output}")


if __name__ == "__main__":
    main()
