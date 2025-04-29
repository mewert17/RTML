# rss_sentiment_fixed.py

import requests, feedparser
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ────────────────────────────────────────────────────────────────
RSS_URL    = "https://www.coindesk.com/arc/outboundfeeds/rss/"  
OUTPUT_CSV = "daily_news_sentiment.csv"
UA         = "crypto-rss-sentiment/0.1"
# ─────────────────────────────────────────────────────────────────────────────

def fetch_headlines(rss_url):
    r = requests.get(rss_url, headers={"User-Agent": UA})
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    posts = []
    for entry in feed.entries:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if not t: 
            continue
        date = datetime(*t[:3]).date()
        posts.append((date, entry.title))
    return posts

def compute_daily_sentiment(posts):
    analyzer = SentimentIntensityAnalyzer()
    buckets  = defaultdict(list)
    for date, text in posts:
        buckets[date].append(analyzer.polarity_scores(text)["compound"])
    return pd.DataFrame([
        {"date": d, "sentiment": sum(scores)/len(scores)}
        for d, scores in sorted(buckets.items())
    ])

if __name__ == "__main__":
    headlines = fetch_headlines(RSS_URL)
    print(f"Fetched {len(headlines)} headlines")
    df = compute_daily_sentiment(headlines)
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.head())
