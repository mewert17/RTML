# multirss_fixed.py

import requests, feedparser
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://cryptopanic.com/news/?format=rss",          # Aggregated crypto news
    "https://news.google.com/rss/search?q=cryptocurrency",
    "https://cointelegraph.com/rss",                     # Cointelegraph RSS
    "https://feeds.feedburner.com/CoinDesk"              # CoinDesk via FeedBurner
]
OUTPUT_CSV = "daily_multi_feed_sentiment.csv"
USER_AGENT = "crypto-multi-rss/0.1 by u/your_username"
# ─────────────────────────────────────────────────────────────────────────────

def fetch_headlines(rss_url):
    resp = requests.get(rss_url, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    posts = []
    for e in feed.entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if not t: 
            continue
        date = datetime(*t[:3]).date()
        posts.append((date, e.title))
    return posts

def compute_daily_sentiment(all_posts):
    analyzer = SentimentIntensityAnalyzer()
    buckets = defaultdict(list)
    for date, text in all_posts:
        buckets[date].append(analyzer.polarity_scores(text)["compound"])
    return pd.DataFrame([
        {"date": d, "sentiment": sum(scores)/len(scores)}
        for d, scores in sorted(buckets.items())
    ])

def main():
    all_posts = []
    for url in RSS_FEEDS:
        try:
            posts = fetch_headlines(url)
            print(f"✅  Fetched {len(posts)} headlines from {url}")
            all_posts.extend(posts)
        except Exception as e:
            print(f"⚠️  Skipping {url}: {e}")

    if not all_posts:
        print("No headlines fetched—check your feed URLs or network.")
        return

    df = compute_daily_sentiment(all_posts)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCombined daily sentiment written to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
