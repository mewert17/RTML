# rss_coin_sentiment.py

import os
import requests
import feedparser
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure the output folder exists
os.makedirs("data", exist_ok=True)

COINS    = ["bitcoin", "ethereum", "dogecoin"]
UA       = "crypto-rss/1.0"
BASE_URL = "https://news.google.com/rss/search?q="

def fetch_rss(term):
    url = BASE_URL + term
    print(f"→ Fetching RSS for '{term}' from {url}")
    r = requests.get(url, headers={"User-Agent": UA})
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    print(f"   Found {len(feed.entries)} entries")
    out = []
    for e in feed.entries:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if not t:
            continue
        date = datetime(*t[:3]).date()
        out.append((date, e.title))
    print(f"   Parsed {len(out)} valid items for '{term}'\n")
    return out

def score_daily(posts):
    analyzer = SentimentIntensityAnalyzer()
    buckets = defaultdict(list)
    for d, t in posts:
        buckets[d].append(analyzer.polarity_scores(t)["compound"])
    records = [{"date": d, "sentiment": sum(v) / len(v)} for d, v in sorted(buckets.items())]
    return pd.DataFrame(records)

def main():
    for coin in COINS:
        posts = fetch_rss(coin)
        if not posts:
            print(f"⚠️  No posts for '{coin}', skipping.\n")
            continue

        df = score_daily(posts)
        fn = f"data/daily_{coin}_news.csv"
        df.to_csv(fn, index=False)
        print(f"✅  Wrote {len(df)} days → {fn}\n")

if __name__ == "__main__":
    main()
