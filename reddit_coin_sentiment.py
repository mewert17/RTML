# reddit_coin_sentiment.py

import os
import requests
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure the output folder exists
os.makedirs("data", exist_ok=True)

COINS      = ["Bitcoin", "ethereum", "dogecoin"]
POST_LIMIT = 500
UA         = "crypto-reddit/1.0 (by u/your_reddit_username)"

def fetch_titles(sub, limit):
    url, headers = f"https://www.reddit.com/r/{sub}/new.json", {"User-Agent": UA}
    posts, after = [], None
    while len(posts) < limit:
        params = {"limit": min(100, limit - len(posts))}
        if after:
            params["after"] = after
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json().get("data", {})
        children = data.get("children", [])
        if not children:
            break
        for c in children:
            ts = c["data"].get("created_utc")
            if ts is None:
                continue
            posts.append((
                datetime.utcfromtimestamp(ts).date(),
                c["data"].get("title", "")
            ))
        after = data.get("after")
        if not after:
            break
    return posts

def score_daily(posts):
    analyzer = SentimentIntensityAnalyzer()
    buckets = defaultdict(list)
    for d, t in posts:
        buckets[d].append(analyzer.polarity_scores(t)["compound"])
    return pd.DataFrame([
        {"date": d, "sentiment": sum(v) / len(v)}
        for d, v in sorted(buckets.items())
    ])

def main():
    for coin in COINS:
        posts = fetch_titles(coin, POST_LIMIT)
        if not posts:
            print(f"⚠️  No posts for '{coin}', skipping.")
            continue

        df = score_daily(posts)
        fn = f"data/daily_{coin.lower()}_reddit.csv"
        df.to_csv(fn, index=False)
        print(f"✅  Wrote {len(df)} days → {fn}")

if __name__ == "__main__":
    main()
