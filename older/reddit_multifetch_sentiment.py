# reddit_multifetch_sentiment.py

import requests
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SUBREDDIT   = "CryptoCurrency"
POST_LIMIT  = 500
OUTPUT_CSV  = "daily_reddit_sentiment.csv"
USER_AGENT  = "crypto-sentiment/0.1 by u/your_reddit_username"

def fetch_reddit_titles(subreddit, limit):
    url     = f"https://www.reddit.com/r/{subreddit}/new.json"
    headers = {"User-Agent": USER_AGENT}
    posts   = []
    after   = None

    while len(posts) < limit:
        params = {"limit": min(100, limit - len(posts))}
        if after:
            params["after"] = after

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data     = resp.json()["data"]
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            ts    = child["data"]["created_utc"]
            title = child["data"]["title"]
            date  = datetime.utcfromtimestamp(ts).date()
            posts.append((date, title))

        after = data.get("after")
        if not after:
            break

    return posts

def compute_daily_sentiment(posts):
    analyzer = SentimentIntensityAnalyzer()
    buckets  = defaultdict(list)
    for date, text in posts:
        buckets[date].append(analyzer.polarity_scores(text)["compound"])
    return pd.DataFrame([
        {"date": d, "sentiment": sum(vals)/len(vals)}
        for d, vals in sorted(buckets.items())
    ])

def main():
    print(f"Fetching up to {POST_LIMIT} Reddit posts…")
    posts = fetch_reddit_titles(SUBREDDIT, POST_LIMIT)
    print(f"→ Retrieved {len(posts)} posts")

    df = compute_daily_sentiment(posts)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote daily sentiment to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
