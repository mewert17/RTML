# ps-sentiment.py

import requests
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ────────────────────────────────────────────────────────────────
SUBREDDIT   = "CryptoCurrency"
SIZE        = 500     # number of posts to fetch (Pushshift allows up to ~1000)
OUTPUT_CSV  = "daily_pushshift_sentiment.csv"
USER_AGENT  = "crypto-sentiment/0.1 by u/your_username"
# ─────────────────────────────────────────────────────────────────────────────

def fetch_pushshift(subreddit, size):
    """
    Fetches the most recent `size` submissions from Pushshift for a subreddit.
    Returns a list of (date, title) tuples.
    """
    url = "https://api.pushshift.io/reddit/search/submission/"
    params = {"subreddit": subreddit, "size": size}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    posts = []
    for item in r.json().get("data", []):
        date  = datetime.utcfromtimestamp(item["created_utc"]).date()
        title = item.get("title", "")
        posts.append((date, title))
    return posts

def compute_daily_sentiment(posts):
    """
    Given a list of (date, text), runs VADER on each text and
    returns a DataFrame with the average compound score per date.
    """
    analyzer = SentimentIntensityAnalyzer()
    buckets  = defaultdict(list)

    for date, text in posts:
        buckets[date].append(analyzer.polarity_scores(text)["compound"])

    records = [
        {"date": d, "sentiment": sum(scores) / len(scores)}
        for d, scores in sorted(buckets.items())
    ]
    return pd.DataFrame(records)

def main():
    # 1. Fetch posts
    posts = fetch_pushshift(SUBREDDIT, SIZE)
    print(f"Fetched {len(posts)} posts via Pushshift")

    # 2. Compute daily sentiment
    df = compute_daily_sentiment(posts)

    # 3. Write out to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote daily sentiment to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
