# reddit_sentiment_no_auth.py

import requests
from datetime import datetime
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ────────────────────────────────────────────────────────────────
SUBREDDIT   = "CryptoCurrency"
POST_LIMIT  = 200               # up to 100 per reques
OUTPUT_CSV  = "daily_sentiment.csv"
USER_AGENT  = "crypto-sentiment-script/0.1 by u/your_reddit_username"
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reddit_titles(subreddit, limit):
    """
    Fetches up to `limit` newest submissions from the subreddit
    via the public JSON endpoint (no API key needed).
    """
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    params = {"limit": min(limit, 100)}
    headers = {"User-Agent": USER_AGENT}
    
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    items = resp.json()["data"]["children"]
    
    posts = []
    for item in items:
        data = item["data"]
        date = datetime.utcfromtimestamp(data["created_utc"]).date()
        posts.append((date, data["title"]))
    return posts

def compute_daily_sentiment(posts):
    """
    Runs VADER on each (date, text) and returns a DataFrame
    of average daily compound scores.
    """
    analyzer = SentimentIntensityAnalyzer()
    buckets = defaultdict(list)
    
    for date, text in posts:
        score = analyzer.polarity_scores(text)["compound"]
        buckets[date].append(score)
    
    records = [
        {"date": d, "sentiment": sum(ss)/len(ss)}
        for d, ss in sorted(buckets.items())
    ]
    return pd.DataFrame(records)

def main():
    posts = fetch_reddit_titles(SUBREDDIT, POST_LIMIT)
    print(f"Fetched {len(posts)} posts from r/{SUBREDDIT}")
    
    df = compute_daily_sentiment(posts)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote daily sentiment to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
