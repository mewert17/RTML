# twitter_sentiment.py

import snscrape.modules.twitter as sntwitter
from collections import defaultdict
from datetime import datetime
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

QUERY      = "bitcoin"      # keyword or hashtag
MAX_TWEETS = 200
OUTPUT_CSV = "daily_twitter_sentiment.csv"

def fetch_tweets(query, limit):
    posts = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit: break
        posts.append((tweet.date.date(), tweet.content))
    return posts

def compute_daily_sentiment(posts):
    analyzer = SentimentIntensityAnalyzer()
    buckets = defaultdict(list)
    for date, text in posts:
        buckets[date].append(analyzer.polarity_scores(text)["compound"])
    return pd.DataFrame([
        {"date": d, "sentiment": sum(ss)/len(ss)}
        for d, ss in sorted(buckets.items())
    ])

if __name__ == "__main__":
    tweets = fetch_tweets(QUERY, MAX_TWEETS)
    df = compute_daily_sentiment(tweets)
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.head())
