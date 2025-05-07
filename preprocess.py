# preprocess.py

import os
import pandas as pd

# Map your price‐file symbols to the lowercase coin names you used
SYMBOL_MAP = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'DOGE': 'dogecoin'
}

INPUT_PRICE_DIR   = 'data/processed'
INPUT_SENTIMENT_DIR = 'data'
OUTPUT_DIR        = 'data/processed'

def merge_sentiment(price_csv, reddit_csv, news_csv):
    # Load price
    df_price = pd.read_csv(price_csv, parse_dates=['Date'])
    
    # Load Reddit sentiment
    df_red = (
        pd.read_csv(reddit_csv, parse_dates=['date'])
          .rename(columns={'date':'Date', 'sentiment':'reddit_sentiment'})
    )
    
    # Load News sentiment
    df_news = (
        pd.read_csv(news_csv, parse_dates=['date'])
          .rename(columns={'date':'Date', 'sentiment':'news_sentiment'})
    )
    
    # Merge price + both sentiment sources
    df = (
        df_price
          .merge(df_red,  on='Date', how='left')
          .merge(df_news, on='Date', how='left')
    )
    
    # Fill missing days with neutral sentiment
    df['reddit_sentiment'].fillna(0.0, inplace=True)
    df['news_sentiment'].fillna(0.0, inplace=True)
    
    return df

def process_coin(symbol):
    coin = SYMBOL_MAP[symbol]
    
    price_csv  = os.path.join(INPUT_PRICE_DIR, f'{symbol}.csv')
    reddit_csv = os.path.join(INPUT_SENTIMENT_DIR, f'daily_{coin}_reddit.csv')
    news_csv   = os.path.join(INPUT_SENTIMENT_DIR, f'daily_{coin}_news.csv')
    
    df_merged = merge_sentiment(price_csv, reddit_csv, news_csv)
    
    out_csv = os.path.join(OUTPUT_DIR, f'{symbol}_merged.csv')
    df_merged.to_csv(out_csv, index=False)
    print(f'✅ Merged data for {symbol} → {out_csv} ({len(df_merged)} rows)')

if __name__ == '__main__':
    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each coin
    for sym in SYMBOL_MAP:
        process_coin(sym)
