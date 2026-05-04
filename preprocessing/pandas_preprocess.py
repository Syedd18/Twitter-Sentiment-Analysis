import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


def latest_with_prefix(directory: str, prefix: str) -> str | None:
	if not os.path.isdir(directory):
		return None
	candidates = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)]
	return max(candidates, key=os.path.getmtime) if candidates else None


def load_raw():
	raw_dir = DATA_CONFIG['raw_data_dir']
	tweets_path = latest_with_prefix(raw_dir, 'tweets_raw_')
	stocks_path = latest_with_prefix(raw_dir, 'stock_data_')
	if tweets_path is None:
		raise FileNotFoundError('No tweets_raw_*.json found')
	# stocks_path may be missing or empty; handle gracefully
	with open(tweets_path, 'r', encoding='utf-8') as f:
		payload = json.load(f)
	tweets_rows = []
	for symbol, items in payload.get('tweets', {}).items():
		for it in items:
			tweets_rows.append({
				'symbol': symbol,
				'id': str(it.get('id')),
				'text': it.get('text'),
				'created_at': it.get('created_at'),
			})
	tweets = pd.DataFrame(tweets_rows)
	stocks = pd.DataFrame()
	if stocks_path is not None and os.path.isfile(stocks_path):
		try:
			stocks = pd.read_csv(stocks_path)
		except Exception:
			stocks = pd.DataFrame()
	return tweets, stocks


def clean_text_series(s: pd.Series) -> pd.Series:
	import re
	s = s.fillna('')
	s = s.str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)
	s = s.str.replace(r'@[A-Za-z0-9_]+', ' ', regex=True)
	s = s.str.replace(r'#[A-Za-z0-9_]+', lambda m: m.group(0)[1:], regex=True)
	s = s.str.replace(r'[^A-Za-z0-9\s\$]', ' ', regex=True)
	s = s.str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
	return s


def aggregate(tweets: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
	if tweets.empty and not stocks.empty:
		# If we have no tweets, synthesize neutral sentiment aligned to stock dates
		out_rows = []
		s = stocks.copy()
		if 'date' in s.columns:
			s['date'] = pd.to_datetime(s['date']).dt.date
		for sym in s['symbol'].dropna().unique():
			sd = s[s['symbol'] == sym]
			for d in sd['date'].dropna().unique():
				out_rows.append({
					'timestamp': pd.Timestamp(d),
					'symbol': sym,
					'avg_sentiment': 0.0,
					'pos_rate': 0.0,
					'neg_rate': 0.0,
					'tweet_volume': 0.0,
					'open': float(sd[sd['date'] == d]['open'].values[-1]) if 'open' in sd.columns else None,
					'close': float(sd[sd['date'] == d]['close'].values[-1]) if 'close' in sd.columns else None,
					'high': float(sd[sd['date'] == d]['high'].values[-1]) if 'high' in sd.columns else None,
					'low': float(sd[sd['date'] == d]['low'].values[-1]) if 'low' in sd.columns else None,
					'volume': float(sd[sd['date'] == d]['volume'].values[-1]) if 'volume' in sd.columns else None,
				})
		return pd.DataFrame(out_rows)
	elif tweets.empty:
		return pd.DataFrame(columns=['timestamp','symbol','avg_sentiment','pos_rate','neg_rate','tweet_volume','open','close','high','low','volume'])

	tweets['created_at'] = pd.to_datetime(tweets['created_at'], errors='coerce')
	tweets = tweets.dropna(subset=['created_at'])
	tweets['clean_text'] = clean_text_series(tweets['text'])

	# If we have sentiment_scored.csv, merge it
	scored_path = os.path.join(DATA_CONFIG['processed_data_dir'], 'sentiment_scored.csv')
	if os.path.isfile(scored_path):
		scored = pd.read_csv(scored_path)
		scored['id'] = scored['id'].astype(str)
		tweets['id'] = tweets['id'].astype(str)
		tweets = tweets.merge(scored[['id','sentiment']], on='id', how='left')
	else:
		tweets['sentiment'] = np.nan

	# 5-minute window per symbol
	tweets['window_start'] = tweets['created_at'].dt.floor('5min')
	grp = tweets.groupby(['symbol','window_start'])
	agg = pd.DataFrame({
		'tweet_volume': grp['id'].count(),
		'pos_rate': grp['sentiment'].apply(lambda x: (x=='positive').sum()/len(x) if len(x)>0 else 0.0),
		'neg_rate': grp['sentiment'].apply(lambda x: (x=='negative').sum()/len(x) if len(x)>0 else 0.0),
		'avg_sentiment': grp['sentiment'].apply(lambda x: ((x=='positive').sum() - (x=='negative').sum())/len(x) if len(x)>0 else 0.0)
	}).reset_index()

	# Prepare stocks and join by date & symbol if available
	stocks = stocks.copy()
	if 'date' in stocks.columns and 'symbol' in stocks.columns:
		stocks['date'] = pd.to_datetime(stocks['date'], errors='coerce').dt.date
		stocks = stocks.dropna(subset=['date'])
	else:
		return agg.rename(columns={'window_start':'timestamp'})

	agg['date'] = agg['window_start'].dt.date
	merged = agg.merge(stocks[['date','symbol','open','close','high','low','volume']], on=['date','symbol'], how='left')
	merged = merged.rename(columns={'window_start':'timestamp'})
	return merged[['timestamp','symbol','avg_sentiment','pos_rate','neg_rate','tweet_volume','open','close','high','low','volume']]


def main():
	processed_dir = DATA_CONFIG['processed_data_dir']
	os.makedirs(processed_dir, exist_ok=True)
	tweets, stocks = load_raw()
	# Fallback if no tweets: use sentiment_scored.csv to synthesize
	if tweets.empty:
		scored_path = os.path.join(processed_dir, 'sentiment_scored.csv')
		if os.path.isfile(scored_path):
			sc = pd.read_csv(scored_path)
			if not sc.empty:
				tweets = sc[['symbol','id','text','created_at']].copy()
	features = aggregate(tweets, stocks)
	out_path = os.path.join(processed_dir, 'processed_features.parquet')
	features.to_parquet(out_path, index=False)
	print(f'Saved {len(features)} rows to {out_path}')


if __name__ == '__main__':
	main()
