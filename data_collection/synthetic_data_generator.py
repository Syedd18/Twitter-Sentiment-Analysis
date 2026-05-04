#!/usr/bin/env python3
"""
Synthetic Data Generator for Stock Price Prediction Project
Generates realistic OHLCV data for testing when real APIs are blocked
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_SYMBOLS, DATA_CONFIG

def generate_synthetic_ohlcv(symbol, start_date='2020-01-01', end_date='2024-12-31', base_price=100):
    """Generate synthetic OHLCV data for a stock symbol"""
    
    # Set different base prices for different stocks
    base_prices = {
        'AAPL': 150, 'TSLA': 200, 'MSFT': 300, 'GOOGL': 2500,
        'AMZN': 3000, 'META': 300, 'NVDA': 400, 'NFLX': 400,
        'RELIANCE.NS': 2500, 'TCS.NS': 3500, 'HDFCBANK.NS': 1500,
        'INFY.NS': 1500, 'HINDUNILVR.NS': 2500, 'ITC.NS': 400,
        'SBIN.NS': 500, 'BHARTIARTL.NS': 800
    }
    
    base_price = base_prices.get(symbol, base_price)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    n_days = len(dates)
    
    # Generate price series with trend and volatility
    np.random.seed(hash(symbol) % 2**32)  # Consistent random seed per symbol
    
    # Add trend (slight upward bias)
    trend = np.linspace(0, 0.3, n_days)  # 30% growth over period
    
    # Add volatility (random walk)
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    returns += trend / n_days  # Add trend component
    
    # Generate price series
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV for each day
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate intraday volatility
        daily_vol = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
        
        # Generate OHLC
        high = close * (1 + np.random.uniform(0, daily_vol))
        low = close * (1 - np.random.uniform(0, daily_vol))
        open_price = close * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume on volatile days)
        base_volume = 1000000
        volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on volatile days
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)

def generate_synthetic_tweets(symbols, start_date='2020-01-01', end_date='2024-12-31'):
    """Generate synthetic tweet data"""
    
    tweet_templates = {
        'AAPL': ['Apple stock looking bullish today!', 'AAPL earnings beat expectations', 'Apple iPhone sales strong'],
        'TSLA': ['Tesla stock volatile as usual', 'Elon Musk tweets affecting TSLA', 'Tesla delivery numbers impressive'],
        'MSFT': ['Microsoft Azure growth continues', 'MSFT cloud revenue up', 'Microsoft Office 365 strong'],
        'GOOGL': ['Google search revenue growing', 'Alphabet AI investments paying off', 'YouTube ad revenue up'],
        'AMZN': ['Amazon AWS dominance continues', 'AMZN e-commerce growth', 'Amazon Prime membership up'],
        'META': ['Meta VR investments showing promise', 'Facebook user engagement stable', 'Instagram revenue growing'],
        'NVDA': ['NVIDIA AI chip demand surging', 'NVDA gaming revenue strong', 'Data center growth for NVIDIA'],
        'NFLX': ['Netflix subscriber growth slowing', 'NFLX content investment paying off', 'Streaming competition intense'],
        'RELIANCE.NS': ['Reliance Jio subscriber growth', 'Reliance retail expansion', 'RIL petrochemicals strong'],
        'TCS.NS': ['TCS digital transformation deals', 'Tata Consultancy hiring spree', 'TCS cloud migration projects'],
        'HDFCBANK.NS': ['HDFC Bank NPA levels stable', 'HDFC digital banking growth', 'HDFC loan book expansion'],
        'INFY.NS': ['Infosys AI and automation focus', 'INFY cloud services growth', 'Infosys employee retention'],
        'HINDUNILVR.NS': ['HUL rural market penetration', 'Hindustan Unilever brand strength', 'HUL FMCG market share'],
        'ITC.NS': ['ITC cigarette business stable', 'ITC FMCG diversification', 'ITC hotel business recovery'],
        'SBIN.NS': ['SBI digital banking initiatives', 'State Bank NPA reduction', 'SBI home loan growth'],
        'BHARTIARTL.NS': ['Airtel 5G rollout progress', 'Bharti Airtel ARPU improvement', 'Airtel subscriber base growth']
    }
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    tweets_data = {}
    
    for symbol in symbols:
        templates = tweet_templates.get(symbol, [f'{symbol} stock analysis', f'{symbol} market update'])
        symbol_tweets = []
        
        # Generate 1-5 tweets per day randomly
        for date in dates:
            n_tweets = np.random.poisson(2)  # Average 2 tweets per day
            for _ in range(n_tweets):
                template = np.random.choice(templates)
                tweet_id = f"{symbol}_{date.strftime('%Y%m%d')}_{np.random.randint(1000, 9999)}"
                
                symbol_tweets.append({
                    'id': tweet_id,
                    'text': template,
                    'created_at': date.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        tweets_data[symbol] = symbol_tweets
    
    return tweets_data

def main():
    """Generate synthetic data for all symbols"""
    
    print("Generating synthetic stock data...")
    
    # Generate OHLCV data
    all_stock_data = []
    for symbol in STOCK_SYMBOLS:
        print(f"Generating data for {symbol}...")
        df = generate_synthetic_ohlcv(symbol)
        all_stock_data.append(df)
    
    # Combine all stock data
    combined_stock_data = pd.concat(all_stock_data, ignore_index=True)
    
    # Save stock data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stock_file = os.path.join(DATA_CONFIG['raw_data_dir'], f'stock_data_{timestamp}.csv')
    combined_stock_data.to_csv(stock_file, index=False)
    print(f"Saved {len(combined_stock_data)} stock records to {stock_file}")
    
    # Generate tweet data
    print("Generating synthetic tweet data...")
    tweets_data = generate_synthetic_tweets(STOCK_SYMBOLS)
    
    # Save tweet data
    tweets_file = os.path.join(DATA_CONFIG['raw_data_dir'], f'tweets_raw_{timestamp}.json')
    with open(tweets_file, 'w', encoding='utf-8') as f:
        json.dump({'tweets': tweets_data}, f, indent=2)
    
    total_tweets = sum(len(tweets) for tweets in tweets_data.values())
    print(f"Saved {total_tweets} tweets to {tweets_file}")
    
    # Create collection summary
    summary = {
        'timestamp': timestamp,
        'stock_symbols': STOCK_SYMBOLS,
        'stock_records': len(combined_stock_data),
        'tweet_records': total_tweets,
        'date_range': '2020-01-01 to 2024-12-31',
        'data_type': 'synthetic'
    }
    
    summary_file = os.path.join(DATA_CONFIG['raw_data_dir'], f'collection_summary_{timestamp}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Collection summary saved to {summary_file}")
    print("Synthetic data generation complete!")

if __name__ == '__main__':
    main()
