import tweepy
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
from typing import List, Dict, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TWITTER_CONFIG, DATA_CONFIG, TWITTER_KEYWORDS

class TwitterCollector:
    def __init__(self):
        """Initialize Twitter API client"""
        self.client = self._setup_twitter_client()
        self.data_dir = DATA_CONFIG['raw_data_dir']
        self._ensure_data_directory()
    
    def _setup_twitter_client(self):
        """Setup Twitter API v2 client"""
        try:
            client = tweepy.Client(
                bearer_token=TWITTER_CONFIG['bearer_token'],
                consumer_key=TWITTER_CONFIG['api_key'],
                consumer_secret=TWITTER_CONFIG['api_secret'],
                access_token=TWITTER_CONFIG['access_token'],
                access_token_secret=TWITTER_CONFIG['access_token_secret'],
                wait_on_rate_limit=True
            )
            return client
        except Exception as e:
            print(f"Error setting up Twitter client: {e}")
            return None
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_tweets_by_keywords(self, keywords: List[str], max_tweets: int = 1000,
                                   start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """
        Collect tweets containing specific keywords
        
        Args:
            keywords: List of keywords to search for
            max_tweets: Maximum number of tweets to collect
            
        Returns:
            List of tweet dictionaries
        """
        if not self.client:
            print("Twitter client not initialized")
            return []
        
        tweets = []
        query = " OR ".join(keywords)
        
        try:
            # Prefer full-archive if date range specified and access allows; fallback to recent
            search_fn = self.client.search_all_tweets if start_time else self.client.search_recent_tweets
            search_kwargs = {
                'query': query,
                'tweet_fields': ['created_at', 'public_metrics', 'context_annotations', 'lang'],
                'max_results': 100
            }
            if start_time:
                # Tweepy expects RFC3339 timestamps
                search_kwargs['start_time'] = start_time.isoformat(timespec='seconds').replace('+00:00', 'Z') if start_time.tzinfo else start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            if end_time:
                search_kwargs['end_time'] = end_time.isoformat(timespec='seconds').replace('+00:00', 'Z') if end_time.tzinfo else end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

            response = tweepy.Paginator(
                search_fn,
                **search_kwargs
            ).flatten(limit=max_tweets)
            
            for tweet in response:
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'quote_count': tweet.public_metrics['quote_count'],
                    'lang': tweet.lang,
                    'keywords_matched': [kw for kw in keywords if kw.lower() in tweet.text.lower()]
                }
                tweets.append(tweet_data)
                
        except Exception as e:
            print(f"Error collecting tweets: {e}")
        
        return tweets
    
    def collect_tweets_for_stocks(self, stock_symbols: List[str] = None,
                                  start_time: datetime = None, end_time: datetime = None,
                                  max_tweets_per_symbol: int = 1000) -> Dict[str, List[Dict]]:
        """
        Collect tweets for multiple stock symbols
        
        Args:
            stock_symbols: List of stock symbols to collect tweets for
            
        Returns:
            Dictionary mapping stock symbols to their tweets
        """
        if stock_symbols is None:
            stock_symbols = list(TWITTER_KEYWORDS.keys())
        
        all_tweets = {}
        
        for symbol in stock_symbols:
            print(f"Collecting tweets for {symbol}...")
            keywords = TWITTER_KEYWORDS.get(symbol, [f'${symbol}', f'{symbol} stock'])
            tweets = self.collect_tweets_by_keywords(
                keywords,
                max_tweets=max_tweets_per_symbol,
                start_time=start_time,
                end_time=end_time
            )
            all_tweets[symbol] = tweets
            print(f"Collected {len(tweets)} tweets for {symbol}")
            
            # Rate limiting - wait between requests
            time.sleep(2)
        
        return all_tweets
    
    def save_tweets_to_json(self, tweets_data: Dict[str, List[Dict]], filename: str = None):
        """
        Save tweets data to JSON file
        
        Args:
            tweets_data: Dictionary containing tweets data
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tweets_raw_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Add metadata
        output_data = {
            'metadata': {
                'collection_timestamp': datetime.now().isoformat(),
                'total_symbols': len(tweets_data),
                'total_tweets': sum(len(tweets) for tweets in tweets_data.values())
            },
            'tweets': tweets_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Tweets saved to {filepath}")
        return filepath
    
    def get_tweet_statistics(self, tweets_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Get statistics about collected tweets
        
        Args:
            tweets_data: Dictionary containing tweets data
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_symbols': len(tweets_data),
            'total_tweets': 0,
            'symbol_stats': {}
        }
        
        for symbol, tweets in tweets_data.items():
            symbol_stat = {
                'tweet_count': len(tweets),
                'avg_likes': sum(t.get('like_count', 0) for t in tweets) / len(tweets) if tweets else 0,
                'avg_retweets': sum(t.get('retweet_count', 0) for t in tweets) / len(tweets) if tweets else 0,
                'languages': list(set(t.get('lang', 'unknown') for t in tweets))
            }
            stats['symbol_stats'][symbol] = symbol_stat
            stats['total_tweets'] += len(tweets)
        
        return stats

def main():
    """Main function to run Twitter data collection"""
    collector = TwitterCollector()
    
    if not collector.client:
        print("Failed to initialize Twitter client. Please check your API credentials.")
        return
    
    print("Starting Twitter data collection...")
    
    # Collect tweets for all stock symbols
    tweets_data = collector.collect_tweets_for_stocks()
    
    # Save to JSON file
    filepath = collector.save_tweets_to_json(tweets_data)
    
    # Print statistics
    stats = collector.get_tweet_statistics(tweets_data)
    print("\nCollection Statistics:")
    print(f"Total symbols: {stats['total_symbols']}")
    print(f"Total tweets: {stats['total_tweets']}")
    
    for symbol, symbol_stats in stats['symbol_stats'].items():
        print(f"\n{symbol}:")
        print(f"  Tweets: {symbol_stats['tweet_count']}")
        print(f"  Avg Likes: {symbol_stats['avg_likes']:.2f}")
        print(f"  Avg Retweets: {symbol_stats['avg_retweets']:.2f}")
        print(f"  Languages: {symbol_stats['languages']}")

if __name__ == "__main__":
    main()
