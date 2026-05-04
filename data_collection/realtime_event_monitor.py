import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
from dataclasses import dataclass
import queue

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_CONFIG, STOCK_SYMBOLS, TWITTER_KEYWORDS
from twitter_collector import TwitterCollector
from unified_stock_collector import UnifiedStockCollector
from sentiment.score_tweets import SentimentAnalyzer

@dataclass
class StockEvent:
    """Represents a stock-affecting event"""
    symbol: str
    event_type: str  # 'tweet_spike', 'price_movement', 'news_event', 'sentiment_change'
    timestamp: datetime
    description: str
    impact_score: float  # -1.0 to 1.0 (negative = bearish, positive = bullish)
    data: Dict[str, Any]
    tweet_count: int = 0
    price_change: float = 0.0
    volume_spike: float = 0.0

class RealTimeEventMonitor:
    """
    Real-time event monitor that detects stock-affecting events from tweets
    and triggers appropriate responses
    """
    
    def __init__(self):
        """Initialize the real-time event monitor"""
        self.data_dir = DATA_CONFIG['raw_data_dir']
        self._ensure_data_directory()
        
        # Initialize collectors
        self.twitter_collector = TwitterCollector()
        self.stock_collector = UnifiedStockCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Event tracking
        self.events = []
        self.event_callbacks = []
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Configuration
        self.monitor_interval = 60  # seconds
        self.tweet_threshold = 10  # minimum tweets to trigger event
        self.sentiment_threshold = 0.3  # minimum sentiment change to trigger event
        self.price_threshold = 0.02  # 2% price change threshold
        
        # Recent data cache
        self.recent_tweets = {}
        self.recent_prices = {}
        self.baseline_sentiment = {}
        
        # Event queue for processing
        self.event_queue = queue.Queue()
        
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def add_event_callback(self, callback: Callable[[StockEvent], None]):
        """Add a callback function to be called when events are detected"""
        self.event_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            print("⚠️ Monitoring is already running")
            return
        
        print("🚀 Starting real-time event monitoring...")
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start event processing thread
        self.process_thread = threading.Thread(target=self._process_events, daemon=True)
        self.process_thread.start()
        
        print("✅ Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_monitoring:
            print("⚠️ Monitoring is not running")
            return
        
        print("🛑 Stopping real-time event monitoring...")
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        print("✅ Real-time monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                print(f"🔍 Monitoring cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Monitor each symbol
                for symbol in STOCK_SYMBOLS:
                    self._monitor_symbol(symbol)
                
                # Wait for next cycle
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _monitor_symbol(self, symbol: str):
        """Monitor a specific symbol for events"""
        try:
            # Get recent tweets for this symbol
            recent_tweets = self._get_recent_tweets(symbol)
            
            # Get current stock data
            current_price_data = self._get_current_price_data(symbol)
            
            # Analyze for events
            events = self._detect_events(symbol, recent_tweets, current_price_data)
            
            # Add events to queue
            for event in events:
                self.event_queue.put(event)
                
        except Exception as e:
            print(f"❌ Error monitoring {symbol}: {e}")
    
    def _get_recent_tweets(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent tweets for a symbol"""
        try:
            # Get tweets from last 5 minutes
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            tweets = self.twitter_collector.collect_tweets_for_symbol(
                symbol, 
                start_time=start_time, 
                end_time=end_time,
                max_tweets=100
            )
            
            return tweets
            
        except Exception as e:
            print(f"❌ Error getting tweets for {symbol}: {e}")
            return []
    
    def _get_current_price_data(self, symbol: str) -> Dict[str, Any]:
        """Get current price data for a symbol"""
        try:
            # Get real-time quote
            quote = self.stock_collector.get_real_time_quote(symbol)
            
            # Get recent price history for comparison
            recent_data = self.stock_collector.get_stock_data(
                symbol, 
                period="1d", 
                interval="1d"
            )
            
            return {
                'quote': quote,
                'recent_data': recent_data
            }
            
        except Exception as e:
            print(f"❌ Error getting price data for {symbol}: {e}")
            return {}
    
    def _detect_events(self, symbol: str, tweets: List[Dict[str, Any]], price_data: Dict[str, Any]) -> List[StockEvent]:
        """Detect events from tweets and price data"""
        events = []
        
        # Check for tweet volume spike
        if len(tweets) >= self.tweet_threshold:
            event = self._create_tweet_spike_event(symbol, tweets)
            if event:
                events.append(event)
        
        # Check for sentiment change
        sentiment_event = self._detect_sentiment_change(symbol, tweets)
        if sentiment_event:
            events.append(sentiment_event)
        
        # Check for price movement
        price_event = self._detect_price_movement(symbol, price_data)
        if price_event:
            events.append(price_event)
        
        return events
    
    def _create_tweet_spike_event(self, symbol: str, tweets: List[Dict[str, Any]]) -> Optional[StockEvent]:
        """Create a tweet spike event"""
        if len(tweets) < self.tweet_threshold:
            return None
        
        # Calculate average sentiment
        sentiments = []
        for tweet in tweets:
            sentiment = self.sentiment_analyzer.predict_sentiment(tweet.get('text', ''))
            sentiments.append(sentiment)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Calculate impact score based on volume and sentiment
        volume_factor = min(len(tweets) / 50, 1.0)  # Normalize to 0-1
        impact_score = avg_sentiment * volume_factor
        
        return StockEvent(
            symbol=symbol,
            event_type='tweet_spike',
            timestamp=datetime.now(),
            description=f"Tweet volume spike: {len(tweets)} tweets in 5 minutes",
            impact_score=impact_score,
            data={
                'tweet_count': len(tweets),
                'avg_sentiment': avg_sentiment,
                'volume_factor': volume_factor
            },
            tweet_count=len(tweets)
        )
    
    def _detect_sentiment_change(self, symbol: str, tweets: List[Dict[str, Any]]) -> Optional[StockEvent]:
        """Detect significant sentiment change"""
        if len(tweets) < 5:  # Need minimum tweets for sentiment analysis
            return None
        
        # Calculate current sentiment
        current_sentiments = []
        for tweet in tweets:
            sentiment = self.sentiment_analyzer.predict_sentiment(tweet.get('text', ''))
            current_sentiments.append(sentiment)
        
        current_avg = sum(current_sentiments) / len(current_sentiments)
        
        # Compare with baseline
        baseline = self.baseline_sentiment.get(symbol, 0)
        sentiment_change = current_avg - baseline
        
        if abs(sentiment_change) >= self.sentiment_threshold:
            # Update baseline
            self.baseline_sentiment[symbol] = current_avg
            
            return StockEvent(
                symbol=symbol,
                event_type='sentiment_change',
                timestamp=datetime.now(),
                description=f"Sentiment change: {sentiment_change:.2f} ({current_avg:.2f} vs {baseline:.2f})",
                impact_score=sentiment_change,
                data={
                    'current_sentiment': current_avg,
                    'baseline_sentiment': baseline,
                    'sentiment_change': sentiment_change,
                    'tweet_count': len(tweets)
                }
            )
        
        return None
    
    def _detect_price_movement(self, symbol: str, price_data: Dict[str, Any]) -> Optional[StockEvent]:
        """Detect significant price movement"""
        quote = price_data.get('quote', {})
        recent_data = price_data.get('recent_data', pd.DataFrame())
        
        if not quote or recent_data.empty:
            return None
        
        current_price = quote.get('c', 0)
        previous_close = quote.get('pc', 0)
        
        if previous_close == 0:
            return None
        
        price_change_pct = (current_price - previous_close) / previous_close
        
        if abs(price_change_pct) >= self.price_threshold:
            return StockEvent(
                symbol=symbol,
                event_type='price_movement',
                timestamp=datetime.now(),
                description=f"Price movement: {price_change_pct:.2%} ({current_price:.2f} vs {previous_close:.2f})",
                impact_score=price_change_pct,
                data={
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'price_change_pct': price_change_pct,
                    'volume': quote.get('v', 0)
                },
                price_change=price_change_pct
            )
        
        return None
    
    def _process_events(self):
        """Process events from the queue"""
        while self.is_monitoring:
            try:
                # Get event from queue (with timeout)
                event = self.event_queue.get(timeout=1)
                
                # Process the event
                self._handle_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing event: {e}")
    
    def _handle_event(self, event: StockEvent):
        """Handle a detected event"""
        print(f"🚨 Event detected: {event.symbol} - {event.event_type}")
        print(f"   Description: {event.description}")
        print(f"   Impact Score: {event.impact_score:.2f}")
        
        # Store event
        self.events.append(event)
        
        # Save event to file
        self._save_event(event)
        
        # Call registered callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"❌ Error in event callback: {e}")
    
    def _save_event(self, event: StockEvent):
        """Save event to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_event_{event.symbol}_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Convert event to dictionary
            event_dict = {
                'symbol': event.symbol,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'description': event.description,
                'impact_score': event.impact_score,
                'data': event.data,
                'tweet_count': event.tweet_count,
                'price_change': event.price_change,
                'volume_spike': event.volume_spike
            }
            
            with open(filepath, 'w') as f:
                json.dump(event_dict, f, indent=2)
            
            print(f"💾 Event saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving event: {e}")
    
    def get_recent_events(self, hours: int = 24) -> List[StockEvent]:
        """Get recent events from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [event for event in self.events if event.timestamp >= cutoff_time]
    
    def get_events_by_symbol(self, symbol: str) -> List[StockEvent]:
        """Get all events for a specific symbol"""
        return [event for event in self.events if event.symbol == symbol]
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of all events"""
        if not self.events:
            return {'total_events': 0}
        
        # Group by symbol
        symbol_events = {}
        for event in self.events:
            if event.symbol not in symbol_events:
                symbol_events[event.symbol] = []
            symbol_events[event.symbol].append(event)
        
        # Calculate statistics
        total_events = len(self.events)
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            'total_events': total_events,
            'symbols_with_events': len(symbol_events),
            'event_types': event_types,
            'symbol_events': {symbol: len(events) for symbol, events in symbol_events.items()},
            'recent_events': len(self.get_recent_events(1))  # Last hour
        }

# Example callback functions
def log_event_callback(event: StockEvent):
    """Example callback that logs events"""
    print(f"📝 Event logged: {event.symbol} - {event.event_type} at {event.timestamp}")

def alert_callback(event: StockEvent):
    """Example callback that sends alerts for high-impact events"""
    if abs(event.impact_score) > 0.5:  # High impact threshold
        print(f"🚨 HIGH IMPACT ALERT: {event.symbol} - {event.description}")

def save_to_database_callback(event: StockEvent):
    """Example callback that saves to database"""
    # This would integrate with your database
    print(f"💾 Event saved to database: {event.symbol} - {event.event_type}")

def main():
    """Main function to run real-time event monitoring"""
    monitor = RealTimeEventMonitor()
    
    # Add example callbacks
    monitor.add_event_callback(log_event_callback)
    monitor.add_event_callback(alert_callback)
    monitor.add_event_callback(save_to_database_callback)
    
    print("🚀 Starting Real-Time Event Monitor")
    print("📊 Monitoring symbols:", ', '.join(STOCK_SYMBOLS[:5]), "...")
    print("⏰ Monitoring interval:", monitor.monitor_interval, "seconds")
    print("🎯 Tweet threshold:", monitor.tweet_threshold)
    print("📈 Sentiment threshold:", monitor.sentiment_threshold)
    print("💰 Price threshold:", monitor.price_threshold)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Keep running until interrupted
        while True:
            time.sleep(10)
            
            # Print summary every 5 minutes
            if len(monitor.events) > 0 and len(monitor.events) % 10 == 0:
                summary = monitor.get_event_summary()
                print(f"\n📊 Event Summary: {summary['total_events']} total events")
                print(f"   Recent events (last hour): {summary['recent_events']}")
                print(f"   Event types: {summary['event_types']}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
        monitor.stop_monitoring()
        print("✅ Monitor stopped")

if __name__ == "__main__":
    main()
