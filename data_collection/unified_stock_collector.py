import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_CONFIG, STOCK_SYMBOLS, STOCK_API_CONFIG
from alpha_vantage_collector import AlphaVantageCollector
from finnhub_collector import FinnhubCollector
from yahoo_finance_collector import YahooFinanceCollector

class UnifiedStockCollector:
    """
    Unified stock data collector that uses multiple APIs with automatic fallback
    Priority: Alpha Vantage -> Finnhub -> Yahoo Finance
    """
    
    def __init__(self):
        """Initialize the unified collector with available APIs"""
        self.data_dir = DATA_CONFIG['raw_data_dir']
        self._ensure_data_directory()
        
        # Initialize available collectors
        self.collectors = {}
        self._initialize_collectors()
        
        # Track API usage for rate limiting
        self.api_usage = {}
        self._initialize_usage_tracking()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _initialize_collectors(self):
        """Initialize available API collectors"""
        # Try Alpha Vantage first
        if STOCK_API_CONFIG['primary']['enabled'] and STOCK_API_CONFIG['primary']['api_key']:
            try:
                self.collectors['alpha_vantage'] = AlphaVantageCollector()
                print("✅ Alpha Vantage collector initialized")
            except Exception as e:
                print(f"❌ Alpha Vantage initialization failed: {e}")
        
        # Try Finnhub second
        if STOCK_API_CONFIG['secondary']['enabled'] and STOCK_API_CONFIG['secondary']['api_key']:
            try:
                self.collectors['finnhub'] = FinnhubCollector()
                print("✅ Finnhub collector initialized")
            except Exception as e:
                print(f"❌ Finnhub initialization failed: {e}")
        
        # Try Yahoo Finance as fallback
        if STOCK_API_CONFIG['fallback']['enabled']:
            try:
                self.collectors['yahoo_finance'] = YahooFinanceCollector()
                print("✅ Yahoo Finance collector initialized (fallback)")
            except Exception as e:
                print(f"❌ Yahoo Finance initialization failed: {e}")
        
        if not self.collectors:
            raise Exception("No stock data collectors available! Please configure at least one API.")
    
    def _initialize_usage_tracking(self):
        """Initialize API usage tracking for rate limiting"""
        for api_name in self.collectors.keys():
            self.api_usage[api_name] = {
                'last_call': 0,
                'calls_made': 0,
                'rate_limit': STOCK_API_CONFIG.get(api_name.split('_')[0], {}).get('rate_limit', 60)
            }
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if we can make an API call without exceeding rate limits"""
        current_time = time.time()
        usage = self.api_usage[api_name]
        
        # Reset counter every minute
        if current_time - usage['last_call'] > 60:
            usage['calls_made'] = 0
            usage['last_call'] = current_time
        
        return usage['calls_made'] < usage['rate_limit']
    
    def _record_api_call(self, api_name: str):
        """Record an API call for rate limiting"""
        self.api_usage[api_name]['calls_made'] += 1
        self.api_usage[api_name]['last_call'] = time.time()
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d",
                      start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get stock data using the best available API
        
        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            start: Optional start datetime
            end: Optional end datetime
            
        Returns:
            DataFrame with stock data
        """
        # Try APIs in priority order
        for api_name in ['alpha_vantage', 'finnhub', 'yahoo_finance']:
            if api_name in self.collectors and self._check_rate_limit(api_name):
                try:
                    print(f"📊 Fetching {symbol} data from {api_name}...")
                    
                    collector = self.collectors[api_name]
                    
                    if api_name == 'alpha_vantage':
                        data = collector.get_stock_data(symbol, outputsize="full", start=start, end=end)
                    elif api_name == 'finnhub':
                        data = collector.get_stock_data(symbol, resolution="D", start=start, end=end)
                    else:  # yahoo_finance
                        data = collector.get_stock_data(symbol, period=period, interval=interval, start=start, end=end)
                    
                    if not data.empty:
                        self._record_api_call(api_name)
                        print(f"✅ Successfully fetched {len(data)} records for {symbol} from {api_name}")
                        return data
                    else:
                        print(f"⚠️ No data returned from {api_name} for {symbol}")
                        
                except Exception as e:
                    print(f"❌ Error with {api_name} for {symbol}: {e}")
                    continue
        
        print(f"❌ Failed to fetch data for {symbol} from all available APIs")
        return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], period: str = "1y", interval: str = "1d",
                                start: Optional[datetime] = None, end: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Get stock data for multiple symbols using the best available APIs
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            start: Optional start datetime
            end: Optional end datetime
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        stocks_data = {}
        
        for i, symbol in enumerate(symbols):
            print(f"\n📈 Processing {symbol} ({i+1}/{len(symbols)})")
            data = self.get_stock_data(symbol, period, interval, start=start, end=end)
            if not data.empty:
                stocks_data[symbol] = data
            else:
                print(f"⚠️ Skipping {symbol} - no data available")
        
        return stocks_data
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        # Try Finnhub first for real-time data
        if 'finnhub' in self.collectors and self._check_rate_limit('finnhub'):
            try:
                collector = self.collectors['finnhub']
                quote = collector.get_quote(symbol)
                if quote:
                    self._record_api_call('finnhub')
                    return quote
            except Exception as e:
                print(f"❌ Error getting real-time quote from Finnhub: {e}")
        
        # Fallback to latest data from other APIs
        data = self.get_stock_data(symbol, period="1d", interval="1d")
        if not data.empty:
            latest = data.iloc[-1]
            return {
                'c': latest['close'],  # current price
                'h': latest['high'],   # high
                'l': latest['low'],     # low
                'o': latest['open'],    # open
                'pc': data.iloc[-2]['close'] if len(data) > 1 else latest['close'],  # previous close
                't': int(latest['date'].timestamp())  # timestamp
            }
        
        return {}
    
    def get_company_news(self, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent company news
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of news articles
        """
        if 'finnhub' in self.collectors and self._check_rate_limit('finnhub'):
            try:
                collector = self.collectors['finnhub']
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                news = collector.get_news(symbol, start_date, end_date)
                if news:
                    self._record_api_call('finnhub')
                    return news
            except Exception as e:
                print(f"❌ Error getting news from Finnhub: {e}")
        
        return []
    
    def combine_stocks_data(self, stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple stock DataFrames into a single DataFrame"""
        if not stocks_data:
            return pd.DataFrame()
        
        combined_data = []
        for symbol, data in stocks_data.items():
            if not data.empty:
                combined_data.append(data)
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_stock_data_to_csv(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save stock data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unified_stock_data_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        
        print(f"💾 Stock data saved to {filepath}")
        return filepath
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the stock data"""
        if data.empty:
            return {'error': 'No data available'}
        
        stats = {
            'total_records': len(data),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            },
            'symbols': data['symbol'].unique().tolist(),
            'symbol_counts': data['symbol'].value_counts().to_dict(),
            'price_statistics': {
                'avg_close': data['close'].mean(),
                'min_close': data['close'].min(),
                'max_close': data['close'].max(),
                'std_close': data['close'].std()
            },
            'volume_statistics': {
                'avg_volume': data['volume'].mean(),
                'min_volume': data['volume'].min(),
                'max_volume': data['volume'].max(),
                'std_volume': data['volume'].std()
            },
            'api_usage': self.api_usage
        }
        
        return stats
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if data.empty:
            return data
        
        # Sort by symbol and date
        data = data.sort_values(['symbol', 'date'])
        
        # Calculate technical indicators for each symbol
        result_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Simple Moving Averages
            symbol_data['sma_5'] = symbol_data['close'].rolling(window=5).mean()
            symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
            symbol_data['sma_50'] = symbol_data['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            symbol_data['ema_12'] = symbol_data['close'].ewm(span=12).mean()
            symbol_data['ema_26'] = symbol_data['close'].ewm(span=26).mean()
            
            # MACD
            symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            symbol_data['bb_middle'] = symbol_data['close'].rolling(window=20).mean()
            bb_std = symbol_data['close'].rolling(window=20).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (bb_std * 2)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (bb_std * 2)
            
            # Price change
            symbol_data['price_change'] = symbol_data['close'].pct_change()
            symbol_data['price_change_abs'] = symbol_data['close'].diff()
            
            result_data.append(symbol_data)
        
        return pd.concat(result_data, ignore_index=True)
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all available APIs"""
        status = {}
        for api_name, collector in self.collectors.items():
            usage = self.api_usage[api_name]
            status[api_name] = {
                'available': True,
                'rate_limit': usage['rate_limit'],
                'calls_made': usage['calls_made'],
                'last_call': datetime.fromtimestamp(usage['last_call']).strftime('%Y-%m-%d %H:%M:%S') if usage['last_call'] > 0 else 'Never'
            }
        
        return status

def main():
    """Main function to run unified stock data collection"""
    collector = UnifiedStockCollector()
    
    print("🚀 Starting Unified Stock Data Collection...")
    print(f"📊 Available APIs: {', '.join(collector.collectors.keys())}")
    
    # Get stock data for all symbols
    stocks_data = collector.get_multiple_stocks_data(STOCK_SYMBOLS, period="1y", interval="1d")
    
    # Combine all data
    combined_data = collector.combine_stocks_data(stocks_data)
    
    if combined_data.empty:
        print("❌ No data collected. Please check your API keys and internet connection.")
        return
    
    # Calculate technical indicators
    enhanced_data = collector.calculate_technical_indicators(combined_data)
    
    # Save to CSV file
    filepath = collector.save_stock_data_to_csv(enhanced_data)
    
    # Print statistics
    stats = collector.get_data_statistics(enhanced_data)
    print("\n📈 Collection Statistics:")
    print(f"Total records: {stats['total_records']}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Symbols: {', '.join(stats['symbols'])}")
    print(f"Average close price: ${stats['price_statistics']['avg_close']:.2f}")
    print(f"Average volume: {stats['volume_statistics']['avg_volume']:,.0f}")
    
    print("\n📊 Records per symbol:")
    for symbol, count in stats['symbol_counts'].items():
        print(f"  {symbol}: {count} records")
    
    print("\n🔧 API Usage:")
    for api_name, usage in stats['api_usage'].items():
        print(f"  {api_name}: {usage['calls_made']} calls made")

if __name__ == "__main__":
    main()
