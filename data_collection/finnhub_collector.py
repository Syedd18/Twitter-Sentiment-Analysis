import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
from typing import List, Dict, Any, Optional
import json

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, STOCK_SYMBOLS

class FinnhubCollector:
    def __init__(self, api_key: str = None):
        """
        Initialize Finnhub collector
        
        Args:
            api_key: Finnhub API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("Finnhub API key is required. Set FINNHUB_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://finnhub.io/api/v1"
        self.data_dir = DATA_CONFIG['raw_data_dir']
        self._ensure_data_directory()
        
        # Rate limiting: Finnhub free tier allows 60 calls per minute
        self.last_call_time = 0
        self.min_call_interval = 1  # seconds (60 calls per minute = 1 second between calls)
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_api_request(self, endpoint: str, params: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Make API request with error handling and rate limiting
        
        Args:
            endpoint: API endpoint
            params: API parameters
            
        Returns:
            API response data
        """
        self._rate_limit()
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if isinstance(data, dict) and 'error' in data:
                raise Exception(f"Finnhub API Error: {data['error']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {e}")
    
    def get_stock_data(self, symbol: str, resolution: str = "D", 
                      start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get stock data for a specific symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            resolution: Data resolution ('1', '5', '15', '30', '60', 'D', 'W', 'M')
            start: Optional start datetime (Unix timestamp)
            end: Optional end datetime (Unix timestamp)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to Unix timestamp
            if start is None:
                start = datetime.now() - timedelta(days=365)  # Default to 1 year
            if end is None:
                end = datetime.now()
            
            start_timestamp = int(start.timestamp())
            end_timestamp = int(end.timestamp())
            
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': start_timestamp,
                'to': end_timestamp
            }
            
            data = self._make_api_request('stock/candle', params)
            
            if data.get('s') != 'ok' or not data.get('c'):
                print(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            closes = data.get('c', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            opens = data.get('o', [])
            volumes = data.get('v', [])
            timestamps = data.get('t', [])
            
            for i, timestamp in enumerate(timestamps):
                if i < len(closes):
                    df_data.append({
                        'date': pd.to_datetime(timestamp, unit='s'),
                        'symbol': symbol,
                        'open': opens[i] if i < len(opens) else closes[i],
                        'high': highs[i] if i < len(highs) else closes[i],
                        'low': lows[i] if i < len(lows) else closes[i],
                        'close': closes[i],
                        'volume': volumes[i] if i < len(volumes) else 0
                    })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], resolution: str = "D",
                                start: Optional[datetime] = None, end: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Get stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            resolution: Data resolution
            start: Optional start datetime
            end: Optional end datetime
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        stocks_data = {}
        
        for i, symbol in enumerate(symbols):
            print(f"Fetching data for {symbol} ({i+1}/{len(symbols)})...")
            data = self.get_stock_data(symbol, resolution, start=start, end=end)
            if not data.empty:
                stocks_data[symbol] = data
                print(f"Successfully fetched {len(data)} records for {symbol}")
            else:
                print(f"Failed to fetch data for {symbol}")
        
        return stocks_data
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            params = {'symbol': symbol}
            data = self._make_api_request('stock/profile2', params)
            
            if not data or 'name' not in data:
                print(f"No company info found for symbol: {symbol}")
                return {}
            
            return data
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return {}
    
    def get_news(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Get company news
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of news articles
        """
        try:
            params = {'symbol': symbol}
            
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            data = self._make_api_request('company-news', params)
            
            if not isinstance(data, list):
                print(f"No news found for symbol: {symbol}")
                return []
            
            return data
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            params = {'symbol': symbol}
            data = self._make_api_request('quote', params)
            
            if not data or 'c' not in data:
                print(f"No quote found for symbol: {symbol}")
                return {}
            
            return data
            
        except Exception as e:
            print(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def combine_stocks_data(self, stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple stock DataFrames into a single DataFrame
        
        Args:
            stocks_data: Dictionary mapping symbols to DataFrames
            
        Returns:
            Combined DataFrame
        """
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
        """
        Save stock data to CSV file
        
        Args:
            data: DataFrame containing stock data
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finnhub_stock_data_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        
        print(f"Stock data saved to {filepath}")
        return filepath
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the stock data
        
        Args:
            data: DataFrame containing stock data
            
        Returns:
            Dictionary with statistics
        """
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
            }
        }
        
        return stats
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
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

def main():
    """Main function to run Finnhub data collection"""
    # Check if API key is available
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        print("Please set FINNHUB_API_KEY environment variable")
        print("You can get a free API key from: https://finnhub.io/register")
        return
    
    collector = FinnhubCollector(api_key)
    
    print("Starting Finnhub data collection...")
    
    # Get stock data for all symbols
    stocks_data = collector.get_multiple_stocks_data(STOCK_SYMBOLS, resolution="D")
    
    # Combine all data
    combined_data = collector.combine_stocks_data(stocks_data)
    
    if combined_data.empty:
        print("No data collected. Please check your API key and internet connection.")
        return
    
    # Calculate technical indicators
    enhanced_data = collector.calculate_technical_indicators(combined_data)
    
    # Save to CSV file
    filepath = collector.save_stock_data_to_csv(enhanced_data)
    
    # Print statistics
    stats = collector.get_data_statistics(enhanced_data)
    print("\nCollection Statistics:")
    print(f"Total records: {stats['total_records']}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Symbols: {', '.join(stats['symbols'])}")
    print(f"Average close price: ${stats['price_statistics']['avg_close']:.2f}")
    print(f"Average volume: {stats['volume_statistics']['avg_volume']:,.0f}")
    
    print("\nRecords per symbol:")
    for symbol, count in stats['symbol_counts'].items():
        print(f"  {symbol}: {count} records")

if __name__ == "__main__":
    main()
