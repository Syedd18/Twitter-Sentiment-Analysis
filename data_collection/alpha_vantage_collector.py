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

class AlphaVantageCollector:
    def __init__(self, api_key: str = None):
        """
        Initialize Alpha Vantage collector
        
        Args:
            api_key: Alpha Vantage API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.data_dir = DATA_CONFIG['raw_data_dir']
        self._ensure_data_directory()
        
        # Rate limiting: Alpha Vantage free tier allows 5 calls per minute
        self.last_call_time = 0
        self.min_call_interval = 12  # seconds (5 calls per minute = 12 seconds between calls)
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            print(f"Rate limiting: sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_api_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make API request with error handling and rate limiting
        
        Args:
            params: API parameters
            
        Returns:
            API response data
        """
        self._rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise Exception(f"Alpha Vantage API Note: {data['Note']}")
            
            if 'Information' in data:
                raise Exception(f"Alpha Vantage API Information: {data['Information']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {e}")
    
    def get_stock_data(self, symbol: str, outputsize: str = "full", 
                      start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get daily stock data for a specific symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (last 100 data points) or 'full' (full historical data)
            start: Optional start datetime (for filtering)
            end: Optional end datetime (for filtering)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            data = self._make_api_request(params)
            
            if 'Time Series (Daily)' not in data:
                print(f"No time series data found for symbol: {symbol}")
                return pd.DataFrame()
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'date': pd.to_datetime(date_str),
                    'symbol': symbol,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Filter by date range if specified
            if start:
                df = df[df['date'] >= start]
            if end:
                df = df[df['date'] <= end]
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = "5min", 
                         outputsize: str = "compact") -> pd.DataFrame:
        """
        Get intraday stock data
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'compact' (last 100 data points) or 'full' (full historical data)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            data = self._make_api_request(params)
            
            if 'Time Series (5min)' not in data:
                print(f"No intraday data found for symbol: {symbol}")
                return pd.DataFrame()
            
            time_series_key = f'Time Series ({interval})'
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for datetime_str, values in time_series.items():
                df_data.append({
                    'datetime': pd.to_datetime(datetime_str),
                    'symbol': symbol,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], outputsize: str = "full",
                                start: Optional[datetime] = None, end: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Get stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            outputsize: Data output size
            start: Optional start datetime
            end: Optional end datetime
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        stocks_data = {}
        
        for i, symbol in enumerate(symbols):
            print(f"Fetching data for {symbol} ({i+1}/{len(symbols)})...")
            data = self.get_stock_data(symbol, outputsize, start=start, end=end)
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
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'datatype': 'json'
            }
            
            data = self._make_api_request(params)
            
            if 'Symbol' not in data:
                print(f"No company info found for symbol: {symbol}")
                return {}
            
            return data
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return {}
    
    def get_technical_indicators(self, symbol: str, function: str = "SMA", 
                               interval: str = "daily", time_period: int = 20,
                               series_type: str = "close") -> pd.DataFrame:
        """
        Get technical indicators
        
        Args:
            symbol: Stock symbol
            function: Technical indicator function (SMA, EMA, RSI, MACD, etc.)
            interval: Data interval
            time_period: Time period for the indicator
            series_type: Series type (close, open, high, low)
            
        Returns:
            DataFrame with technical indicator data
        """
        try:
            params = {
                'function': function,
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'series_type': series_type,
                'datatype': 'json'
            }
            
            data = self._make_api_request(params)
            
            # The response structure varies by indicator
            if function in data:
                indicator_data = data[function]
                
                df_data = []
                for date_str, value in indicator_data.items():
                    df_data.append({
                        'date': pd.to_datetime(date_str),
                        'symbol': symbol,
                        'indicator': function,
                        'value': float(value)
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('date').reset_index(drop=True)
                
                return df
            else:
                print(f"No {function} data found for symbol: {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching {function} data for {symbol}: {e}")
            return pd.DataFrame()
    
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
            filename = f"alpha_vantage_stock_data_{timestamp}.csv"
        
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
    """Main function to run Alpha Vantage data collection"""
    # Check if API key is available
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
        print("You can get a free API key from: https://www.alphavantage.co/support/#api-key")
        return
    
    collector = AlphaVantageCollector(api_key)
    
    print("Starting Alpha Vantage data collection...")
    
    # Get stock data for all symbols
    stocks_data = collector.get_multiple_stocks_data(STOCK_SYMBOLS, outputsize="full")
    
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
