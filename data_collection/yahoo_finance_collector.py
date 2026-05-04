import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Any, Optional
import pandas_datareader.data as web

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, STOCK_SYMBOLS

class YahooFinanceCollector:
    def __init__(self):
        """Initialize Yahoo Finance collector"""
        self.data_dir = DATA_CONFIG['raw_data_dir']
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d",
                       start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get stock data for a specific symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Optional start datetime (overrides period if provided)
            end: Optional end datetime (defaults to now if start provided and end is None)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            if start is not None:
                _end = end or datetime.now()
                data = ticker.history(start=start, end=_end, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)

            # Fallback to stooq if yfinance empty/blocked
            if data.empty:
                try:
                    stooq = web.DataReader(symbol, 'stooq')
                    if not stooq.empty:
                        data = stooq.sort_index()
                except Exception as _:
                    data = pd.DataFrame()

            if data.empty:
                print(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Reset index to make Date a column
            data = data.reset_index()
            # Normalize column names across providers
            if 'Date' in data.columns:
                data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                # Add missing columns
                data['dividends'] = 0.0
                data['stock_splits'] = 0.0
            else:
                data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'symbol']
            
            # Select only relevant columns
            data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks_data(self, symbols: List[str], period: str = "1y", interval: str = "1d",
                                 start: Optional[datetime] = None, end: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Get stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        stocks_data = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.get_stock_data(symbol, period, interval, start=start, end=end)
            if not data.empty:
                stocks_data[symbol] = data
                print(f"Successfully fetched {len(data)} records for {symbol}")
            else:
                print(f"Failed to fetch data for {symbol}")
        
        return stocks_data
    
    def get_intraday_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """
        Get intraday data (1-minute intervals) for recent days
        
        Args:
            symbol: Stock symbol
            days: Number of recent days to fetch
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get data for the specified number of days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = ticker.history(start=start_date, end=end_date, interval="1m")
            
            if data.empty:
                print(f"No intraday data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns
            data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'symbol']
            
            # Select only relevant columns
            data = data[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            return data
            
        except Exception as e:
            print(f"Error fetching intraday data for {symbol}: {e}")
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
            filename = f"stock_data_{timestamp}.csv"
        
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
    """Main function to run Yahoo Finance data collection"""
    collector = YahooFinanceCollector()
    
    print("Starting Yahoo Finance data collection...")
    
    # Get stock data for all symbols
    stocks_data = collector.get_multiple_stocks_data(STOCK_SYMBOLS, period="1y", interval="1d")
    
    # Combine all data
    combined_data = collector.combine_stocks_data(stocks_data)
    
    if combined_data.empty:
        print("No data collected. Please check your internet connection and symbol validity.")
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
