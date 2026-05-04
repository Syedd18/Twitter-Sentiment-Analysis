import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API Configuration
TWITTER_CONFIG = {
    'api_key': os.getenv('TWITTER_API_KEY'),
    'api_secret': os.getenv('TWITTER_API_SECRET'),
    'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN')
}

# Stock Data API Configuration (Priority Order)
STOCK_API_CONFIG = {
    'primary': {
        'name': 'Alpha Vantage',
        'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'rate_limit': 5,  # calls per minute
        'enabled': True
    },
    'secondary': {
        'name': 'Finnhub',
        'api_key': os.getenv('FINNHUB_API_KEY'),
        'rate_limit': 60,  # calls per minute
        'enabled': True
    },
    'fallback': {
        'name': 'Yahoo Finance',
        'api_key': None,
        'rate_limit': 100,  # calls per minute
        'enabled': True
    }
}

# Data Storage Configuration
DATA_CONFIG = {
    'data_dir': os.getenv('DATA_DIR', './data'),
    'raw_data_dir': os.getenv('RAW_DATA_DIR', './data/raw'),
    'processed_data_dir': os.getenv('PROCESSED_DATA_DIR', './data/processed'),
    'tweets_file': 'tweets_raw.json',
    'stock_data_file': 'stock_data.csv'
}

# Stock Symbols to Track
STOCK_SYMBOLS = [
    # US Stocks
    'AAPL',  # Apple
    'TSLA',  # Tesla
    'MSFT',  # Microsoft
    'GOOGL', # Google
    'AMZN',  # Amazon
    'META',  # Meta
    'NVDA',  # NVIDIA
    'NFLX',  # Netflix
    # Indian Stocks (NSE)
    'RELIANCE.NS',  # Reliance Industries
    'TCS.NS',       # Tata Consultancy Services
    'HDFCBANK.NS',  # HDFC Bank
    'INFY.NS',      # Infosys
    'HINDUNILVR.NS', # Hindustan Unilever
    'ITC.NS',       # ITC Limited
    'SBIN.NS',      # State Bank of India
    'BHARTIARTL.NS' # Bharti Airtel
]

# Twitter Search Keywords
TWITTER_KEYWORDS = {
    # US Stocks
    'AAPL': ['$AAPL', 'Apple stock', 'AAPL stock', 'Apple Inc'],
    'TSLA': ['$TSLA', 'Tesla stock', 'TSLA stock', 'Tesla Inc', 'Elon Musk'],
    'MSFT': ['$MSFT', 'Microsoft stock', 'MSFT stock', 'Microsoft Corp'],
    'GOOGL': ['$GOOGL', 'Google stock', 'GOOGL stock', 'Alphabet'],
    'AMZN': ['$AMZN', 'Amazon stock', 'AMZN stock', 'Amazon.com'],
    'META': ['$META', 'Meta stock', 'META stock', 'Facebook stock'],
    'NVDA': ['$NVDA', 'NVIDIA stock', 'NVDA stock', 'NVIDIA Corp'],
    'NFLX': ['$NFLX', 'Netflix stock', 'NFLX stock', 'Netflix Inc'],
    # Indian Stocks
    'RELIANCE.NS': ['$RELIANCE', 'Reliance stock', 'RELIANCE stock', 'Reliance Industries'],
    'TCS.NS': ['$TCS', 'TCS stock', 'Tata Consultancy', 'TCS share'],
    'HDFCBANK.NS': ['$HDFC', 'HDFC Bank stock', 'HDFC stock', 'HDFC Bank'],
    'INFY.NS': ['$INFY', 'Infosys stock', 'INFY stock', 'Infosys'],
    'HINDUNILVR.NS': ['$HUL', 'Hindustan Unilever', 'HUL stock', 'Hindustan Unilever stock'],
    'ITC.NS': ['$ITC', 'ITC stock', 'ITC Limited', 'ITC share'],
    'SBIN.NS': ['$SBI', 'State Bank of India', 'SBI stock', 'SBI share'],
    'BHARTIARTL.NS': ['$BHARTI', 'Bharti Airtel', 'BHARTI stock', 'Airtel stock']
}
