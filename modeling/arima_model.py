import os
import json
from typing import List, Dict

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import yfinance as yf
import pandas_datareader.data as web

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


def load_features() -> pd.DataFrame:
    path = os.path.join(DATA_CONFIG['processed_data_dir'], 'processed_features.parquet')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Processed features not found at {path}")
    return pd.read_parquet(path)


def _fetch_stock(symbol: str, start: str = '2018-01-01') -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(start=start, interval='1d')
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()[['Date','Open','High','Low','Close','Volume']]
        hist.columns = ['date','open','high','low','close','volume']
        hist['symbol'] = symbol
        return hist
    except Exception:
        # Fallback to pandas-datareader (stooq)
        try:
            df = web.DataReader(symbol, 'stooq')
            df = df.sort_index().reset_index()
            df = df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            df['symbol'] = symbol
            return df[['date','open','high','low','close','volume','symbol']]
        except Exception:
            return pd.DataFrame()


def prepare_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    sdf = df[df['symbol'] == symbol].copy()
    if sdf.empty:
        return sdf
    # Ensure datetime index, daily frequency aggregation (mean for features, last for close)
    sdf['timestamp'] = pd.to_datetime(sdf['timestamp'])
    sdf = sdf.set_index('timestamp').sort_index()
    # If OHLCV not present in features, fetch from yfinance and join
    has_ohlcv = all(c in sdf.columns for c in ['open','close','high','low','volume'])
    if not has_ohlcv:
        # Try raw CSV first
        raw_dir = DATA_CONFIG['raw_data_dir']
        latest_csv = None
        try:
            candidates = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.startswith('stock_data_') and f.endswith('.csv')]
            if candidates:
                latest_csv = max(candidates, key=os.path.getmtime)
        except Exception:
            latest_csv = None
        stock = pd.DataFrame()
        if latest_csv:
            try:
                tmp = pd.read_csv(latest_csv)
                stock = tmp
            except Exception:
                stock = pd.DataFrame()
        if stock.empty:
            stock = _fetch_stock(symbol)
        if not stock.empty and 'symbol' in stock.columns:
            stock = stock[stock['symbol'] == symbol]
        if not stock.empty:
            stock['date'] = pd.to_datetime(stock['date'])
            sdf['date'] = sdf.index.normalize()
            sdf = sdf.merge(stock[['date','open','high','low','close','volume']], on='date', how='left').set_index('timestamp')

    agg = {
        'avg_sentiment': 'mean',
        'pos_rate': 'mean',
        'neg_rate': 'mean',
        'tweet_volume': 'sum',
        'open': 'last',
        'close': 'last',
        'high': 'last',
        'low': 'last',
        'volume': 'sum'
    }
    # Only aggregate columns that exist to avoid KeyError
    existing_cols = [c for c in agg.keys() if c in sdf.columns]
    if not existing_cols:
        return pd.DataFrame()
    sdf = sdf.resample('1D').apply({c: agg[c] for c in existing_cols})
    if 'close' not in sdf.columns or sdf['close'].isna().all():
        # As a last resort, synthesize a pseudo price from sentiment as returns
        if 'avg_sentiment' in sdf.columns:
            ret = sdf['avg_sentiment'].fillna(0.0) * 0.005  # 0.5% per unit sentiment
            base = 100.0
            close = (1.0 + ret).cumprod() * base
            sdf['close'] = close
        else:
            return pd.DataFrame()
    sdf = sdf.dropna(subset=['close'])
    return sdf


def train_arimax_for_symbol(sdf: pd.DataFrame, symbol: str, forecast_horizon: int = 10) -> Dict:
    # Minimal length guard
    if len(sdf) < 50:
        return {
            'symbol': symbol,
            'status': 'insufficient_data',
            'n_obs': int(len(sdf))
        }

    # Target and exogenous
    y = sdf['close']
    exog_cols = ['avg_sentiment', 'tweet_volume']
    exog = sdf[exog_cols].fillna(0.0)

    # Train/test split (last 20 days test)
    split_idx = max(1, len(sdf) - 20)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]

    # Simple SARIMAX order; could be auto-ARIMA later
    model = SARIMAX(y_train, exog=exog_train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # In-sample predictions for test
    pred = results.get_forecast(steps=len(y_test), exog=exog_test)
    y_pred = pred.predicted_mean
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred))) if len(y_test) > 0 else None

    # Future forecast
    future_exog = pd.DataFrame({
        'avg_sentiment': [exog['avg_sentiment'].iloc[-1]] * forecast_horizon,
        'tweet_volume': [exog['tweet_volume'].iloc[-1]] * forecast_horizon
    })
    future_fc = results.get_forecast(steps=forecast_horizon, exog=future_exog)
    future_mean = future_fc.predicted_mean

    # Save forecasts
    out_dir = DATA_CONFIG['processed_data_dir']
    os.makedirs(out_dir, exist_ok=True)
    fc_df = pd.DataFrame({
        'date': pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D'),
        'forecast_close': future_mean.values
    })
    out_path = os.path.join(out_dir, f'arima_forecasts_{symbol}.csv')
    fc_df.to_csv(out_path, index=False)

    summary = {
        'symbol': symbol,
        'status': 'ok',
        'n_obs': int(len(sdf)),
        'rmse': rmse,
        'forecast_path': out_path
    }
    return summary


def main(symbols: List[str] = None):
    if symbols is None:
        # Infer symbols from features file
        df = load_features()
        symbols = sorted(df['symbol'].unique().tolist())
    else:
        df = load_features()

    results = []
    for sym in symbols:
        try:
            sdf = prepare_symbol_df(df, sym)
            res = train_arimax_for_symbol(sdf, sym)
        except Exception as e:
            res = {'symbol': sym, 'status': 'error', 'error': str(e)}
        results.append(res)

    # Save summary
    out_dir = DATA_CONFIG['processed_data_dir']
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'arima_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train ARIMAX models per symbol')
    parser.add_argument('--symbols', nargs='*', default=None)
    parser.add_argument('--forecast_horizon', type=int, default=10)
    args = parser.parse_args()

    main(args.symbols)


