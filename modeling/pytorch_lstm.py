"""
Advanced PyTorch LSTM Model with Attention Mechanism
Optimized for GPU training and high accuracy (target: 80%+)
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import sys
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class AdvancedLSTM(nn.Module):
    """Advanced LSTM with attention and multiple layers"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_layers=3, 
                 dropout=0.3, bidirectional=True, use_attention=True):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        current_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(
                    current_size, 
                    hidden_size, 
                    num_layers=1,
                    batch_first=True,
                    dropout=0,
                    bidirectional=bidirectional
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1)))
            self.dropouts.append(nn.Dropout(dropout))
            current_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(current_size)
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(current_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Pass through LSTM layers
        for i, (lstm, batch_norm, dropout) in enumerate(zip(self.lstm_layers, self.batch_norms, self.dropouts)):
            x, _ = lstm(x)
            # Apply batch norm (need to transpose for BatchNorm1d)
            x = x.transpose(1, 2)  # (batch, hidden, seq)
            x = batch_norm(x)
            x = x.transpose(1, 2)  # (batch, seq, hidden)
            x = dropout(x)
        
        # Apply attention if enabled
        if self.use_attention:
            context_vector, attention_weights = self.attention(x)
            output = self.dense_layers(context_vector)
        else:
            # Use last timestep
            output = self.dense_layers(x[:, -1, :])
        
        return output

class StockDataset(torch.utils.data.Dataset):
    """Custom dataset for stock data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_features() -> pd.DataFrame:
    """Load processed features"""
    path = os.path.join(DATA_CONFIG['processed_data_dir'], 'processed_features.parquet')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Processed features not found at {path}")
    return pd.read_parquet(path)

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 80+ advanced technical indicators"""
    df = df.copy()
    
    # Basic price features
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['volume_price_trend'] = df['volume'] * df['price_change']
    
    # Multiple moving averages
    for window in [3, 5, 7, 10, 14, 20, 30, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        df[f'sma_{window}_slope'] = df[f'sma_{window}'].diff()
        df[f'ema_{window}_slope'] = df[f'ema_{window}'].diff()
    
    # Volatility measures
    for window in [3, 5, 10, 20, 30]:
        df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
        df[f'volatility_{window}_normalized'] = df[f'volatility_{window}'] / df['close']
    
    df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
    df['volatility_ratio_10_30'] = df['volatility_10'] / df['volatility_30']
    
    # RSI with multiple periods
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}_normalized'] = (df[f'rsi_{period}'] - 50) / 50
    
    # MACD variations
    for fast, slow in [(12, 26), (8, 21), (5, 13)]:
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
        df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        df[f'macd_histogram_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
    
    # Bollinger Bands with multiple periods
    for period in [10, 20, 30]:
        df[f'bb_middle_{period}'] = df['close'].rolling(period).mean()
        bb_std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
    
    # Volume indicators
    for window in [5, 10, 20]:
        df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
        df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        df[f'volume_price_correlation_{window}'] = df['volume'].rolling(window).corr(df['close'])
    
    # Momentum indicators
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        df[f'roc_{period}'] = df['close'].pct_change(period)
        df[f'price_acceleration_{period}'] = df['price_change'].rolling(period).mean()
    
    # Stochastic Oscillator
    for period in [14, 21]:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
    
    # Williams %R
    for period in [14, 21]:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
    
    # Commodity Channel Index (CCI)
    for period in [14, 20]:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Average True Range (ATR)
    for period in [14, 20]:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df[f'atr_{period}'] = true_range.rolling(period).mean()
        df[f'atr_normalized_{period}'] = df[f'atr_{period}'] / df['close']
    
    # Sentiment features
    if 'avg_sentiment' in df.columns:
        for window in [3, 5, 10, 20]:
            df[f'sentiment_ma_{window}'] = df['avg_sentiment'].rolling(window).mean()
            df[f'sentiment_momentum_{window}'] = df['avg_sentiment'] - df[f'sentiment_ma_{window}']
            df[f'sentiment_volatility_{window}'] = df['avg_sentiment'].rolling(window).std()
        
        if 'pos_rate' in df.columns and 'neg_rate' in df.columns:
            df['sentiment_bias'] = df['pos_rate'] - df['neg_rate']
            df['sentiment_strength'] = df['pos_rate'] + df['neg_rate']
            df['sentiment_ratio'] = df['pos_rate'] / (df['neg_rate'] + 1e-8)
    
    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        if 'avg_sentiment' in df.columns:
            df[f'sentiment_lag_{lag}'] = df['avg_sentiment'].shift(lag)
    
    # Market regime features
    df['trend_short'] = (df['close'] > df['sma_10']).astype(int)
    df['trend_medium'] = (df['close'] > df['sma_20']).astype(int)
    df['trend_long'] = (df['close'] > df['sma_50']).astype(int)
    
    # Handle NaN values in volatility_regime
    df['volatility_regime'] = pd.cut(df['volatility_20'], bins=3, labels=[0, 1, 2])
    df['volatility_regime'] = df['volatility_regime'].fillna(1).astype(int)
    
    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    
    # Handle division by zero in doji calculation
    hl_diff = df['high'] - df['low']
    hl_diff = hl_diff.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    df['doji'] = (abs(df['close'] - df['open']) / hl_diff < 0.1).astype(int)
    df['doji'] = df['doji'].fillna(0).astype(int)
    
    return df

def prepare_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Prepare data for a specific symbol with advanced features"""
    sdf = df[df['symbol'] == symbol].copy()
    if sdf.empty:
        return sdf
    
    sdf['timestamp'] = pd.to_datetime(sdf['timestamp'])
    sdf = sdf.set_index('timestamp').sort_index()
    
    # Create advanced features
    sdf = create_advanced_features(sdf)
    
    # Handle infinite values and NaN values more carefully
    sdf = sdf.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with forward fill, then backward fill, then 0
    sdf = sdf.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Drop rows with too many NaN values (more than 50% NaN)
    sdf = sdf.dropna(thresh=len(sdf.columns) * 0.5)
    
    # Final check for any remaining NaN values
    if sdf.isnull().any().any():
        logger.warning(f"Still have NaN values in {symbol}, filling with 0")
        sdf = sdf.fillna(0)
    
    return sdf

def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i, :])
        y.append(labels[i])
    return np.array(X), np.array(y)

def train_pytorch_lstm(
    sdf: pd.DataFrame,
    symbol: str,
    seq_len: int = 60,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    classification: bool = True
) -> Dict[str, Any]:
    """Train PyTorch LSTM model with advanced features"""
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Using device: {device}")
    
    # Select features (exclude timestamp and target columns)
    exclude_cols = ['timestamp', 'symbol', 'target', 'next_close']
    feature_cols = [col for col in sdf.columns if col not in exclude_cols]
    
    # Create target
    if classification:
        # Multi-step ahead prediction
        next_close = sdf['close'].shift(-1)
        target = (next_close > sdf['close']).astype(int)
        # Drop last row without target
        sdf = sdf.iloc[:-1]
        target = target.iloc[:-1]
    else:
        # Regression target
        next_close = sdf['close'].shift(-1)
        sdf = sdf.iloc[:-1]
        target = next_close.iloc[:-1]
    
    if len(sdf) < seq_len + 20:
        return {'symbol': symbol, 'status': 'insufficient_data', 'n_obs': len(sdf)}
    
    # Prepare features
    feature_data = sdf[feature_cols].copy()
    
    # Handle infinite values and NaN values
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Final check for any remaining NaN values
    if feature_data.isnull().any().any():
        logger.warning(f"Still have NaN values in features for {symbol}, filling with 0")
        feature_data = feature_data.fillna(0)
    
    # Time-based split (80% train, 20% test)
    split_idx = int(0.8 * len(feature_data))
    train_features = feature_data.iloc[:split_idx]
    test_features = feature_data.iloc[split_idx:]
    train_target = target.iloc[:split_idx]
    test_target = target.iloc[split_idx:]
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, train_target.values, seq_len)
    X_test, y_test = create_sequences(test_scaled, test_target.values, seq_len)
    
    if len(X_train) == 0 or len(X_test) == 0:
        return {'symbol': symbol, 'status': 'insufficient_sequences'}
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = AdvancedLSTM(
        input_size=X_train.shape[-1],
        hidden_sizes=[256, 128, 64],
        num_layers=3,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss() if classification else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Class weights for imbalanced data
    if classification:
        # Calculate class weights manually
        class_counts = np.bincount(y_train.astype(int))
        total_samples = len(y_train)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * count) if count > 0 else 0 
            for count in class_counts
        ]).to(device)
        logger.info(f"Class weights: {class_weights}")
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    
    # Training loop
    logger.info(f"Training {symbol} with {len(X_train)} training samples, {len(X_test)} test samples")
    logger.info(f"Features: {X_train.shape[-1]}, Sequence length: {seq_len}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # Apply class weights manually for imbalanced data
            if classification and class_weights is not None:
                # Calculate weighted loss
                batch_targets_int = batch_targets.long()
                weights = class_weights[batch_targets_int]
                loss = criterion(outputs.squeeze(), batch_targets)
                loss = (loss * weights).mean()
            else:
                loss = criterion(outputs.squeeze(), batch_targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            if classification:
                predicted = (outputs.squeeze() > 0.5).float()
                train_correct += (predicted == batch_targets).sum().item()
                train_total += batch_targets.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_targets)
                
                val_loss += loss.item()
                if classification:
                    predicted = (outputs.squeeze() > 0.5).float()
                    val_correct += (predicted == batch_targets).sum().item()
                    val_total += batch_targets.size(0)
                    all_predictions.extend(outputs.squeeze().cpu().numpy())
                    all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        if classification:
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            if len(all_predictions) > 0:
                roc_auc = roc_auc_score(all_targets, all_predictions)
            else:
                roc_auc = 0
        else:
            train_acc = val_acc = roc_auc = 0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(DATA_CONFIG['processed_data_dir'], f'best_{symbol}_pytorch.pth'))
        else:
            patience_counter += 1
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, ROC AUC: {roc_auc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(DATA_CONFIG['processed_data_dir'], f'best_{symbol}_pytorch.pth')))
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features)
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
    
    # Calculate final metrics
    if classification:
        y_pred = (np.array(all_predictions) > 0.5).astype(int)
        accuracy = accuracy_score(all_targets, y_pred)
        precision = precision_score(all_targets, y_pred, average='weighted')
        recall = recall_score(all_targets, y_pred, average='weighted')
        f1 = f1_score(all_targets, y_pred, average='weighted')
        roc_auc = roc_auc_score(all_targets, all_predictions)
        
        final_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
    else:
        mse = np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        
        final_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae)
        }
    
    # Save model
    model_path = os.path.join(DATA_CONFIG['processed_data_dir'], f'lstm_{symbol}_pytorch.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'seq_len': seq_len,
        'model_config': {
            'input_size': X_train.shape[-1],
            'hidden_sizes': [256, 128, 64],
            'num_layers': 3,
            'dropout': 0.3,
            'bidirectional': True,
            'use_attention': True
        }
    }, model_path)
    
    # Prepare results
    results = {
        'symbol': symbol,
        'status': 'ok',
        'n_obs': int(len(sdf)),
        'n_features': int(X_train.shape[-1]),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'model_path': model_path,
        'device': device,
        'epochs_trained': epoch + 1,
        'best_val_loss': float(best_val_loss)
    }
    
    results.update(final_metrics)
    
    return results

def main(
    symbols: List[str] = None,
    seq_len: int = 60,
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    classification: bool = True,
    device: str = 'cuda'
):
    """Main training function"""
    
    logger.info("Loading data...")
    df = load_features()
    
    if symbols is None:
        symbols = sorted(df['symbol'].unique().tolist())
    
    logger.info(f"Training on symbols: {symbols}")
    logger.info(f"Configuration: seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"Device: {device}")
    
    results = []
    for i, symbol in enumerate(symbols):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol} ({i+1}/{len(symbols)})")
        logger.info(f"{'='*60}")
        
        try:
            sdf = prepare_symbol_data(df, symbol)
            if sdf.empty:
                results.append({'symbol': symbol, 'status': 'no_data'})
                continue
            
            result = train_pytorch_lstm(
                sdf, symbol,
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                classification=classification
            )
            results.append(result)
            
            logger.info(f"\n{symbol} Results:")
            for key, value in result.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
                    
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            results.append({'symbol': symbol, 'status': 'error', 'error': str(e)})
    
    # Save results
    out_dir = DATA_CONFIG['processed_data_dir']
    results_path = os.path.join(out_dir, 'pytorch_lstm_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE!")
    logger.info(f"{'='*60}")
    
    # Summary
    successful = [r for r in results if r.get('status') == 'ok']
    if successful:
        accuracies = [r.get('accuracy', 0) for r in successful]
        logger.info(f"Successfully trained: {len(successful)} models")
        logger.info(f"Best accuracy: {max(accuracies):.4f}")
        logger.info(f"Average accuracy: {np.mean(accuracies):.4f}")
        logger.info(f"Results saved to: {results_path}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PyTorch LSTM models with advanced features')
    parser.add_argument('--symbols', nargs='*', default=None, help='Symbols to train (default: all)')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--regression', action='store_true', help='Use regression instead of classification')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else 'cuda'
    
    main(
        symbols=args.symbols,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        classification=not args.regression,
        device=device
    )
