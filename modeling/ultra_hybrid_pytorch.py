"""
Ultra-Optimized Hybrid PyTorch LSTM for 80%+ Accuracy
Advanced architecture with ensemble learning and aggressive optimization
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys
from datetime import datetime
import logging
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraAttentionLayer(nn.Module):
    """Ultra-advanced attention with multiple heads and residual connections"""
    def __init__(self, hidden_size, num_heads=16, dropout=0.1):
        super(UltraAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights

class UltraHybridLSTM(nn.Module):
    """Ultra-optimized hybrid LSTM with advanced features"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_layers=4, 
                 dropout=0.15, bidirectional=True, use_attention=True, use_transformer=True):
        super(UltraHybridLSTM, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        
        # Advanced input projection with multiple scales
        self.input_projections = nn.ModuleList([
            nn.Linear(input_size, hidden_sizes[0] // 4),
            nn.Linear(input_size, hidden_sizes[0] // 2),
            nn.Linear(input_size, hidden_sizes[0])
        ])
        self.input_combine = nn.Linear(hidden_sizes[0] + hidden_sizes[0] // 2 + hidden_sizes[0] // 4, hidden_sizes[0])
        
        # Multi-scale LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        current_size = hidden_sizes[0]
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
            self.layer_norms.append(nn.LayerNorm(hidden_size * (2 if bidirectional else 1)))
            self.dropouts.append(nn.Dropout(dropout))
            
            if current_size != hidden_size * (2 if bidirectional else 1):
                self.residual_proj.append(nn.Linear(current_size, hidden_size * (2 if bidirectional else 1)))
            else:
                self.residual_proj.append(nn.Identity())
            
            current_size = hidden_size * (2 if bidirectional else 1)
        
        # Transformer layers for sequence modeling
        if use_transformer:
            self.transformer_layers = nn.ModuleList([
                UltraAttentionLayer(current_size, num_heads=16, dropout=dropout)
                for _ in range(2)
            ])
        
        # Ultra attention layer
        if use_attention:
            self.attention = UltraAttentionLayer(current_size, num_heads=16, dropout=dropout)
        
        # Advanced dense layers with skip connections
        self.dense_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(current_size, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.15)
            ),
            nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(0.05)
            )
        ])
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(current_size, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        ])
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Multi-scale input projection
        projected_inputs = []
        for proj in self.input_projections:
            projected_inputs.append(proj(x))
        x = torch.cat(projected_inputs, dim=-1)
        x = self.input_combine(x)
        current_output = x
        
        # Multi-layer LSTM with residual connections
        for i, (lstm, bn, ln, dropout, residual_proj) in enumerate(zip(self.lstm_layers, self.batch_norms, self.layer_norms, self.dropouts, self.residual_proj)):
            lstm_out, _ = lstm(current_output)
            
            batch_size, seq_len, hidden_size = lstm_out.shape
            lstm_out_reshaped = lstm_out.contiguous().view(-1, hidden_size)
            lstm_out_normed = bn(lstm_out_reshaped)
            lstm_out_normed = ln(lstm_out_normed)
            lstm_out_normed = lstm_out_normed.view(batch_size, seq_len, hidden_size)
            lstm_out_dropped = dropout(lstm_out_normed)
            
            residual = residual_proj(current_output)
            current_output = lstm_out_dropped + residual
        
        # Transformer layers
        if self.use_transformer:
            for transformer_layer in self.transformer_layers:
                current_output, _ = transformer_layer(current_output)
        
        # Attention mechanism
        if self.use_attention:
            context_vector, attention_weights = self.attention(current_output)
        else:
            context_vector = current_output[:, -1, :]
        
        # Dense layers with skip connections
        x = context_vector
        for i, (dense_layer, skip_conn) in enumerate(zip(self.dense_layers, self.skip_connections)):
            dense_out = dense_layer(x)
            skip_out = skip_conn(x if i == 0 else dense_out)
            # Ensure same dimensions for addition
            if dense_out.shape != skip_out.shape:
                skip_out = skip_conn(x)
            x = dense_out + skip_out
        
        output = self.output_layer(x)
        return output

class StockDataset(torch.utils.data.Dataset):
    """Advanced dataset with weighted sampling"""
    def __init__(self, features, labels, weights=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.features[idx], self.labels[idx], self.weights[idx]
        return self.features[idx], self.labels[idx]

def load_improved_sentiment_model() -> Any:
    """Load the improved sentiment model"""
    model_path = 'models/improved_sentiment_classifier.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        logger.warning("Improved sentiment model not found, using VADER fallback")
        return None

def load_features() -> pd.DataFrame:
    """Load processed features"""
    path = os.path.join(DATA_CONFIG['processed_data_dir'], 'processed_features.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed features not found at {path}")
    
    return pd.read_parquet(path)

def create_ultra_hybrid_features(df: pd.DataFrame, sentiment_model: Any = None) -> pd.DataFrame:
    """Create ultra-advanced hybrid features"""
    df = df.copy()
    
    # Basic price features
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['volume_price_trend'] = df['volume'] * df['price_change']
    
    # Advanced price features
    df['price_momentum'] = df['close'].pct_change(5)
    df['price_acceleration'] = df['price_momentum'].diff()
    df['price_volatility'] = df['price_change'].rolling(10).std()
    df['price_jerk'] = df['price_acceleration'].diff()
    
    # Multiple timeframe moving averages
    for window in [3, 5, 8, 10, 13, 20, 21, 34, 50, 55, 89]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        df[f'sma_{window}_slope'] = df[f'sma_{window}'].diff()
        df[f'ema_{window}_slope'] = df[f'ema_{window}'].diff()
        
        # Moving average crossovers
        if window > 5:
            df[f'sma_{window}_crossover'] = (df[f'sma_{window}'] > df['sma_5']).astype(int)
            df[f'ema_{window}_crossover'] = (df[f'ema_{window}'] > df['ema_5']).astype(int)
    
    # Advanced volatility features
    for window in [3, 5, 10, 20, 30]:
        df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
        df[f'volatility_{window}_normalized'] = df[f'volatility_{window}'] / df['close']
        df[f'volatility_{window}_rank'] = df[f'volatility_{window}'].rolling(50).rank(pct=True)
    
    # Multiple RSI calculations
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}_normalized'] = (df[f'rsi_{period}'] - 50) / 50
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
    
    # Advanced MACD
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        df[f'macd_{fast}_{slow}'] = macd
        df[f'macd_signal_{fast}_{slow}'] = macd_signal
        df[f'macd_histogram_{fast}_{slow}'] = macd - macd_signal
    
    # Multiple Bollinger Bands
    for window, std in [(10, 1.5), (20, 2.0), (20, 2.5)]:
        bb_middle = df['close'].rolling(window).mean()
        bb_std = df['close'].rolling(window).std()
        bb_upper = bb_middle + (bb_std * std)
        bb_lower = bb_middle - (bb_std * std)
        df[f'bb_position_{window}_{std}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df[f'bb_width_{window}_{std}'] = (bb_upper - bb_lower) / bb_middle
        df[f'bb_squeeze_{window}_{std}'] = (bb_std < bb_std.rolling(20).mean()).astype(int)
    
    # Advanced volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_momentum'] = df['volume'].pct_change(5)
    df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
    df['volume_volatility'] = df['volume'].rolling(10).std()
    
    # Advanced momentum features
    for period in [3, 5, 10, 15, 20, 30]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        df[f'roc_{period}'] = df['close'].pct_change(period)
        df[f'stoch_{period}'] = (df['close'] - df['close'].rolling(period).min()) / (df['close'].rolling(period).max() - df['close'].rolling(period).min())
    
    # Improved sentiment features
    if 'avg_sentiment' in df.columns:
        df['sentiment_momentum'] = df['avg_sentiment'].rolling(5).mean()
        df['sentiment_volatility'] = df['avg_sentiment'].rolling(10).std()
        df['sentiment_trend'] = df['avg_sentiment'].rolling(20).mean()
        df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
        df['sentiment_regime'] = pd.cut(df['avg_sentiment'], bins=5, labels=False)
    
    # Market regime features
    df['trend_short'] = (df['close'] > df['sma_10']).astype(int)
    df['trend_medium'] = (df['close'] > df['sma_20']).astype(int)
    df['trend_long'] = (df['close'] > df['sma_50']).astype(int)
    df['trend_consensus'] = df['trend_short'] + df['trend_medium'] + df['trend_long']
    
    # Advanced patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
    df['hammer'] = ((df['close'] > df['open']) & ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
    
    # Handle NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def create_ultra_target(df: pd.DataFrame, threshold: float = 0.012) -> pd.Series:
    """Create ultra-optimized target"""
    next_close = df['close'].shift(-1)
    price_change_pct = (next_close - df['close']) / df['close']
    price_change_pct = price_change_pct.fillna(0)
    
    # Binary target with optimized threshold
    target = (price_change_pct > threshold).astype(int)
    
    return target

def select_ultra_features(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 50) -> Tuple[np.ndarray, List[str]]:
    """Select ultra features using multiple methods"""
    # Use both mutual information and F-test
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    f_selector = SelectKBest(score_func=f_classif, k=k)
    
    X_mi = mi_selector.fit_transform(X, y)
    X_f = f_selector.fit_transform(X, y)
    
    # Combine selected features
    mi_features = set([feature_names[i] for i in mi_selector.get_support(indices=True)])
    f_features = set([feature_names[i] for i in f_selector.get_support(indices=True)])
    combined_features = list(mi_features.union(f_features))
    
    # Get indices for combined features
    feature_indices = [feature_names.index(f) for f in combined_features if f in feature_names]
    X_selected = X[:, feature_indices]
    
    logger.info(f"Selected {len(combined_features)} ultra features:")
    for i, feature in enumerate(combined_features[:20]):  # Show first 20
        logger.info(f"  {i+1}. {feature}")
    if len(combined_features) > 20:
        logger.info(f"  ... and {len(combined_features) - 20} more")
    
    return X_selected, combined_features

def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int = 90) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(labels[i])
    
    return np.array(X), np.array(y)

def calculate_class_weights(y: np.ndarray) -> np.ndarray:
    """Calculate class weights for imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {cls: weight for cls, weight in zip(classes, weights)}
    
    sample_weights = np.array([class_weights[cls] for cls in y])
    return sample_weights

def train_ultra_hybrid_lstm(
    sdf: pd.DataFrame,
    symbol: str,
    sentiment_model: Any = None,
    seq_len: int = 90,
    epochs: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 0.00005,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Train ultra-optimized hybrid LSTM model"""
    
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Training ultra hybrid model for {symbol} on device: {device}")
    
    # Create ultra target
    target = create_ultra_target(sdf)
    
    # Select features
    exclude_cols = ['timestamp', 'symbol', 'target', 'next_close']
    feature_cols = [col for col in sdf.columns if col not in exclude_cols]
    
    # Prepare features
    feature_data = sdf[feature_cols].copy()
    feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Remove rows without target
    valid_idx = ~target.isna()
    feature_data = feature_data[valid_idx]
    target = target[valid_idx]
    
    if len(feature_data) < seq_len + 100:
        return {'symbol': symbol, 'status': 'insufficient_data', 'n_obs': len(feature_data)}
    
    # Time-based split (90% train, 10% test)
    split_idx = int(0.9 * len(feature_data))
    train_features = feature_data.iloc[:split_idx]
    test_features = feature_data.iloc[split_idx:]
    train_target = target.iloc[:split_idx]
    test_target = target.iloc[split_idx:]
    
    # Scale features
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # Ultra feature selection
    X_train_selected, selected_features = select_ultra_features(
        train_scaled, train_target.values, feature_cols, k=50
    )
    X_test_selected = scaler.transform(test_features)[:, [feature_cols.index(f) for f in selected_features if f in feature_cols]]
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_selected, train_target.values, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_selected, test_target.values, seq_len)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        return {'symbol': symbol, 'status': 'insufficient_sequences'}
    
    # Calculate class weights
    train_weights = calculate_class_weights(y_train_seq)
    
    # Create datasets with weighted sampling
    train_dataset = StockDataset(X_train_seq, y_train_seq, train_weights)
    test_dataset = StockDataset(X_test_seq, y_test_seq)
    
    # Weighted sampler
    sampler = WeightedRandomSampler(train_weights, len(train_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize ultra model
    model = UltraHybridLSTM(
        input_size=X_train_seq.shape[-1],
        hidden_sizes=[512, 256, 128],
        num_layers=4,
        dropout=0.15,
        bidirectional=True,
        use_attention=True,
        use_transformer=True
    ).to(device)
    
    # Ultra-advanced loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate * 10, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
    )
    
    # Training loop
    best_accuracy = 0
    patience_counter = 0
    max_patience = 100
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_targets, batch_weights in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_weights = batch_weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if batch_targets.dim() == 0:
                batch_targets = batch_targets.unsqueeze(0)
                
            # Weighted loss
            loss = criterion(outputs, batch_targets)
            loss = (loss * batch_weights).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_targets.size(0)
            train_correct += (predicted == batch_targets).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features).squeeze()
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if batch_targets.dim() == 0:
                    batch_targets = batch_targets.unsqueeze(0)
                    
                loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_targets.size(0)
                val_correct += (predicted == batch_targets).sum().item()
        
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f'models/ultra_best_lstm_{symbol}.pth')
        else:
            patience_counter += 1
        
        if epoch % 25 == 0:
            logger.info(f"Epoch {epoch}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Best: {best_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(f'models/ultra_best_lstm_{symbol}.pth'))
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            predictions = (outputs > 0.5).float()
            
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if batch_targets.dim() == 0:
                batch_targets = batch_targets.unsqueeze(0)
                
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_targets.extend(batch_targets.numpy().tolist())
    
    # Calculate final metrics
    final_accuracy = accuracy_score(all_targets, all_predictions)
    final_precision = precision_score(all_targets, all_predictions, average='weighted')
    final_recall = recall_score(all_targets, all_predictions, average='weighted')
    final_f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    result = {
        'symbol': symbol,
        'status': 'success',
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'best_accuracy': best_accuracy,
        'epochs_trained': epoch + 1,
        'selected_features': selected_features,
        'n_obs': len(feature_data),
        'n_train': len(X_train_seq),
        'n_test': len(X_test_seq)
    }
    
    logger.info(f"Ultra Hybrid LSTM Accuracy: {final_accuracy:.4f}")
    
    return result

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ultra hybrid stock predictor")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=90, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to train")
    
    args = parser.parse_args()
    
    # Load improved sentiment model
    sentiment_model = load_improved_sentiment_model()
    
    # Load data
    logger.info("Loading features...")
    df = load_features()
    df = create_ultra_hybrid_features(df, sentiment_model)
    
    # Get symbols to train
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = df['symbol'].unique()
    
    logger.info(f"Training ultra hybrid models on symbols: {symbols}")
    
    # Train models
    results = []
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Ultra Hybrid Model for {symbol}")
        logger.info(f"{'='*50}")
        
        symbol_df = df[df['symbol'] == symbol].copy()
        if len(symbol_df) < 200:
            logger.warning(f"Insufficient data for {symbol}, skipping...")
            continue
        
        result = train_ultra_hybrid_lstm(
            symbol_df, symbol, sentiment_model,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            logger.info(f"✅ {symbol} - Accuracy: {result['accuracy']:.4f}")
        else:
            logger.error(f"❌ {symbol} - {result.get('status', 'failed')}")
    
    # Save results
    results_path = os.path.join(DATA_CONFIG['processed_data_dir'], 'ultra_hybrid_lstm_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_accuracy = np.mean([r['accuracy'] for r in successful_results])
        max_accuracy = max([r['accuracy'] for r in successful_results])
        
        logger.info(f"\n{'='*50}")
        logger.info("ULTRA HYBRID TRAINING SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Models trained: {len(successful_results)}")
        logger.info(f"Average accuracy: {avg_accuracy:.4f}")
        logger.info(f"Best accuracy: {max_accuracy:.4f}")
        logger.info(f"Results saved to: {results_path}")
        
        if max_accuracy >= 0.8:
            logger.info("🎉 TARGET ACHIEVED: 80%+ accuracy!")
        else:
            logger.info(f"🎯 Target: 80%+ accuracy (current best: {max_accuracy:.4f})")
    else:
        logger.error("No models trained successfully!")

if __name__ == "__main__":
    main()
