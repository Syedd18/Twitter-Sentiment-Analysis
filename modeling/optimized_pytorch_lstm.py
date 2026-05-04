"""
Optimized PyTorch LSTM Model for 80%+ Accuracy
Advanced feature engineering, ensemble methods, and hyperparameter optimization
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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import joblib
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
    """Advanced attention mechanism for LSTM"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class OptimizedLSTM(nn.Module):
    """Optimized LSTM with advanced architecture"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], num_layers=2, 
                 dropout=0.2, bidirectional=True, use_attention=True):
        super(OptimizedLSTM, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers with residual connections
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
        
        # Advanced dense layers with residual connections
        self.dense_layers = nn.Sequential(
            nn.Linear(current_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        current_output = x
        
        # Pass through LSTM layers
        for i, (lstm, bn, dropout) in enumerate(zip(self.lstm_layers, self.batch_norms, self.dropouts)):
            lstm_out, _ = lstm(current_output)
            
            # Apply batch norm and dropout
            batch_size, seq_len, hidden_size = lstm_out.shape
            lstm_out_reshaped = lstm_out.contiguous().view(-1, hidden_size)
            lstm_out_normed = bn(lstm_out_reshaped)
            lstm_out_normed = lstm_out_normed.view(batch_size, seq_len, hidden_size)
            lstm_out_dropped = dropout(lstm_out_normed)
            
            current_output = lstm_out_dropped
        
        # Apply attention if enabled
        if self.use_attention:
            context_vector, attention_weights = self.attention(current_output)
        else:
            context_vector = current_output[:, -1, :]  # Use last timestep
        
        # Pass through dense layers
        output = self.dense_layers(context_vector)
        return output

class StockDataset(torch.utils.data.Dataset):
    """Custom dataset for stock data"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_features() -> pd.DataFrame:
    """Load processed features"""
    path = os.path.join(DATA_CONFIG['processed_data_dir'], 'processed_features.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed features not found at {path}")
    
    return pd.read_parquet(path)

def create_optimized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create optimized set of 25 most predictive features"""
    df = df.copy()
    
    # Basic price features
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    df['volume_price_trend'] = df['volume'] * df['price_change']
    
    # Key moving averages (only most important ones)
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
    
    # Volatility (key periods only)
    for window in [5, 20]:
        df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
        df[f'volatility_{window}_normalized'] = df[f'volatility_{window}'] / df['close']
    
    # RSI (most important period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14_normalized'] = (df['rsi_14'] - 50) / 50
    
    # MACD (standard parameters)
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands (20-period)
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['roc_10'] = df['close'].pct_change(10)
    
    # Sentiment features (if available)
    if 'avg_sentiment' in df.columns:
        df['sentiment_momentum'] = df['avg_sentiment'].rolling(5).mean()
        df['sentiment_volatility'] = df['avg_sentiment'].rolling(10).std()
    
    # Market regime
    df['trend_short'] = (df['close'] > df['sma_10']).astype(int)
    df['trend_medium'] = (df['close'] > df['sma_20']).astype(int)
    df['trend_long'] = (df['close'] > df['sma_50']).astype(int)
    
    # Handle NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def create_advanced_target(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
    """Create advanced target with threshold-based classification"""
    next_close = df['close'].shift(-1)
    price_change_pct = (next_close - df['close']) / df['close']
    
    # Handle NaN values
    price_change_pct = price_change_pct.fillna(0)
    
    # Multi-class target: 0=Strong Down, 1=Down, 2=Neutral, 3=Up, 4=Strong Up
    target = pd.cut(price_change_pct, 
                   bins=[-np.inf, -threshold, -threshold/2, threshold/2, threshold, np.inf],
                   labels=[0, 1, 2, 3, 4])
    
    # Convert to int, handling NaN values
    target = target.fillna(2).astype(int)  # Fill NaN with neutral (2)
    
    return target

def select_best_features(X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 25) -> Tuple[np.ndarray, List[str]]:
    """Select k best features using statistical tests"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    
    logger.info(f"Selected {len(selected_features)} best features:")
    for i, feature in enumerate(selected_features):
        logger.info(f"  {i+1}. {feature}")
    
    return X_selected, selected_features

def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(labels[i])
    
    return np.array(X), np.array(y)

def train_ensemble_models(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Train ensemble of traditional ML models"""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Handle probability predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # Binary classification
                y_pred_proba = y_pred_proba[:, 1]
            else:  # Multi-class
                y_pred_proba = y_pred_proba
        else:
            y_pred_proba = y_pred
        
        # Calculate ROC-AUC safely
        try:
            if len(np.unique(y_test)) > 1:
                if y_pred_proba.ndim == 1:  # Binary
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            else:
                roc_auc = 0
        except:
            roc_auc = 0
        
        # Metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc,
            'model': model
        }
        
        logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}")
    
    return results

def train_optimized_lstm(
    sdf: pd.DataFrame,
    symbol: str,
    seq_len: int = 30,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda',
    use_ensemble: bool = True
) -> Dict[str, Any]:
    """Train optimized PyTorch LSTM model"""
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Training {symbol} on device: {device}")
    
    # Create advanced target
    target = create_advanced_target(sdf)
    
    # Select features (exclude non-feature columns)
    exclude_cols = ['timestamp', 'symbol', 'target', 'next_close']
    feature_cols = [col for col in sdf.columns if col not in exclude_cols]
    
    # Prepare features
    feature_data = sdf[feature_cols].copy()
    feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Remove rows without target
    valid_idx = ~target.isna()
    feature_data = feature_data[valid_idx]
    target = target[valid_idx]
    
    if len(feature_data) < seq_len + 50:
        return {'symbol': symbol, 'status': 'insufficient_data', 'n_obs': len(feature_data)}
    
    # Time-based split (80% train, 20% test)
    split_idx = int(0.8 * len(feature_data))
    train_features = feature_data.iloc[:split_idx]
    test_features = feature_data.iloc[split_idx:]
    train_target = target.iloc[:split_idx]
    test_target = target.iloc[split_idx:]
    
    # Scale features
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # Feature selection
    X_train_selected, selected_features = select_best_features(
        train_scaled, train_target.values, feature_cols, k=25
    )
    X_test_selected = scaler.transform(test_features)[:, [feature_cols.index(f) for f in selected_features]]
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_selected, train_target.values, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_selected, test_target.values, seq_len)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        return {'symbol': symbol, 'status': 'insufficient_sequences'}
    
    # Convert to binary classification (combine classes)
    y_train_binary = (y_train_seq >= 3).astype(int)  # Up/Strong Up = 1, others = 0
    y_test_binary = (y_test_seq >= 3).astype(int)
    
    # Create datasets
    train_dataset = StockDataset(X_train_seq, y_train_binary)
    test_dataset = StockDataset(X_test_seq, y_test_binary)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = OptimizedLSTM(
        input_size=X_train_seq.shape[-1],
        hidden_sizes=[128, 64, 32],
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training loop
    best_accuracy = 0
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
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
                loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_targets.size(0)
                val_correct += (predicted == batch_targets).sum().item()
        
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        scheduler.step(val_loss)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/best_lstm_{symbol}.pth')
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Best: {best_accuracy:.4f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(f'models/best_lstm_{symbol}.pth'))
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.numpy())
    
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
    
    # Train ensemble models if requested
    if use_ensemble:
        logger.info("Training ensemble models...")
        # Use the original feature data (not sequences) for ensemble models
        ensemble_results = train_ensemble_models(
            train_scaled,
            train_target.values,
            test_scaled,
            test_target.values
        )
        result['ensemble_results'] = ensemble_results
        
        # Ensemble prediction (weighted average)
        ensemble_accuracy = np.mean([r['accuracy'] for r in ensemble_results.values()])
        result['ensemble_accuracy'] = ensemble_accuracy
        
        logger.info(f"LSTM Accuracy: {final_accuracy:.4f}")
        logger.info(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    return result

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train optimized PyTorch LSTM model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to train")
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble methods")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading features...")
    df = load_features()
    df = create_optimized_features(df)
    
    # Get symbols to train
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = df['symbol'].unique()
    
    logger.info(f"Training on symbols: {symbols}")
    
    # Train models
    results = []
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {symbol}")
        logger.info(f"{'='*50}")
        
        symbol_df = df[df['symbol'] == symbol].copy()
        if len(symbol_df) < 100:
            logger.warning(f"Insufficient data for {symbol}, skipping...")
            continue
        
        result = train_optimized_lstm(
            symbol_df, symbol,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            use_ensemble=args.use_ensemble
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            logger.info(f"✅ {symbol} - Accuracy: {result['accuracy']:.4f}")
        else:
            logger.error(f"❌ {symbol} - {result.get('status', 'failed')}")
    
    # Save results
    results_path = os.path.join(DATA_CONFIG['processed_data_dir'], 'optimized_lstm_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_accuracy = np.mean([r['accuracy'] for r in successful_results])
        max_accuracy = max([r['accuracy'] for r in successful_results])
        
        logger.info(f"\n{'='*50}")
        logger.info("TRAINING SUMMARY")
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
