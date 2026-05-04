"""
Hybrid LSTM Model with Sequential & Sentiment Analysis
Combines time series patterns with sentiment features for stock prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AttentionLayer(nn.Module):
    """Multi-head attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_size, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.out_proj(attended)

class HybridLSTMSentiment(nn.Module):
    """
    Hybrid LSTM model combining sequential patterns with sentiment analysis
    """
    
    def __init__(self, 
                 sequence_features=20,  # Technical indicators
                 sentiment_features=5,  # Sentiment features
                 hidden_sizes=[256, 128, 64],
                 num_layers=3,
                 dropout=0.1,
                 bidirectional=True,
                 use_attention=True):
        super(HybridLSTMSentiment, self).__init__()
        
        self.sequence_features = sequence_features
        self.sentiment_features = sentiment_features
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layers for sequential data
        self.lstm_layers = nn.ModuleList()
        input_size = sequence_features
        
        for i, hidden_size in enumerate(hidden_sizes):
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=dropout if i < len(hidden_sizes) - 1 else 0,
                bidirectional=bidirectional
            )
            self.lstm_layers.append(lstm)
            input_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(input_size)
        
        # Sentiment processing branch
        self.sentiment_branch = nn.Sequential(
            nn.Linear(sentiment_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        lstm_output_size = input_size
        sentiment_output_size = 16
        fusion_input_size = lstm_output_size + sentiment_output_size
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequence_data, sentiment_data):
        """
        Forward pass through the hybrid model
        
        Args:
            sequence_data: Tensor of shape (batch_size, seq_len, sequence_features)
            sentiment_data: Tensor of shape (batch_size, sentiment_features)
        """
        batch_size = sequence_data.size(0)
        
        # Process sequential data through LSTM layers
        x = sequence_data
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Apply attention mechanism
        if self.use_attention:
            x = self.attention(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, hidden_size)
        
        # Process sentiment data
        sentiment_out = self.sentiment_branch(sentiment_data)
        
        # Fuse LSTM and sentiment features
        fused_features = torch.cat([x, sentiment_out], dim=1)
        
        # Final processing
        fused_out = self.fusion_layer(fused_features)
        output = self.output_layer(fused_out)
        
        return output.squeeze()

class HybridLSTMSentimentTrainer:
    """Trainer class for Hybrid LSTM with Sentiment Analysis"""
    
    def __init__(self, model_config=None):
        self.model_config = model_config or {
            'sequence_features': 20,
            'sentiment_features': 5,
            'hidden_sizes': [256, 128, 64],
            'num_layers': 3,
            'dropout': 0.1,
            'bidirectional': True,
            'use_attention': True
        }
        
        self.model = None
        self.scaler_sequence = RobustScaler()
        self.scaler_sentiment = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, df, sequence_length=90):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with stock data and sentiment features
            sequence_length: Length of sequences for LSTM
        """
        # Separate sequence features (technical indicators) and sentiment features
        sequence_cols = [
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
            'sma_20', 'ema_12', 'ema_26', 'volume_sma', 'price_change',
            'volatility', 'momentum', 'williams_r', 'cci', 'roc', 'mfi',
            'adx', 'aroon_up', 'aroon_down'
        ]
        
        sentiment_cols = [
            'avg_sentiment', 'sentiment_std', 'sentiment_positive_ratio',
            'sentiment_negative_ratio', 'sentiment_neutral_ratio'
        ]
        
        # Filter available columns
        available_seq_cols = [col for col in sequence_cols if col in df.columns]
        available_sent_cols = [col for col in sentiment_cols if col in df.columns]
        
        # Create sequences
        sequences = []
        sentiment_features = []
        targets = []
        
        for i in range(sequence_length, len(df)):
            # Sequence data (technical indicators)
            seq_data = df[available_seq_cols].iloc[i-sequence_length:i].values
            sequences.append(seq_data)
            
            # Sentiment features (current day)
            sent_data = df[available_sent_cols].iloc[i].values
            sentiment_features.append(sent_data)
            
            # Target (next day price direction)
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i+1] if i+1 < len(df) else current_price
            target = 1 if next_price > current_price else 0
            targets.append(target)
        
        return np.array(sequences), np.array(sentiment_features), np.array(targets)
    
    def train_model(self, df, symbol, epochs=300, batch_size=32, learning_rate=0.0001):
        """
        Train the hybrid LSTM model
        
        Args:
            df: DataFrame with stock data
            symbol: Stock symbol
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        print(f"Training Hybrid LSTM with Sentiment for {symbol}...")
        
        # Prepare data
        sequences, sentiment_features, targets = self.prepare_data(df)
        
        if len(sequences) == 0:
            print(f"No data available for {symbol}")
            return None
        
        # Split data
        X_seq_train, X_seq_test, X_sent_train, X_sent_test, y_train, y_test = train_test_split(
            sequences, sentiment_features, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # Scale features
        X_seq_train_scaled = self.scaler_sequence.fit_transform(
            X_seq_train.reshape(-1, X_seq_train.shape[-1])
        ).reshape(X_seq_train.shape)
        
        X_seq_test_scaled = self.scaler_sequence.transform(
            X_seq_test.reshape(-1, X_seq_test.shape[-1])
        ).reshape(X_seq_test.shape)
        
        X_sent_train_scaled = self.scaler_sentiment.fit_transform(X_sent_train)
        X_sent_test_scaled = self.scaler_sentiment.transform(X_sent_test)
        
        # Convert to tensors
        X_seq_train_tensor = torch.FloatTensor(X_seq_train_scaled).to(self.device)
        X_sent_train_tensor = torch.FloatTensor(X_sent_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        X_seq_test_tensor = torch.FloatTensor(X_seq_test_scaled).to(self.device)
        X_sent_test_tensor = torch.FloatTensor(X_sent_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Initialize model
        self.model = HybridLSTMSentiment(**self.model_config).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate*10, epochs=epochs, steps_per_epoch=len(X_seq_train_tensor)//batch_size
        )
        
        # Training loop
        best_accuracy = 0
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_seq_train_tensor), batch_size):
                batch_seq = X_seq_train_tensor[i:i+batch_size]
                batch_sent = X_sent_train_tensor[i:i+batch_size]
                batch_targets = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq, batch_sent)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_seq_test_tensor, X_sent_test_tensor)
                    val_predictions = (val_outputs > 0.5).float()
                    val_accuracy = accuracy_score(y_test, val_predictions.cpu().numpy())
                    
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_model_state = self.model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    print(f"Epoch {epoch}: Loss={total_loss/len(X_seq_train_tensor)*batch_size:.4f}, "
                          f"Val Accuracy={val_accuracy:.4f}, Best={best_accuracy:.4f}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_seq_test_tensor, X_sent_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            final_accuracy = accuracy_score(y_test, test_predictions.cpu().numpy())
            
            print(f"\nFinal Results for {symbol}:")
            print(f"Accuracy: {final_accuracy:.4f}")
            print(f"Classification Report:")
            print(classification_report(y_test, test_predictions.cpu().numpy()))
        
        return {
            'symbol': symbol,
            'accuracy': final_accuracy,
            'model_config': self.model_config,
            'test_predictions': test_predictions.cpu().numpy(),
            'test_targets': y_test
        }
    
    def save_model(self, symbol, results):
        """Save the trained model and results"""
        if self.model is not None:
            # Save model
            model_path = f'models/hybrid_lstm_sentiment_{symbol}.pth'
            torch.save(self.model.state_dict(), model_path)
            
            # Save scalers
            joblib.dump(self.scaler_sequence, f'models/scaler_sequence_{symbol}.joblib')
            joblib.dump(self.scaler_sentiment, f'models/scaler_sentiment_{symbol}.joblib')
            
            # Save results
            results_path = f'evaluation/hybrid_lstm_sentiment_{symbol}.json'
            os.makedirs('evaluation', exist_ok=True)
            
            results_to_save = {
                'symbol': symbol,
                'accuracy': float(results['accuracy']),
                'model_config': self.model_config,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'Hybrid LSTM with Sentiment Analysis'
            }
            
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            print(f"Model and results saved for {symbol}")

def train_hybrid_lstm_sentiment_models():
    """Train hybrid LSTM with sentiment models for all symbols"""
    import os
    
    # Load processed data
    data_path = 'data/processed/processed_features.parquet'
    if not os.path.exists(data_path):
        print("Processed data not found. Please run data processing first.")
        return
    
    df = pd.read_parquet(data_path)
    symbols = df['symbol'].unique()
    
    trainer = HybridLSTMSentimentTrainer()
    results = {}
    
    for symbol in symbols:
        try:
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(symbol_data) < 200:  # Need sufficient data
                print(f"Insufficient data for {symbol}, skipping...")
                continue
            
            result = trainer.train_model(symbol_data, symbol)
            if result:
                trainer.save_model(symbol, result)
                results[symbol] = result['accuracy']
                
        except Exception as e:
            print(f"Error training {symbol}: {str(e)}")
            continue
    
    # Save overall results
    overall_results = {
        'model_type': 'Hybrid LSTM with Sentiment Analysis',
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'average_accuracy': np.mean(list(results.values())) if results else 0
    }
    
    with open('evaluation/hybrid_lstm_sentiment_overall.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nHybrid LSTM with Sentiment Analysis Training Complete!")
    print(f"Average Accuracy: {overall_results['average_accuracy']:.4f}")
    print(f"Models trained for {len(results)} symbols")

if __name__ == "__main__":
    train_hybrid_lstm_sentiment_models()
