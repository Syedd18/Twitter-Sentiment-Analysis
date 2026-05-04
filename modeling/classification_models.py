import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, STOCK_SYMBOLS


class StockDirectionClassifier:
    """
    Classification models to predict stock price direction (up/down)
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        self.scalers = {}
        self.label_encoders = {}
        self.trained_models = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Prepare features for classification
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bollinger_upper'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
        df['bollinger_lower'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
        df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
            df[f'volume_change_lag_{lag}'] = df['volume_change'].shift(lag)
        
        # Target variable: next day direction
        df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
        df['direction'] = (df['next_day_return'] > 0).astype(int)  # 1 for up, 0 for down
        
        # Add symbol
        df['symbol'] = symbol
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def load_and_prepare_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Load stock data and prepare features for all symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Combined DataFrame with features for all symbols
        """
        all_data = []
        
        for symbol in symbols:
            # Try to load processed features first
            processed_path = os.path.join(DATA_CONFIG['processed_data_dir'], 'processed_features.parquet')
            if os.path.exists(processed_path):
                df = pd.read_parquet(processed_path)
                df = df[df['symbol'] == symbol].copy()
                if not df.empty:
                    # Use existing processed data
                    df = self.prepare_features(df, symbol)
                    all_data.append(df)
                    continue
            
            # Fallback: try to load raw stock data
            raw_files = [f for f in os.listdir(DATA_CONFIG['raw_data_dir']) if f.startswith('stock_data_')]
            if raw_files:
                latest_file = max(raw_files, key=lambda x: os.path.getmtime(os.path.join(DATA_CONFIG['raw_data_dir'], x)))
                df = pd.read_csv(os.path.join(DATA_CONFIG['raw_data_dir'], latest_file))
                df = df[df['symbol'] == symbol].copy()
                if not df.empty:
                    df = self.prepare_features(df, symbol)
                    all_data.append(df)
                    continue
            
            print(f"Warning: No data found for symbol {symbol}")
        
        if not all_data:
            raise ValueError("No data found for any symbols")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for classification
        
        Args:
            df: DataFrame with features
            
        Returns:
            X (features), y (targets)
        """
        # Select feature columns
        feature_cols = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'volume_change',
            'sma_5', 'sma_20', 'rsi', 'bollinger_position',
            'volatility_5', 'volatility_20'
        ]
        
        # Add lagged features
        for lag in [1, 2, 3, 5]:
            feature_cols.extend([f'price_change_lag_{lag}', f'volume_change_lag_{lag}'])
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        # Prepare data
        df_clean = df.dropna(subset=available_cols + ['direction'])
        
        X = df_clean[available_cols].values
        y = df_clean['direction'].values
        
        return X, y
    
    def train_models(self, symbols: List[str], test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train classification models for stock direction prediction
        
        Args:
            symbols: List of stock symbols to train on
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"Loading data for symbols: {symbols}")
        df = self.load_and_prepare_data(symbols)
        
        print(f"Preparing training data...")
        X, y = self.prepare_training_data(df)
        
        if len(X) == 0:
            raise ValueError("No valid training data found")
        
        print(f"Training data shape: {X.shape}, Target shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Store trained model
            self.trained_models[name] = pipeline
            
            print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Save models and results
        self._save_models_and_results(results, symbols)
        
        return results
    
    def _save_models_and_results(self, results: Dict[str, Any], symbols: List[str]):
        """Save trained models and results"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save models
        for name, model in self.trained_models.items():
            model_path = os.path.join(models_dir, f'{name}_direction_classifier.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'feature_columns': self.feature_columns,
            'results': results
        }
        
        results_path = os.path.join(models_dir, 'classification_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"Saved results to {results_path}")
    
    def predict_direction(self, symbol: str, model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Predict direction for a specific symbol
        
        Args:
            symbol: Stock symbol
            model_name: Name of the model to use
            
        Returns:
            Dictionary with prediction results
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        # Load latest data for the symbol
        df = self.load_and_prepare_data([symbol])
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Prepare features
        X, _ = self.prepare_training_data(df)
        if len(X) == 0:
            raise ValueError("No valid features found")
        
        # Get latest data point
        latest_features = X[-1:].reshape(1, -1)
        
        # Predict
        model = self.trained_models[model_name]
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'symbol': symbol,
            'prediction': int(prediction),
            'direction': 'up' if prediction == 1 else 'down',
            'confidence': float(probability[1]) if probability is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return result


def train_classification_models(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Train classification models for stock direction prediction
    
    Args:
        symbols: List of symbols to train on. If None, uses default symbols.
        
    Returns:
        Training results
    """
    if symbols is None:
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']  # Default symbols
    
    classifier = StockDirectionClassifier()
    results = classifier.train_models(symbols)
    
    return results


def load_best_model() -> Tuple[str, Any]:
    """
    Load the best performing classification model
    
    Returns:
        Tuple of (model_name, model)
    """
    models_dir = 'models'
    results_path = os.path.join(models_dir, 'classification_results.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError("No classification results found. Train models first.")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Find best model by accuracy
    best_model = None
    best_accuracy = 0
    
    for model_name, metrics in results['results'].items():
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = model_name
    
    if best_model is None:
        raise ValueError("No valid models found in results")
    
    # Load the best model
    model_path = os.path.join(models_dir, f'{best_model}_direction_classifier.joblib')
    model = joblib.load(model_path)
    
    return best_model, model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train classification models for stock direction prediction")
    parser.add_argument("--symbols", nargs='+', default=['AAPL', 'TSLA', 'MSFT', 'GOOGL'],
                       help="Stock symbols to train on")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Fraction of data to use for testing")
    args = parser.parse_args()

    print("Training classification models for stock direction prediction...")
    results = train_classification_models(args.symbols)
    
    print("\nTraining Summary:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
