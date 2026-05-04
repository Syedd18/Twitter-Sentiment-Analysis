"""
Random Forest Model for Ensemble Stock Prediction
Feature-based ensemble learning approach for stock direction prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RandomForestStockPredictor:
    """
    Random Forest model for stock direction prediction
    """
    
    def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 bootstrap=True,
                 random_state=42,
                 n_jobs=-1):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Prepare comprehensive features for random forest
        
        Args:
            df: DataFrame with stock data
        """
        # Technical indicators
        technical_cols = [
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
            'sma_20', 'ema_12', 'ema_26', 'volume_sma', 'price_change',
            'volatility', 'momentum', 'williams_r', 'cci', 'roc', 'mfi',
            'adx', 'aroon_up', 'aroon_down', 'stoch_k', 'stoch_d',
            'atr', 'obv', 'volume_ratio', 'price_volume_trend'
        ]
        
        # Sentiment features
        sentiment_cols = [
            'avg_sentiment', 'sentiment_std', 'sentiment_positive_ratio',
            'sentiment_negative_ratio', 'sentiment_neutral_ratio'
        ]
        
        # Price-based features
        price_cols = [
            'open', 'high', 'low', 'close', 'volume'
        ]
        
        # Create additional engineered features
        df_features = df.copy()
        
        # Price ratios and relationships
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_open_ratio'] = df_features['close'] / df_features['open']
        df_features['volume_price_ratio'] = df_features['volume'] / df_features['close']
        df_features['high_close_ratio'] = df_features['high'] / df_features['close']
        df_features['low_close_ratio'] = df_features['low'] / df_features['close']
        
        # Moving averages relationships
        if 'sma_20' in df_features.columns and 'ema_12' in df_features.columns:
            df_features['sma_ema_ratio'] = df_features['sma_20'] / df_features['ema_12']
            df_features['price_sma_ratio'] = df_features['close'] / df_features['sma_20']
            df_features['price_ema_ratio'] = df_features['close'] / df_features['ema_12']
        
        # Bollinger Band features
        if all(col in df_features.columns for col in ['bb_upper', 'bb_lower', 'close']):
            df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / \
                                       (df_features['bb_upper'] - df_features['bb_lower'])
            df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['close']
            df_features['bb_squeeze'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['sma_20']
        
        # RSI features
        if 'rsi_14' in df_features.columns:
            df_features['rsi_momentum'] = df_features['rsi_14'].diff()
            df_features['rsi_overbought'] = (df_features['rsi_14'] > 70).astype(int)
            df_features['rsi_oversold'] = (df_features['rsi_14'] < 30).astype(int)
            df_features['rsi_neutral'] = ((df_features['rsi_14'] >= 30) & (df_features['rsi_14'] <= 70)).astype(int)
        
        # MACD features
        if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
            df_features['macd_momentum'] = df_features['macd'].diff()
            df_features['macd_signal_diff'] = df_features['macd'] - df_features['macd_signal']
            df_features['macd_bullish'] = (df_features['macd'] > df_features['macd_signal']).astype(int)
            df_features['macd_bearish'] = (df_features['macd'] < df_features['macd_signal']).astype(int)
        
        # Volume features
        if 'volume' in df_features.columns and 'volume_sma' in df_features.columns:
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
            df_features['volume_spike'] = (df_features['volume'] > df_features['volume_sma'] * 1.5).astype(int)
            df_features['volume_dry'] = (df_features['volume'] < df_features['volume_sma'] * 0.5).astype(int)
        
        # Price momentum features
        df_features['price_momentum_1'] = df_features['close'].pct_change(1)
        df_features['price_momentum_3'] = df_features['close'].pct_change(3)
        df_features['price_momentum_5'] = df_features['close'].pct_change(5)
        df_features['price_momentum_10'] = df_features['close'].pct_change(10)
        
        # Volatility features
        df_features['volatility_3'] = df_features['close'].rolling(3).std()
        df_features['volatility_5'] = df_features['close'].rolling(5).std()
        df_features['volatility_10'] = df_features['close'].rolling(10).std()
        
        # Trend features
        df_features['trend_3'] = df_features['close'].rolling(3).mean()
        df_features['trend_5'] = df_features['close'].rolling(5).mean()
        df_features['trend_10'] = df_features['close'].rolling(10).mean()
        
        # Combine all feature columns
        all_feature_cols = technical_cols + sentiment_cols + price_cols + [
            'high_low_ratio', 'close_open_ratio', 'volume_price_ratio', 'high_close_ratio', 'low_close_ratio',
            'sma_ema_ratio', 'price_sma_ratio', 'price_ema_ratio',
            'bb_position', 'bb_width', 'bb_squeeze',
            'rsi_momentum', 'rsi_overbought', 'rsi_oversold', 'rsi_neutral',
            'macd_momentum', 'macd_signal_diff', 'macd_bullish', 'macd_bearish',
            'volume_ratio', 'volume_spike', 'volume_dry',
            'price_momentum_1', 'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
            'volatility_3', 'volatility_5', 'volatility_10',
            'trend_3', 'trend_5', 'trend_10'
        ]
        
        # Filter available columns
        available_cols = [col for col in all_feature_cols if col in df_features.columns]
        
        # Remove any remaining NaN values
        df_features = df_features.dropna(subset=available_cols)
        
        return df_features[available_cols], available_cols
    
    def create_targets(self, df):
        """
        Create binary targets for classification
        
        Args:
            df: DataFrame with stock data
        """
        targets = []
        
        for i in range(len(df) - 1):
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            
            # Binary classification: 1 if price goes up, 0 if down
            target = 1 if next_price > current_price else 0
            targets.append(target)
        
        return np.array(targets)
    
    def train_model(self, df, symbol):
        """
        Train random forest model
        
        Args:
            df: DataFrame with stock data
            symbol: Stock symbol
        """
        print(f"Training Random Forest for {symbol}...")
        
        # Prepare features and targets
        X, feature_names = self.prepare_features(df)
        y = self.create_targets(df)
        
        # Align features and targets (remove last row from features)
        X = X[:-1]
        
        if len(X) < 50:  # Need sufficient data
            print(f"Insufficient data for {symbol}, skipping...")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        feature_selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_names)))
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        selected_features = feature_selector.get_support(indices=True)
        self.feature_names = [feature_names[i] for i in selected_features]
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5]
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train_selected, y_train)
        
        # Use best parameters
        self.model = grid_search.best_estimator_
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_selected, y_train, cv=5, scoring='accuracy'
        )
        
        # Final predictions
        y_pred = self.model.predict(X_test_selected)
        y_pred_proba = self.model.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        print(f"\nResults for {symbol}:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Selected Features: {len(self.feature_names)}")
        print(f"Top 5 Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.4f}")
        
        return {
            'symbol': symbol,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_scores': cv_scores.tolist(),
            'best_params': grid_search.best_params_,
            'selected_features': self.feature_names,
            'feature_importance': feature_importance,
            'test_predictions': y_pred,
            'test_targets': y_test,
            'test_probabilities': y_pred_proba
        }
    
    def save_model(self, symbol, results):
        """Save the trained model and results"""
        # Save model
        model_path = f'models/random_forest_{symbol}.joblib'
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = f'models/scaler_rf_{symbol}.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save results
        results_path = f'evaluation/random_forest_{symbol}.json'
        import os
        os.makedirs('evaluation', exist_ok=True)
        
        results_to_save = {
            'symbol': symbol,
            'accuracy': float(results['accuracy']),
            'auc_score': float(results['auc_score']),
            'cv_scores': results['cv_scores'],
            'best_params': results['best_params'],
            'selected_features': results['selected_features'],
            'feature_importance': results['feature_importance'],
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Random Forest'
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Model and results saved for {symbol}")

def train_random_forest_models():
    """Train random forest models for all symbols"""
    import os
    
    # Load processed data
    data_path = 'data/processed/processed_features.parquet'
    if not os.path.exists(data_path):
        print("Processed data not found. Please run data processing first.")
        return
    
    df = pd.read_parquet(data_path)
    symbols = df['symbol'].unique()
    
    results = {}
    
    for symbol in symbols:
        try:
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(symbol_data) < 200:  # Need sufficient data
                print(f"Insufficient data for {symbol}, skipping...")
                continue
            
            trainer = RandomForestStockPredictor()
            result = trainer.train_model(symbol_data, symbol)
            
            if result:
                trainer.save_model(symbol, result)
                results[symbol] = result['accuracy']
                
        except Exception as e:
            print(f"Error training {symbol}: {str(e)}")
            continue
    
    # Save overall results
    overall_results = {
        'model_type': 'Random Forest',
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'average_accuracy': np.mean(list(results.values())) if results else 0
    }
    
    with open('evaluation/random_forest_overall.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nRandom Forest Training Complete!")
    print(f"Average Accuracy: {overall_results['average_accuracy']:.4f}")
    print(f"Models trained for {len(results)} symbols")

if __name__ == "__main__":
    train_random_forest_models()
