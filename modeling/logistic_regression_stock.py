"""
Logistic Regression Model for Binary Stock Classification
Classical machine learning approach for stock direction prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionStockPredictor:
    """
    Logistic Regression model for stock direction prediction
    """
    
    def __init__(self, 
                 feature_selection_method='selectkbest',
                 n_features=20,
                 regularization='l2',
                 C=1.0,
                 max_iter=1000,
                 random_state=42):
        
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.regularization = regularization
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = LogisticRegression(
            C=C,
            penalty=regularization,
            max_iter=max_iter,
            random_state=random_state,
            solver='liblinear' if regularization == 'l1' else 'lbfgs'
        )
        
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Prepare features for logistic regression
        
        Args:
            df: DataFrame with stock data
        """
        # Technical indicators
        feature_cols = [
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
            'open', 'high', 'low', 'close', 'volume',
            'high_low_ratio', 'close_open_ratio', 'volume_price_ratio'
        ]
        
        # Additional engineered features
        df_features = df.copy()
        
        # Price ratios
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_open_ratio'] = df_features['close'] / df_features['open']
        df_features['volume_price_ratio'] = df_features['volume'] / df_features['close']
        
        # Moving averages ratios
        if 'sma_20' in df_features.columns and 'ema_12' in df_features.columns:
            df_features['sma_ema_ratio'] = df_features['sma_20'] / df_features['ema_12']
        
        # Bollinger Band position
        if all(col in df_features.columns for col in ['bb_upper', 'bb_lower', 'close']):
            df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / \
                                       (df_features['bb_upper'] - df_features['bb_lower'])
        
        # RSI momentum
        if 'rsi_14' in df_features.columns:
            df_features['rsi_momentum'] = df_features['rsi_14'].diff()
        
        # MACD momentum
        if 'macd' in df_features.columns:
            df_features['macd_momentum'] = df_features['macd'].diff()
        
        # Combine all feature columns
        all_feature_cols = feature_cols + sentiment_cols + price_cols + [
            'high_low_ratio', 'close_open_ratio', 'volume_price_ratio',
            'sma_ema_ratio', 'bb_position', 'rsi_momentum', 'macd_momentum'
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
    
    def select_features(self, X, y, feature_names):
        """
        Select best features using various methods
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        if self.feature_selection_method == 'selectkbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = self.feature_selector.get_support(indices=True)
            
        elif self.feature_selection_method == 'rfe':
            self.feature_selector = RFE(
                estimator=LogisticRegression(random_state=self.random_state),
                n_features_to_select=self.n_features
            )
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = self.feature_selector.get_support(indices=True)
        
        else:
            # Use all features
            X_selected = X
            self.selected_features = np.arange(X.shape[1])
        
        self.feature_names = [feature_names[i] for i in self.selected_features]
        
        return X_selected
    
    def train_model(self, df, symbol):
        """
        Train logistic regression model
        
        Args:
            df: DataFrame with stock data
            symbol: Stock symbol
        """
        print(f"Training Logistic Regression for {symbol}...")
        
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
        X_train_selected = self.select_features(X_train_scaled, y_train, feature_names)
        X_test_selected = X_test_scaled[:, self.selected_features]
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000, 2000]
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
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
        
        print(f"\nResults for {symbol}:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Selected Features: {self.feature_names}")
        
        return {
            'symbol': symbol,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_scores': cv_scores.tolist(),
            'best_params': grid_search.best_params_,
            'selected_features': self.feature_names,
            'feature_importance': dict(zip(self.feature_names, self.model.coef_[0])),
            'test_predictions': y_pred,
            'test_targets': y_test,
            'test_probabilities': y_pred_proba
        }
    
    def save_model(self, symbol, results):
        """Save the trained model and results"""
        # Save model
        model_path = f'models/logistic_regression_{symbol}.joblib'
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = f'models/scaler_logistic_{symbol}.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature selector
        if self.feature_selector is not None:
            selector_path = f'models/feature_selector_{symbol}.joblib'
            joblib.dump(self.feature_selector, selector_path)
        
        # Save results
        results_path = f'evaluation/logistic_regression_{symbol}.json'
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
            'model_type': 'Logistic Regression'
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Model and results saved for {symbol}")

def train_logistic_regression_models():
    """Train logistic regression models for all symbols"""
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
            
            trainer = LogisticRegressionStockPredictor()
            result = trainer.train_model(symbol_data, symbol)
            
            if result:
                trainer.save_model(symbol, result)
                results[symbol] = result['accuracy']
                
        except Exception as e:
            print(f"Error training {symbol}: {str(e)}")
            continue
    
    # Save overall results
    overall_results = {
        'model_type': 'Logistic Regression',
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'average_accuracy': np.mean(list(results.values())) if results else 0
    }
    
    with open('evaluation/logistic_regression_overall.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nLogistic Regression Training Complete!")
    print(f"Average Accuracy: {overall_results['average_accuracy']:.4f}")
    print(f"Models trained for {len(results)} symbols")

if __name__ == "__main__":
    train_logistic_regression_models()
