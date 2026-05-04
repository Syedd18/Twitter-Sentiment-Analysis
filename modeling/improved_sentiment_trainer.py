"""
Improved Sentiment Analysis Trainer using Sentiment140 Dataset
Train a high-accuracy sentiment classifier on 1.6M tweets
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedSentimentTrainer:
    """Train improved sentiment classifier on Sentiment140 dataset"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor()
        
    def load_sentiment140_data(self, sample_size: int = 100000) -> pd.DataFrame:
        """Load and preprocess Sentiment140 dataset"""
        logger.info(f"Loading Sentiment140 dataset (sample size: {sample_size:,})...")
        
        # Load data with proper encoding
        df = pd.read_csv(
            'data/datasets/sentiment140/training.1600000.processed.noemoticon.csv',
            header=None,
            names=['target', 'id', 'date', 'query', 'user', 'text'],
            encoding='latin-1'
        )
        
        # Convert target: 0=negative, 4=positive -> 0=negative, 1=positive
        df['target'] = df['target'].map({0: 0, 4: 1})
        
        # Remove neutral tweets (target 2) if any
        df = df[df['target'].isin([0, 1])]
        
        # Sample balanced data
        negative_samples = df[df['target'] == 0].sample(n=sample_size//2, random_state=42)
        positive_samples = df[df['target'] == 1].sample(n=sample_size//2, random_state=42)
        df = pd.concat([negative_samples, positive_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Clean and preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        logger.info(f"Loaded {len(df):,} samples")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple sentiment models and select the best"""
        logger.info("Training sentiment models...")
        
        # Prepare data
        X = df['cleaned_text']
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'gradient_boosting': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]),
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', MultinomialNB())
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('clf', SVC(random_state=42, probability=True))
            ])
        }
        
        # Train and evaluate models
        results = {}
        best_accuracy = 0
        best_model_name = None
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'model': model
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                self.best_model = model
                self.vectorizer = model.named_steps['tfidf']
        
        # Save best model
        if self.best_model:
            model_path = 'models/improved_sentiment_classifier.joblib'
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model ({best_model_name}) saved to {model_path}")
        
        return results
    
    def optimize_best_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize the best model with hyperparameter tuning"""
        logger.info("Optimizing best model...")
        
        X = df['cleaned_text']
        y = df['target']
        
        # Use Logistic Regression as it's usually best for text classification
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'tfidf__max_features': [5000, 10000, 15000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__min_df': [1, 2, 3],
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'clf__C': [0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2']
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Get best model
        self.best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        # Save optimized model
        model_path = 'models/optimized_sentiment_classifier.joblib'
        joblib.dump(self.best_model, model_path)
        logger.info(f"Optimized model saved to {model_path}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'model': self.best_model
        }

class TextPreprocessor:
    """Text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords and short words
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Stem words
        words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train improved sentiment classifier")
    parser.add_argument("--sample_size", type=int, default=100000, help="Sample size from Sentiment140")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ImprovedSentimentTrainer()
    
    # Load data
    df = trainer.load_sentiment140_data(sample_size=args.sample_size)
    
    # Train models
    results = trainer.train_models(df)
    
    # Optimize if requested
    if args.optimize:
        optimization_results = trainer.optimize_best_model(df)
        results['optimization'] = optimization_results
    
    # Save results
    results_path = 'data/processed/improved_sentiment_results.json'
    with open(results_path, 'w') as f:
        # Convert models to strings for JSON serialization
        json_results = {}
        for name, result in results.items():
            if name != 'optimization':
                json_results[name] = {
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1'],
                    'model_type': str(type(result['model']))
                }
            else:
                json_results[name] = {
                    'best_params': optimization_results['best_params'],
                    'best_score': optimization_results['best_score']
                }
        
        json.dump(json_results, f, indent=2)
    
    # Summary
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'] if x[0] != 'optimization' else 0)
    logger.info(f"\n{'='*50}")
    logger.info("SENTIMENT TRAINING SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Best model: {best_model[0]}")
    logger.info(f"Best accuracy: {best_model[1]['accuracy']:.4f}")
    logger.info(f"Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    main()
