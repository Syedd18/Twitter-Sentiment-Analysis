"""
Naive Bayes Model for Text-Based Sentiment Classification
Specialized sentiment analysis model for stock-related text data
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime
import re
import string
import warnings
warnings.filterwarnings('ignore')

class NaiveBayesSentimentClassifier:
    """
    Naive Bayes model for sentiment classification of stock-related text
    """
    
    def __init__(self, 
                 model_type='multinomial',
                 vectorizer_type='tfidf',
                 max_features=10000,
                 ngram_range=(1, 2),
                 min_df=2,
                 max_df=0.95,
                 random_state=42):
        
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        
        # Initialize vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
        
        # Initialize model
        if model_type == 'multinomial':
            self.model = MultinomialNB()
        elif model_type == 'gaussian':
            self.model = GaussianNB()
        elif model_type == 'bernoulli':
            self.model = BernoulliNB()
        else:
            raise ValueError("Model type must be 'multinomial', 'gaussian', or 'bernoulli'")
        
        self.label_encoder = LabelEncoder()
        
    def preprocess_text(self, text):
        """
        Preprocess text data for sentiment analysis
        
        Args:
            text: Raw text string
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove stock symbols (e.g., $AAPL, $TSLA)
        text = re.sub(r'\$[A-Z]{1,5}', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for sentiment indicators
        text = re.sub(r'[^\w\s!?]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def load_sentiment_data(self):
        """
        Load sentiment data from various sources
        """
        texts = []
        labels = []
        
        # Load Sentiment140 dataset if available
        sentiment140_path = 'data/datasets/sentiment140/training.1600000.processed.noemoticon.csv'
        if pd.io.common.file_exists(sentiment140_path):
            try:
                df_sentiment140 = pd.read_csv(sentiment140_path, encoding='latin-1', header=None)
                df_sentiment140.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
                
                # Map targets: 0=negative, 2=neutral, 4=positive
                df_sentiment140['sentiment'] = df_sentiment140['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
                
                # Sample data for training (too large otherwise)
                df_sample = df_sentiment140.sample(n=min(50000, len(df_sentiment140)), random_state=self.random_state)
                
                texts.extend(df_sample['text'].tolist())
                labels.extend(df_sample['sentiment'].tolist())
                
                print(f"Loaded {len(df_sample)} samples from Sentiment140 dataset")
                
            except Exception as e:
                print(f"Error loading Sentiment140 dataset: {str(e)}")
        
        # Load processed sentiment data if available
        sentiment_data_path = 'data/processed/sentiment_scored.csv'
        if pd.io.common.file_exists(sentiment_data_path):
            try:
                df_sentiment = pd.read_csv(sentiment_data_path)
                
                if 'text' in df_sentiment.columns and 'sentiment' in df_sentiment.columns:
                    # Filter out NaN values
                    df_sentiment = df_sentiment.dropna(subset=['text', 'sentiment'])
                    
                    texts.extend(df_sentiment['text'].tolist())
                    labels.extend(df_sentiment['sentiment'].tolist())
                    
                    print(f"Loaded {len(df_sentiment)} samples from processed sentiment data")
                
            except Exception as e:
                print(f"Error loading processed sentiment data: {str(e)}")
        
        # Load Vader sentiment data if available
        vader_data_path = 'data/processed/vader_sentiment_scored.csv'
        if pd.io.common.file_exists(vader_data_path):
            try:
                df_vader = pd.read_csv(vader_data_path)
                
                if 'text' in df_vader.columns and 'sentiment' in df_vader.columns:
                    df_vader = df_vader.dropna(subset=['text', 'sentiment'])
                    
                    texts.extend(df_vader['text'].tolist())
                    labels.extend(df_vader['sentiment'].tolist())
                    
                    print(f"Loaded {len(df_vader)} samples from Vader sentiment data")
                
            except Exception as e:
                print(f"Error loading Vader sentiment data: {str(e)}")
        
        # Create synthetic data if no data available
        if not texts:
            print("No sentiment data found, creating synthetic data...")
            synthetic_texts, synthetic_labels = self.create_synthetic_data()
            texts.extend(synthetic_texts)
            labels.extend(synthetic_labels)
        
        return texts, labels
    
    def create_synthetic_data(self):
        """
        Create synthetic sentiment data for training
        """
        positive_texts = [
            "This stock is going to the moon! Great earnings report.",
            "Amazing growth potential, buying more shares.",
            "Excellent management team and strong fundamentals.",
            "Bullish on this company, expecting big gains.",
            "Strong buy recommendation, undervalued stock.",
            "Great quarterly results, revenue up significantly.",
            "Innovative product launch, market leader.",
            "Positive outlook, strong balance sheet.",
            "Outstanding performance, exceeding expectations.",
            "Fantastic opportunity, high growth potential."
        ]
        
        negative_texts = [
            "This stock is crashing, terrible earnings.",
            "Poor management, selling all shares.",
            "Weak fundamentals, avoid this stock.",
            "Bearish outlook, expecting losses.",
            "Strong sell recommendation, overvalued.",
            "Disappointing quarterly results, revenue down.",
            "Failed product launch, losing market share.",
            "Negative outlook, weak balance sheet.",
            "Underperforming, missing expectations.",
            "Risky investment, declining growth."
        ]
        
        neutral_texts = [
            "Stock price remains stable, no major changes.",
            "Company reported average quarterly results.",
            "Market conditions are uncertain, holding position.",
            "Mixed signals, waiting for more information.",
            "Neutral outlook, maintaining current position.",
            "Standard performance, meeting expectations.",
            "Regular trading day, no significant news.",
            "Balanced view, neither bullish nor bearish.",
            "Steady performance, consistent with market.",
            "Normal market activity, no surprises."
        ]
        
        # Expand synthetic data
        texts = []
        labels = []
        
        for text in positive_texts * 100:
            texts.append(text)
            labels.append('positive')
        
        for text in negative_texts * 100:
            texts.append(text)
            labels.append('negative')
        
        for text in neutral_texts * 100:
            texts.append(text)
            labels.append('neutral')
        
        return texts, labels
    
    def train_model(self):
        """
        Train Naive Bayes sentiment classifier
        """
        print(f"Training Naive Bayes ({self.model_type}) Sentiment Classifier...")
        
        # Load data
        texts, labels = self.load_sentiment_data()
        
        if not texts:
            print("No data available for training")
            return None
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
        processed_texts = [processed_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        print(f"Training on {len(processed_texts)} text samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        # Hyperparameter tuning
        if self.model_type == 'multinomial':
            param_grid = {
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0],
                'classifier__fit_prior': [True, False]
            }
        elif self.model_type == 'bernoulli':
            param_grid = {
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0],
                'classifier__binarize': [0.0, 0.5, 1.0]
            }
        else:  # gaussian
            param_grid = {
                'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Use best parameters
        self.pipeline = grid_search.best_estimator_
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, cv=5, scoring='accuracy'
        )
        
        # Final predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nResults:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'cv_scores': cv_scores.tolist(),
            'best_params': grid_search.best_params_,
            'test_predictions': y_pred,
            'test_targets': y_test,
            'model_type': self.model_type,
            'vectorizer_type': self.vectorizer_type
        }
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text: Text to analyze
        """
        if not hasattr(self, 'pipeline'):
            print("Model not trained yet")
            return None
        
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        
        return {
            'prediction': prediction,
            'probabilities': dict(zip(self.pipeline.classes_, probabilities))
        }
    
    def save_model(self, results):
        """Save the trained model and results"""
        # Save model
        model_path = f'models/naive_bayes_sentiment_{self.model_type}.joblib'
        joblib.dump(self.pipeline, model_path)
        
        # Save results
        results_path = f'evaluation/naive_bayes_sentiment_{self.model_type}.json'
        import os
        os.makedirs('evaluation', exist_ok=True)
        
        results_to_save = {
            'accuracy': float(results['accuracy']),
            'cv_scores': results['cv_scores'],
            'best_params': results['best_params'],
            'model_type': results['model_type'],
            'vectorizer_type': results['vectorizer_type'],
            'timestamp': datetime.now().isoformat(),
            'model_name': f'Naive Bayes ({self.model_type.title()}) Sentiment Classifier'
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Model and results saved for {self.model_type} Naive Bayes")

def train_naive_bayes_models():
    """Train different Naive Bayes models for sentiment classification"""
    import os
    
    models_to_train = [
        {'model_type': 'multinomial', 'vectorizer_type': 'tfidf'},
        {'model_type': 'bernoulli', 'vectorizer_type': 'tfidf'},
        {'model_type': 'gaussian', 'vectorizer_type': 'tfidf'}
    ]
    
    results = {}
    
    for config in models_to_train:
        try:
            trainer = NaiveBayesSentimentClassifier(**config)
            result = trainer.train_model()
            
            if result:
                trainer.save_model(result)
                results[config['model_type']] = result['accuracy']
                
        except Exception as e:
            print(f"Error training {config['model_type']} model: {str(e)}")
            continue
    
    # Save overall results
    overall_results = {
        'model_type': 'Naive Bayes Sentiment Classification',
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'average_accuracy': np.mean(list(results.values())) if results else 0
    }
    
    with open('evaluation/naive_bayes_sentiment_overall.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"\nNaive Bayes Sentiment Classification Training Complete!")
    print(f"Average Accuracy: {overall_results['average_accuracy']:.4f}")
    print(f"Models trained: {list(results.keys())}")

if __name__ == "__main__":
    train_naive_bayes_models()
