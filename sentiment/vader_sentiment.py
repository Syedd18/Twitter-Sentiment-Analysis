import os
import json
from typing import Optional, Dict, Any
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


class VADERSentimentAnalyzer:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer
    Specifically designed for social media text with emojis, slang, and informal language
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores: {'compound', 'pos', 'neu', 'neg'}
        """
        if not text or pd.isna(text):
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        scores = self.analyzer.polarity_scores(str(text))
        return scores
    
    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score
        
        Args:
            compound_score: VADER compound score (-1 to 1)
            
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_batch(self, texts: pd.Series) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: Pandas Series of text strings
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for text in texts:
            scores = self.analyze_text(text)
            sentiment = self.classify_sentiment(scores['compound'])
            
            results.append({
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg'],
                'sentiment': sentiment
            })
        
        return pd.DataFrame(results)


def latest_tweets_json(raw_dir: str) -> Optional[str]:
    """Find the most recent tweets JSON file"""
    candidates = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.startswith("tweets_raw_")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def flatten_tweets(json_path: str) -> pd.DataFrame:
    """Extract tweets from JSON file into DataFrame"""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    rows = []
    tweets = payload.get("tweets", {})
    for symbol, items in tweets.items():
        for t in items:
            rows.append({
                "symbol": symbol,
                "id": str(t.get("id")),
                "created_at": t.get("created_at"),
                "text": t.get("text")
            })
    return pd.DataFrame(rows)


def score_tweets_with_vader(output_csv: str = "vader_sentiment_scored.csv") -> str:
    """
    Score tweets using VADER sentiment analysis
    
    Args:
        output_csv: Output filename for scored tweets
        
    Returns:
        Path to the output file
    """
    analyzer = VADERSentimentAnalyzer()
    raw_dir = DATA_CONFIG['raw_data_dir']
    processed_dir = DATA_CONFIG['processed_data_dir']
    os.makedirs(processed_dir, exist_ok=True)

    # Try to load real tweets first
    tweets_json = latest_tweets_json(raw_dir)
    if tweets_json is None:
        # Fallback: use Sentiment140 sample if no collected tweets available
        ds_path = os.path.join('data', 'processed', 'sentiment140_train.csv')
        if not os.path.isfile(ds_path):
            raise FileNotFoundError("No tweets_raw_*.json found and no sentiment140_train.csv fallback found")
        
        base = pd.read_csv(ds_path).head(1000).copy()
        base['id'] = base.index.astype(str)
        base['symbol'] = 'AAPL'
        base['created_at'] = pd.Timestamp.utcnow().isoformat()
        df = base[['id', 'symbol', 'created_at', 'text']]
    else:
        df = flatten_tweets(tweets_json)
        if df.empty:
            # Fallback to sentiment140 if file had no tweets
            ds_path = os.path.join('data', 'processed', 'sentiment140_train.csv')
            if not os.path.isfile(ds_path):
                raise ValueError("No tweets found in raw JSON and no sentiment140_train.csv fallback found")
            
            base = pd.read_csv(ds_path).head(1000).copy()
            base['id'] = base.index.astype(str)
            base['symbol'] = 'AAPL'
            base['created_at'] = pd.Timestamp.utcnow().isoformat()
            df = base[['id', 'symbol', 'created_at', 'text']]

    print(f"Analyzing {len(df)} tweets with VADER...")
    
    # Analyze sentiment
    sentiment_results = analyzer.analyze_batch(df["text"].fillna(""))
    
    # Combine original data with sentiment results
    output_df = pd.concat([
        df[["id", "symbol", "created_at", "text"]].reset_index(drop=True),
        sentiment_results.reset_index(drop=True)
    ], axis=1)

    # Save results
    output_path = os.path.join(processed_dir, output_csv)
    output_df.to_csv(output_path, index=False)
    
    print(f"VADER sentiment analysis complete!")
    print(f"Results saved to: {output_path}")
    print(f"Sentiment distribution:")
    print(sentiment_results['sentiment'].value_counts())
    
    return output_path


def compare_vader_vs_naive_bayes() -> Dict[str, Any]:
    """
    Compare VADER vs Naive Bayes performance on the same dataset
    """
    analyzer = VADERSentimentAnalyzer()
    
    # Load sentiment140 training data for comparison
    ds_path = os.path.join('data', 'processed', 'sentiment140_train.csv')
    if not os.path.isfile(ds_path):
        raise FileNotFoundError("sentiment140_train.csv not found for comparison")
    
    df = pd.read_csv(ds_path).head(1000)
    
    # VADER predictions
    vader_results = analyzer.analyze_batch(df['text'])
    vader_preds = vader_results['sentiment'].tolist()
    
    # Labels are already in text format
    true_labels = df['label'].tolist()
    
    # Calculate accuracy
    vader_accuracy = sum(1 for pred, true in zip(vader_preds, true_labels) if pred == true) / len(true_labels)
    
    comparison = {
        'vader_accuracy': vader_accuracy,
        'naive_bayes_accuracy': 0.7956,  # From existing metrics
        'improvement': vader_accuracy - 0.7956,
        'vader_distribution': pd.Series(vader_preds).value_counts().to_dict(),
        'true_distribution': pd.Series(true_labels).value_counts().to_dict()
    }
    
    print("VADER vs Naive Bayes Comparison:")
    print(f"VADER Accuracy: {vader_accuracy:.4f}")
    print(f"Naive Bayes Accuracy: {comparison['naive_bayes_accuracy']:.4f}")
    print(f"Improvement: {comparison['improvement']:+.4f}")
    
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score tweets with VADER sentiment analysis")
    parser.add_argument("--output_csv", default="vader_sentiment_scored.csv", 
                       help="Output CSV filename")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare VADER vs Naive Bayes performance")
    args = parser.parse_args()

    if args.compare:
        compare_vader_vs_naive_bayes()
    else:
        score_tweets_with_vader(args.output_csv)
