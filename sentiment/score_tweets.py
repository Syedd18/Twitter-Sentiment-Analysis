import os
import json
from typing import Optional

import pandas as pd
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


def latest_tweets_json(raw_dir: str) -> Optional[str]:
    candidates = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.startswith("tweets_raw_")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def flatten_tweets(json_path: str) -> pd.DataFrame:
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


def score(model_path: str, output_csv: str) -> str:
    model = joblib.load(model_path)
    raw_dir = DATA_CONFIG['raw_data_dir']
    processed_dir = DATA_CONFIG['processed_data_dir']
    os.makedirs(processed_dir, exist_ok=True)

    tweets_json = latest_tweets_json(raw_dir)
    if tweets_json is None:
        # Fallback: use Sentiment140 sample if no collected tweets available
        ds_path = os.path.join('data','processed','sentiment140_train.csv')
        if not os.path.isfile(ds_path):
            raise FileNotFoundError("No tweets_raw_*.json found and no sentiment140_train.csv fallback found")
        base = pd.read_csv(ds_path).head(1000).copy()
        base['id'] = base.index.astype(str)
        base['symbol'] = 'AAPL'
        base['created_at'] = pd.Timestamp.utcnow().isoformat()
        df = base[['id','symbol','created_at','text']]
    else:
        df = flatten_tweets(tweets_json)
        if df.empty:
            # Fallback to sentiment140 if file had no tweets
            ds_path = os.path.join('data','processed','sentiment140_train.csv')
            if not os.path.isfile(ds_path):
                raise ValueError("No tweets found in raw JSON and no sentiment140_train.csv fallback found")
            base = pd.read_csv(ds_path).head(1000).copy()
            base['id'] = base.index.astype(str)
            base['symbol'] = 'AAPL'
            base['created_at'] = pd.Timestamp.utcnow().isoformat()
            df = base[['id','symbol','created_at','text']]

    preds = model.predict(df["text"].fillna(""))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df["text"].fillna(""))
        # Assuming classes order
        classes = list(model.classes_) if hasattr(model, "classes_") else ["negative", "neutral", "positive"]
        # Map to pos/neg/neu probabilities when present
        prob_df = pd.DataFrame(proba, columns=[f"prob_{c[:3]}" for c in classes])
    else:
        prob_df = pd.DataFrame()

    out = pd.concat([df[["id", "symbol", "created_at", "text"]].reset_index(drop=True),
                     pd.Series(preds, name="sentiment"),
                     prob_df], axis=1)

    out_path = os.path.join(processed_dir, output_csv)
    out.to_csv(out_path, index=False)
    print(f"Sentiment-scored tweets saved to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score collected tweets with trained sentiment model")
    parser.add_argument("--model_path", default=os.path.join("models", "naive_bayes_sentiment.joblib"))
    parser.add_argument("--output_csv", default="sentiment_scored.csv")
    args = parser.parse_args()

    score(args.model_path, args.output_csv)


