import os
import json
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib


def load_training_data(csv_path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df["text"].tolist(), df["label"].tolist()


def train_and_save_model(train_csv: str, model_dir: str = "models") -> dict:
    os.makedirs(model_dir, exist_ok=True)

    X, y = load_training_data(train_csv)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2)) ),
        ("nb", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    model_path = os.path.join(model_dir, "naive_bayes_sentiment.joblib")
    joblib.dump(pipeline, model_path)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": report,
        "model_path": model_path
    }
    metrics_path = os.path.join(model_dir, "naive_bayes_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 (weighted): {metrics['f1_weighted']:.4f}")
    return metrics


if __name__ == "__main__":
    # Expect a CSV with columns: text,label
    import argparse
    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment model")
    parser.add_argument("--train_csv", required=True, help="Path to CSV with columns text,label")
    parser.add_argument("--model_dir", default="models", help="Directory to save model and metrics")
    args = parser.parse_args()

    train_and_save_model(args.train_csv, args.model_dir)


