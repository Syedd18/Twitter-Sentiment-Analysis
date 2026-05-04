import os
import json
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG


def _latest_file_with_prefix(directory: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    candidates = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def create_spark(app_name: str = "StockSentimentPreprocessing") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
    return spark


def load_raw_data(spark: SparkSession,
                  raw_dir: str) -> tuple:
    tweets_path = _latest_file_with_prefix(raw_dir, "tweets_raw_")
    stocks_path = _latest_file_with_prefix(raw_dir, "stock_data_")

    if tweets_path is None:
        raise FileNotFoundError("No tweets_raw_*.json found in raw data directory")
    if stocks_path is None:
        raise FileNotFoundError("No stock_data_*.csv found in raw data directory")

    # Load tweets via Python for robust nested parsing, then create Spark DF
    with open(tweets_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    tweets_map = payload.get("tweets", {})
    for symbol, items in tweets_map.items():
        for item in items:
            rows.append({
                "symbol": symbol,
                "id": str(item.get("id")),
                "text": item.get("text"),
                "created_at": item.get("created_at"),
                "lang": item.get("lang"),
                "like_count": item.get("like_count"),
                "retweet_count": item.get("retweet_count"),
                "reply_count": item.get("reply_count"),
                "quote_count": item.get("quote_count"),
            })
    schema = T.StructType([
        T.StructField("symbol", T.StringType(), True),
        T.StructField("id", T.StringType(), True),
        T.StructField("text", T.StringType(), True),
        T.StructField("created_at", T.StringType(), True),
        T.StructField("lang", T.StringType(), True),
        T.StructField("like_count", T.LongType(), True),
        T.StructField("retweet_count", T.LongType(), True),
        T.StructField("reply_count", T.LongType(), True),
        T.StructField("quote_count", T.LongType(), True),
    ])
    tweets_df = spark.createDataFrame(rows, schema=schema)

    stocks_raw = spark.read.option("header", True).csv(stocks_path)
    # If CSV read yielded no columns (empty file), create empty DF with expected schema
    if len(stocks_raw.columns) == 0:
        stock_schema = T.StructType([
            T.StructField("date", T.StringType(), True),
            T.StructField("symbol", T.StringType(), True),
            T.StructField("open", T.DoubleType(), True),
            T.StructField("high", T.DoubleType(), True),
            T.StructField("low", T.DoubleType(), True),
            T.StructField("close", T.DoubleType(), True),
            T.StructField("volume", T.DoubleType(), True),
        ])
        stocks_raw = spark.createDataFrame([], schema=stock_schema)

    return tweets_df, stocks_raw


@F.udf(T.StringType())
def clean_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    import re
    # Remove URLs, mentions, hashtags (keep word), emojis and non-alphanumerics except spaces and $
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", lambda m: m.group(0)[1:], text)
    text = re.sub(r"[^A-Za-z0-9\s\$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def preprocess_tweets(spark: SparkSession, tweets_raw_df):
    # We receive an already-flattened DataFrame from load_raw_data
    tweets = tweets_raw_df.withColumn(
        "created_at",
        F.to_timestamp(F.col("created_at"))
    )
    tweets = tweets.withColumn("clean_text", clean_text(F.col("text")))
    tweets = tweets.filter((F.col("clean_text").isNotNull()) & (F.length("clean_text") > 0))
    return tweets


def aggregate_features(spark: SparkSession, tweets_df, stocks_df):
    # Load sentiment-scored tweets if present
    processed_dir = DATA_CONFIG['processed_data_dir']
    sentiment_scored_path = os.path.join(processed_dir, "sentiment_scored.csv")
    if os.path.isfile(sentiment_scored_path):
        scored = spark.read.option("header", True).csv(sentiment_scored_path)
        scored = scored.withColumn("id", F.col("id").cast("string"))
        tweets_df = tweets_df.withColumn("id", F.col("id").cast("string"))
        tweets_df = tweets_df.join(scored.select("id", "sentiment", "prob_pos", "prob_neg", "prob_neu"),
                                   on="id", how="left")
    else:
        tweets_df = tweets_df.withColumn("sentiment", F.lit(None).cast("string")) 
        tweets_df = tweets_df.withColumn("prob_pos", F.lit(None).cast("double"))
        tweets_df = tweets_df.withColumn("prob_neg", F.lit(None).cast("double"))
        tweets_df = tweets_df.withColumn("prob_neu", F.lit(None).cast("double"))

    # 5-minute window aggregation per symbol
    tweets_df = tweets_df.withColumn("window_start", F.window("created_at", "5 minutes").getField("start"))

    agg = tweets_df.groupBy("symbol", "window_start").agg(
        F.count("id").alias("tweet_volume"),
        F.avg(F.when(F.col("sentiment") == "positive", 1).otherwise(0)).alias("pos_rate"),
        F.avg(F.when(F.col("sentiment") == "negative", 1).otherwise(0)).alias("neg_rate"),
        F.avg(
            F.when(F.col("sentiment") == "positive", 1)
             .when(F.col("sentiment") == "negative", -1)
             .otherwise(0)
        ).alias("avg_sentiment")
    )

    # Prepare stock data (daily) and forward-fill to window_start timestamps by join on date
    stocks_df = stocks_df.withColumn("date", F.to_date("date"))
    agg = agg.withColumn("date", F.to_date("window_start"))

    joined = agg.join(
        stocks_df.select(
            F.col("symbol").alias("stock_symbol"),
            "date", "open", "close", "high", "low", "volume"
        ),
        (agg.symbol == F.col("stock_symbol")) & (agg.date == F.col("date")),
        how="left"
    ).drop("stock_symbol")

    result = joined.select(
        F.col("window_start").alias("timestamp"),
        "symbol",
        "avg_sentiment", "pos_rate", "neg_rate", "tweet_volume",
        "open", "close", "high", "low", "volume"
    )
    return result


def main():
    raw_dir = DATA_CONFIG['raw_data_dir']
    processed_dir = DATA_CONFIG['processed_data_dir']
    os.makedirs(processed_dir, exist_ok=True)

    spark = create_spark()

    tweets_raw, stocks_raw = load_raw_data(spark, raw_dir)
    tweets_df = preprocess_tweets(spark, tweets_raw)
    # Fallback: if no tweets available, use sentiment_scored.csv sample
    if tweets_df.rdd.isEmpty():
        fallback_path = os.path.join(processed_dir, "sentiment_scored.csv")
        if os.path.isfile(fallback_path):
            scored = spark.read.option("header", True).csv(fallback_path)
            tweets_df = scored.select(
                F.col("symbol"),
                F.col("id").cast("string").alias("id"),
                F.col("text"),
                F.to_timestamp(F.col("created_at")).alias("created_at"),
            ).withColumn("clean_text", clean_text(F.col("text")))
        else:
            # Create empty DF with expected schema
            tweets_schema = T.StructType([
                T.StructField("symbol", T.StringType(), True),
                T.StructField("id", T.StringType(), True),
                T.StructField("text", T.StringType(), True),
                T.StructField("created_at", T.TimestampType(), True),
                T.StructField("clean_text", T.StringType(), True),
            ])
            tweets_df = spark.createDataFrame([], schema=tweets_schema)

    # Cast stock columns to proper types
    if len(stocks_raw.columns) > 0:
        stocks_df = (
            stocks_raw
            .withColumn("open", F.col("open").cast("double"))
            .withColumn("close", F.col("close").cast("double"))
            .withColumn("high", F.col("high").cast("double"))
            .withColumn("low", F.col("low").cast("double"))
            .withColumn("volume", F.col("volume").cast("double"))
        )
    else:
        stocks_df = stocks_raw

    features_df = aggregate_features(spark, tweets_df, stocks_df)

    out_path = os.path.join(processed_dir, "processed_features.parquet")
    features_df.write.mode("overwrite").parquet(out_path)

    print(f"Processed features saved to {out_path}")


if __name__ == "__main__":
    main()


