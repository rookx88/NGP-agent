import time
import json
import logging
import os
from db import get_db, execute_db
from datetime import datetime, timezone, timedelta
from twitter.client import get_twitter_client
from twitter.utils import rate_limit_retry
import math
import re
from collections import deque, defaultdict
from dateutil import parser as date_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest_service")

ACCOUNTS_PATH = "./app/backend/tracked_accounts.json"
KEYWORDS_PATH = "./app/backend/tracked_keywords.json"
POLL_INTERVAL = 120  # seconds
TWEETS_PER_ACCOUNT = 5  # Conservative for paid basic tier
Z_SCORE_WINDOW = 20  # Number of past tweets for z-score
VIRALITY_THRESHOLD = 2.0  # z-score threshold for spike

# In-memory velocity history for z-score (persist to DB for prod)
velocity_history = defaultdict(lambda: deque(maxlen=Z_SCORE_WINDOW))

# Helper: convert Twitter time to datetime

def load_tracked_accounts():
    with open(ACCOUNTS_PATH, "r") as f:
        return json.load(f)

def load_tracked_keywords():
    with open(KEYWORDS_PATH, "r") as f:
        return json.load(f)

def extract_keywords(text):
    # Extract hashtags
    hashtags = re.findall(r"#\w+", text)
    # Extract capitalized words (simple proper noun heuristic)
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", text)
    keywords = set([h.lower() for h in hashtags] + [p.lower() for p in proper_nouns])
    return list(keywords)

def update_keyword_stats(keywords, velocity, now):
    conn = get_db()
    for kw in keywords:
        # Update or insert keyword stats
        conn.execute(
            """
            INSERT INTO keywords (term, frequency, avg_score, last_seen)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                frequency = frequency + 1,
                avg_score = (avg_score * (frequency - 1) + ?) / frequency,
                last_seen = ?
            """,
            (kw, velocity, now, velocity, now)
        )
    conn.commit()
    conn.close()

def log_activity(event_type, details):
    conn = get_db()
    conn.execute(
        "INSERT INTO activity_log (timestamp, event_type, details) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(), event_type, json.dumps(details))
    )
    conn.commit()
    conn.close()

def update_account_stats(account_handle, velocity, is_viral, now):
    conn = get_db()
    # Update avg_velocity, viral_count (last 24h), score
    # For MVP, viral_count = count of tweets flagged viral in last 24h
    if is_viral:
        conn.execute(
            """
            INSERT INTO accounts (handle, last_checked, avg_velocity, viral_count, score)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(handle) DO UPDATE SET
                last_checked = ?,
                avg_velocity = (avg_velocity + ?) / 2,
                viral_count = viral_count + 1,
                score = ?
            """,
            (account_handle, now, velocity, velocity, now, velocity, velocity)
        )
    else:
        conn.execute(
            """
            INSERT INTO accounts (handle, last_checked, avg_velocity, viral_count, score)
            VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(handle) DO UPDATE SET
                last_checked = ?,
                avg_velocity = (avg_velocity + ?) / 2,
                score = ?
            """,
            (account_handle, now, velocity, velocity, now, velocity, velocity)
        )
    conn.commit()
    conn.close()

def store_tweet(tweet, account, velocity, normalized, z, virality, keywords, is_viral):
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO tweets (id, text, author_id, author_handle, created_at, retweets, likes, replies, quotes, followers, score, velocity, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tweet['id'],
                tweet['text'],
                tweet['author_id'],
                account['handle'],
                tweet['created_at'],
                tweet['public_metrics']['retweet_count'],
                tweet['public_metrics']['like_count'],
                tweet['public_metrics']['reply_count'],
                tweet['public_metrics'].get('quote_count', 0),
                tweet.get('followers', 0),
                virality,
                velocity,
                account['category']
            )
        )
        # Optionally update if already exists
        conn.execute(
            """
            UPDATE tweets SET score=?, velocity=? WHERE id=?
            """,
            (virality, velocity, tweet['id'])
        )
        conn.commit()
    finally:
        conn.close()
    # Update keyword stats
    update_keyword_stats(keywords, velocity, datetime.utcnow().isoformat())
    # Update account stats
    update_account_stats(account['handle'], velocity, is_viral, datetime.utcnow().isoformat())
    # Log viral event
    if is_viral:
        log_activity('flagged_viral', {'tweet_id': tweet['id'], 'author': account['handle'], 'score': virality})

def compute_z_score(account_handle, velocity):
    history = velocity_history[account_handle]
    if len(history) < 3:
        history.append(velocity)
        return 0.0
    mean = sum(history) / len(history)
    std = math.sqrt(sum((v - mean) ** 2 for v in history) / len(history))
    z = (velocity - mean) / std if std > 0 else 0.0
    history.append(velocity)
    return z

def fetch_and_store_tweets(client, account):
    # Get user ID and followers with public_metrics
    user = client.client.get_user(username=account['handle'], user_fields=["public_metrics"])
    if not user or not hasattr(user, 'data') or not user.data:
        logger.warning(f"Could not fetch user for @{account['handle']}")
        return
    user_id = user.data.id
    followers = 0
    if hasattr(user.data, 'public_metrics') and user.data.public_metrics:
        followers = user.data.public_metrics.get('followers_count', 0)
    else:
        logger.warning(f"No public_metrics for @{account['handle']}, user object: {user.data}")
    # Fetch recent tweets
    tweets = client.client.get_users_tweets(
        user_id,
        max_results=TWEETS_PER_ACCOUNT,
        tweet_fields=["created_at", "public_metrics", "text"],
    )
    if not tweets or not hasattr(tweets, 'data') or not tweets.data:
        logger.info(f"No tweets for @{account['handle']}")
        return
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    for tweet in tweets.data:
        tweet_dict = tweet.data if hasattr(tweet, 'data') else tweet
        tweet_dict['followers'] = followers
        tweet_dict['author_id'] = user_id  # Ensure author_id is present
        # Calculate velocity and scores
        created_at = date_parser.parse(tweet_dict['created_at']).replace(tzinfo=timezone.utc)
        age_min = max((now - created_at).total_seconds() / 60, 1)
        metrics = tweet_dict['public_metrics']
        velocity = (metrics['retweet_count'] + metrics['like_count'] + metrics['reply_count']) / age_min
        normalized = velocity / math.log10(followers + 10) if followers > 0 else velocity
        z = compute_z_score(account['handle'], velocity)
        virality = normalized * (1 + z if z > VIRALITY_THRESHOLD else 1)
        keywords = extract_keywords(tweet_dict['text'])
        is_viral = z > VIRALITY_THRESHOLD or virality > 10  # 10 is a placeholder threshold
        store_tweet(tweet_dict, account, velocity, normalized, z, virality, keywords, is_viral)
        logger.info(f"Stored tweet {tweet_dict['id']} from @{account['handle']} (velocity={velocity:.2f}, z={z:.2f}, viral={is_viral})")

def main_loop():
    logger.info("Starting Synch Ingestion Service...")
    accounts = load_tracked_accounts()
    keywords = load_tracked_keywords()
    logger.info(f"Loaded {len(accounts)} accounts, {len(keywords)} keywords.")
    client = get_twitter_client()
    account_idx = 0
    while True:
        account = accounts[account_idx % len(accounts)]
        try:
            fetch_and_store_tweets(client, account)
        except Exception as e:
            logger.error(f"Error fetching tweets for @{account['handle']}: {e}")
        account_idx += 1
        logger.info("Sleeping until next poll...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop() 