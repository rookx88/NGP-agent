import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import tweepy
import time
import logging
from dotenv import load_dotenv
import os
from database.base import SessionLocal, Base
from ai.personality import VintageRadioHost
from twitter.client import send_tweet
from database.models import Interaction
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from twitter.utils import check_rate_limits
from .core import get_client
from .auth import get_bot_user_id
from ai.news_processor import NewsDigester
from openai import OpenAI
from datetime import datetime, timedelta, timezone
from .utils import rate_limit_retry
from .client import get_twitter_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)  

# Initialize Sentry after loading .env
sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    integrations=[SqlalchemyIntegration()],
    traces_sample_rate=1.0,
    environment="production" if os.getenv('ENV') == "prod" else "development",
    shutdown_timeout=5  # Seconds to flush events
)

logger.info(f"Loaded BOT_HANDLE from .env: {os.getenv('BOT_HANDLE')}")

def check_mentions(client, callback, last_checked_id=None):
    """Poll for new mentions"""
    try:
        # Get bot's user ID
        bot_username = os.getenv('BOT_HANDLE').replace('@', '')
        logger.info(f"Checking mentions for bot: @{bot_username}")
        bot_user = client.get_user(username=bot_username)
        logger.info(f"Found bot user ID: {bot_user.data.id}")
        
        # Get mentions
        mentions = client.get_users_mentions(
            bot_user.data.id,
            since_id=last_checked_id,
            tweet_fields=['created_at', 'author_id'],
            user_fields=['username'],
            expansions=['author_id']
        )
        
        if not mentions.data:
            return last_checked_id
            
        # Process mentions
        for tweet in mentions.data:
            author = next((user for user in mentions.includes['users'] 
                         if user.id == tweet.author_id), None)
            if author:
                logger.info(f"New mention from @{author.username}: {tweet.text}")
                # Create a simple object with required attributes
                class TweetData:
                    def __init__(self, tweet, author):
                        self.id = tweet.id
                        self.text = tweet.text
                        self.author = author
                        self.created_at = tweet.created_at
                
                tweet_data = TweetData(tweet, author)
                callback(tweet_data)  # Process the tweet
        
        # Return the newest tweet id
        return mentions.data[0].id

    except Exception as e:
        logger.error(f"Error checking mentions: {e}")
        return last_checked_id

def start_mention_polling(dry_run=False):
    client = get_twitter_client().client  # Access underlying tweepy client
    bot_user_id = get_bot_user_id(client)
    last_id = None
    poll_count = 0
    
    while True:
        try:
            # Add jitter to avoid rate limits
            sleep_time = 30 + (poll_count % 10) * 5
            time.sleep(sleep_time)
            poll_count += 1
            
            # Get mentions with max results and pagination
            mentions_response = client.get_users_mentions(
                id=bot_user_id,
                since_id=last_id,
                max_results=50,
                tweet_fields=['created_at', 'author_id'],
                expansions=['author_id'],
                pagination_token=None
            )
            
            tweets = mentions_response.get('data', [])
            users = {u['id']: u for u in mentions_response.get('includes', {}).get('users', [])}

            if not tweets:
                continue

            for tweet in reversed(tweets):
                author = users.get(tweet.get('author_id'), {})
                logger.info(f"[{'DRY RUN' if dry_run else 'LIVE'}] Found mention from @{author.get('username', 'unknown')}")
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would process: {tweet.get('text')}")
                else:
                    handle_mention(tweet, users)
                    
            # Update last_id AFTER processing all tweets
            if tweets:
                last_id = str(max(int(t["id"]) for t in tweets))
                
        except tweepy.TooManyRequests:
            logger.warning("Rate limited - sleeping 15 minutes")
            time.sleep(900)

def handle_mention(tweet, users):
    """Process mention and generate response"""
    try:
        with SessionLocal() as db:
            # Check for duplicates first
            existing = db.query(Interaction).filter_by(tweet_id=tweet.get('id')).first()
            if existing:
                logger.warning(f"Skipping duplicate tweet {tweet.get('id')}")
                return

            # Generate response BEFORE creating interaction
            host = VintageRadioHost()
            original_text = tweet.get('text')  # Dict-style access
            
            # And users lookups:
            user_handle = users.get(tweet.get('author_id'), {}).get('username')
            
            # Also check this line:
            if len(original_text) > 1000:
                logger.warning(f"Long original tweet ({len(original_text)} chars)")
            
            # Update in VintageRadioHost call:
            response = host.create_full_reply(
                tweet_text=original_text,
                user_handle=user_handle,
                context="Historical context needed"
            )

            # Truncate response before saving
            MAX_RESPONSE_LENGTH = 2000
            response = response[:MAX_RESPONSE_LENGTH].strip() + "…"
            
            # Validate lengths
            if len(response) > 4000:
                raise ValueError("Response exceeds 4000 character limit")
            
            # Now create interaction with the response
            interaction = Interaction(
                tweet_id=tweet.get('id'),
                user_handle=user_handle,
                original_content=original_text,
                bot_response=response,
                timestamp=tweet.get('created_at')
            )
            
            db.add(interaction)
            db.commit()
            
            # Replace the problematic debug log with:
            logger.debug(
                f"Database entry created: {{\n"
                f"    'tweet_id': '{interaction.tweet_id}',\n"
                f"    'user': '@{interaction.user_handle}',\n"
                f"    'bot_response': '{interaction.bot_response[:50]}...',\n"
                f"    'timestamp': '{interaction.timestamp}'\n"
                f"}}"
            )
            
            logger.debug(f"Processed response: {response}")
            
            send_tweet(
                text=response,
                in_reply_to_tweet_id=tweet.get('id')
            )
            logger.info(f"Sent response: {response}")
            time.sleep(300)  # 5-minute cooldown
        
        logger.info(f"Processed mention from @{users.get(tweet.get('author_id'), {}).get('username')}")
        
    except Exception as e:
        logger.critical(f"Critical error handling mention: {e}")
        sentry_sdk.capture_exception(e)

@rate_limit_retry(max_retries=3)
def get_mentions(client, user_id, last_id):
    return client.get_users_mentions(
        id=user_id,
        since_id=last_id,
        max_results=50,
        tweet_fields=['created_at', 'author_id'],
        expansions=['author_id']
    )

if __name__ == "__main__":
    logger.info("Starting AI Twitter Agent in LIVE MODE...")
    start_mention_polling(dry_run=False) 