# New file for shared Twitter utilities
import tweepy
import os
from dotenv import load_dotenv
from datetime import  timedelta
import logging
import time
from .core import get_client
from .auth import get_bot_user_id
from functools import wraps
import re

logger = logging.getLogger(__name__)

load_dotenv()

RATE_LIMIT_WINDOW = timedelta(minutes=15)

def get_twitter_client():
    """Shared client creation for all Twitter functions"""
    return tweepy.Client(
        consumer_key=os.getenv('TWITTER_CONSUMER_KEY'),
        consumer_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
        access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        access_token_secret=os.getenv('TWITTER_ACCESS_SECRET')
    )

def check_rate_limits(client):
    """Check rate limits with guaranteed response handling"""
    try:
        # Make API call that always returns headers
        try:
            # Get application rate limits (works with both auth types)
            client.get_application_rate_limit_status()
            headers = client.last_response.headers
        except AttributeError:
            # Fallback if last_response isn't available
            return {'remaining': 150, 'reset_in': 900}  # Conservative default

        remaining = int(headers.get('x-rate-limit-remaining', 15))
        reset_time = int(headers.get('x-rate-limit-reset', time.time() + 900))
        
        return {
            'remaining': remaining,
            'reset_in': max(0, reset_time - time.time()),
            'window': 900
        }
    except Exception as e:
        logger.error(f"Rate check failed: {str(e)}", exc_info=True)
        return {'remaining': 0, 'reset_in': 900}

def get_rate_limit_status(client):
    """Get current rate limit status"""
    return client.get_rate_limits()

def rate_limit_retry(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except tweepy.TooManyRequests:
                    logger.warning(f"Rate limited. Retry {retries+1}/{max_retries}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
            raise Exception("Max rate limit retries exceeded")
        return wrapper
    return decorator 

def count_content_length(text: str) -> int:
    """Count actual content length without formatting"""
    # Remove formatting characters and whitespace
    clean_text = ''.join(text.split('\n'))
    # Remove emoji and special characters
    clean_text = re.sub(r'[^\w\s.,!?-]', '', clean_text)
    return len(clean_text.strip()) 