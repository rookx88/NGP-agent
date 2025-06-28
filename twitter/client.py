import tweepy
import os
from dotenv import load_dotenv
import logging
import time
from tweepy import Client
from time import sleep
import requests

logger = logging.getLogger(__name__)

load_dotenv()  

class TwitterStream(tweepy.StreamingClient):
    def __init__(self, bearer_token, callback):
        super().__init__(bearer_token)
        self.callback = callback

    def on_tweet(self, tweet):
        if tweet.text.startswith(f"@{os.getenv('BOT_HANDLE')}"):
            self.callback(tweet)

def create_api():
    auth = tweepy.OAuthHandler(
        os.getenv('TWITTER_CONSUMER_KEY'),
        os.getenv('TWITTER_CONSUMER_SECRET')
    )
    auth.set_access_token(
        os.getenv('TWITTER_ACCESS_TOKEN'),
        os.getenv('TWITTER_ACCESS_SECRET')
    )
    return tweepy.API(auth, wait_on_rate_limit=True)

def send_tweet(text, in_reply_to_tweet_id=None):
    try:
        client = get_client()
        return client.create_tweet(
            text=text,
            in_reply_to_tweet_id=in_reply_to_tweet_id
        )
    except tweepy.TooManyRequests as e:
        sleep_time = 60 * 15  # 15 minutes
        logger.warning(f"Rate limited. Sleeping {sleep_time//60} minutes")
        sleep(sleep_time)
        return send_tweet(text, in_reply_to_tweet_id)  # Retry
    except tweepy.TweepyException as e:
        logger.error(f"Twitter error: {e}")
        raise

def start_stream(callback):
    stream = TwitterStream(
        bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
        callback=callback
    )
    return stream 

class TwitterAPIClient:
    """Wrapper providing both v1 and v2 Twitter functionality."""

    def __init__(self):
        load_dotenv()

        # v1.1 API client for media uploads
        auth = tweepy.OAuth1UserHandler(
            os.getenv('TWITTER_CONSUMER_KEY'),
            os.getenv('TWITTER_CONSUMER_SECRET'),
            os.getenv('TWITTER_ACCESS_TOKEN'),
            os.getenv('TWITTER_ACCESS_SECRET'),
        )
        self.v1 = tweepy.API(auth, wait_on_rate_limit=True)

        # v2 client for posting tweets and other endpoints
        self.client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_CONSUMER_KEY'),
            consumer_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_SECRET'),
            wait_on_rate_limit=True,
        )

        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('TWITTER_BEARER_TOKEN')}",
            "User-Agent": "HistoricalRadioBot/1.0"
        }

    def create_tweet(self, *args, **kwargs):
        """Proxy to Tweepy Client.create_tweet"""
        return self.client.create_tweet(*args, **kwargs)
    
    def get_trends(self, woeid: int) -> dict:
        """Get trends using official API v2 endpoint"""
        response = requests.get(
            f"{self.base_url}/trends/by/woeid/{woeid}",
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

def get_twitter_client():
    return TwitterAPIClient()

def get_twitter_client_v2():
    """Create Twitter API v2 authenticated client"""
    return Client(
        bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
        consumer_key=os.getenv('TWITTER_CONSUMER_KEY'),
        consumer_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
        access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        access_token_secret=os.getenv('TWITTER_ACCESS_SECRET'),
        wait_on_rate_limit=True
    ) 