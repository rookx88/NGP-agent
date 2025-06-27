import tweepy
import logging
from .client import get_twitter_client
from ai.personality import VintageRadioHost
import requests
from .utils import count_content_length

logger = logging.getLogger(__name__)

class BroadcastScheduler:
    def __init__(self):
        self.twitter_client = get_twitter_client()
        self.host = VintageRadioHost(twitter_client=self.twitter_client)
        
    def post_daily_broadcast(self, dry_run: bool = False) -> dict:
        """Execute full flow without actual posting when dry_run=True"""
        content = self.host.generate_daily_broadcast()
        
        if dry_run:
            logger.info("DRY RUN: Would post:")
            logger.info(f"Text: {content['text']}")
            logger.info(f"Image: {content['image_url']}")
            logger.info(f"Alt Text: {content['alt_text']}")
            return {
                "dry_run": True,
                "text_length": count_content_length(content['text']),
                "image_valid": bool(content['image_url']),
                "components": list(content.keys())
            }
        
        # Proceed with real posting
        media_id = self._upload_media(content['image_url'])
        return self._post_tweet(content['text'], media_id)

    def _upload_media(self, image_url):
        if image_url:
            # Download image with a custom User-Agent to comply with
            # Wikimedia's policy. The default python-requests agent is
            # often blocked with a 403 error.
            headers = {"User-Agent": "HistoricalRadioBot/1.0"}
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Upload to Twitter
            media = self.twitter_client.v1.media_upload(
                filename="historical_image.jpg",
                file=response.content
            )
            return media.media_id
        return None

    def _post_tweet(self, tweet_text, media_id):
        """Use Twitter Blue API endpoint"""
        try:
            return self.twitter_client.create_tweet(
                text=tweet_text,
                media_ids=[media_id] if media_id else None,
                user_auth=True,  # Premium account flag
                format='detailed'  # Allow extended content
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Image download failed: {e}, posting text-only")
            return self.twitter_client.create_tweet(text=tweet_text)
            
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            return {"error": str(e)} 

if __name__ == "__main__":
    scheduler = BroadcastScheduler()
    result = scheduler.post_daily_broadcast()
    print("Post result:", result)
