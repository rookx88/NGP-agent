import pytest
from unittest.mock import Mock, patch
import sys
sys.path.append(".")  # Add project root to Python path
from twitter.poster import BroadcastScheduler  # Direct import

@pytest.fixture
def mock_scheduler():
    # Create a test scheduler with mocked dependencies
    scheduler = BroadcastScheduler()
    
    # Mock the content generation to return consistent test data
    scheduler.host.generate_daily_broadcast = Mock(return_value={
        'text': 'Test broadcast 📻',
        'image_url': 'https://commons.wikimedia.org/valid.jpg',
        'alt_text': 'Test image',
        'sound_effect': 'vinyl scratch'
    })
    
    # Mock the Twitter API response
    scheduler.twitter_client.create_tweet = Mock(return_value={'id': '123'})
    
    return scheduler

def test_successful_post(mock_scheduler):
    # Mock the media upload to return a fake media ID
    mock_scheduler._upload_media = Mock(return_value='media_123')
    
    # Execute the main posting flow
    result = mock_scheduler.post_daily_broadcast()
    
    # Verify the expected successful outcome
    assert 'id' in result
    mock_scheduler.twitter_client.create_tweet.assert_called_with(
        text='Test broadcast 📻',
        media_ids=['media_123'],
        user_auth=True,
        format='detailed'
    )

def test_historical_image_fallback(mock_scheduler):
    # Mock failed image search and valid fallback
    mock_scheduler.host.generate_daily_broadcast = Mock(return_value={
        'text': 'Test broadcast 📻',
        'image_url': 'https://example.com/valid-fallback.jpg',
        'alt_text': 'Fallback image',
        'sound_effect': 'vinyl scratch'
    })
    
    # Mock media upload response
    mock_scheduler._upload_media = Mock(return_value='fallback_media')
    
    result = mock_scheduler.post_daily_broadcast()
    
    # Check media ID is in the API call args
    mock_scheduler.twitter_client.create_tweet.assert_called_with(
        text='Test broadcast 📻',
        media_ids=['fallback_media'],
        user_auth=True,
        format='detailed'
    )

def test_image_failure(mock_scheduler):
    # Simulate image upload failure
    mock_scheduler._upload_media = Mock(return_value=None)
    
    # Execute the flow
    result = mock_scheduler.post_daily_broadcast()
    
    # Verify fallback behavior
    assert 'id' in result
    mock_scheduler.twitter_client.create_tweet.assert_called_with(
        text='Test broadcast 📻',
        media_ids=None,
        user_auth=True,
        format='detailed'
    )

def test_character_limit(mock_scheduler):
    # Create text that's 100 characters over limit
    long_text = 'Historical analysis shows...' * 50  # 1600 chars
    mock_scheduler.host.generate_daily_broadcast.return_value = {
        'text': long_text,
        'image_url': None,
        'alt_text': 'Test image',
        'sound_effect': 'static'
    }
    
    mock_scheduler.post_daily_broadcast()
    
    # Get the actual text sent to Twitter
    args, kwargs = mock_scheduler.twitter_client.create_tweet.call_args
    sent_text = kwargs['text']
    
    # Only verify length constraint
    assert len(sent_text) <= 1500, f"Text length {len(sent_text)} exceeds limit"
    
    # Optional: Verify original content preserved as much as possible
    assert sent_text.startswith('Historical analysis shows'), "Truncation corrupted content"
