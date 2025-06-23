from twitter.mentions import start_mention_polling, handle_mention
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timedelta
import pytest
import tweepy
import os

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_users_mentions.return_value = MagicMock(
        data=[
            MagicMock(
                id="123456",
                text="@TestBot hello",
                author_id="987654",
                created_at=datetime.utcnow() - timedelta(minutes=5)
            )
        ],
        includes={'users': [MagicMock(id="987654", username="testuser")]}
    )
    return client

@pytest.fixture
def mock_db_session():
    from database.base import SessionLocal
    session = MagicMock()
    SessionLocal.return_value = session
    return session

def test_mention_processing(mock_client, mock_db_session, caplog):
    with patch('twitter.mentions.get_client', return_value=mock_client):
        with patch('twitter.mentions.time.sleep') as mock_sleep:
            # Run polling for one iteration
            start_mention_polling()
            
            # Verify API call
            mock_client.get_users_mentions.assert_called_once_with(
                id=os.getenv('BOT_USER_ID'),
                since_id=None,
                tweet_fields=['created_at', 'author_id'],
                expansions=['author_id']
            )
            
            # Verify database interaction
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            
            # Verify logging
            assert "New mention from @testuser" in caplog.text
            assert "Processed mention from @testuser" in caplog.text
            
            # Verify rate limit handling
            assert mock_sleep.call_count == 1
            assert mock_sleep.call_args[0][0] == 60

def test_mention_handling():
    # Setup mock tweet with all required attributes
    mock_tweet = MagicMock()
    mock_tweet.text = "Test mention"
    mock_tweet.id = 12345
    mock_tweet.author.username = "test_user"
    mock_tweet.created_at = datetime.now()
    
    with patch('twitter.mentions.generate_response') as mock_ai, \
         patch('twitter.mentions.send_tweet') as mock_send, \
         patch('twitter.mentions.SessionLocal') as mock_db, \
         patch('time.sleep') as mock_sleep:
        
        # Configure mock AI response
        mock_ai.return_value = "Test response"
        
        # Create a mock database session
        mock_session = MagicMock()
        
        # Proper context manager setup
        mock_db.return_value.__enter__.return_value = mock_session
        mock_db.return_value.__exit__.return_value = None
        
        # Execute
        handle_mention(mock_tweet)
        
        # Verify database interaction
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        saved_interaction = mock_session.add.call_args[0][0]
        assert saved_interaction.tweet_id == 12345
        assert saved_interaction.user_handle == "test_user"
        assert saved_interaction.content == "Test mention"
        
        # Verify AI generation
        mock_ai.assert_called_once_with("Test mention", "test_user")
        
        # Verify tweet sending
        mock_send.assert_called_once_with("@test_user Test response")
        
        # Verify delay between actions
        mock_sleep.assert_called_once_with(30)

def test_empty_ai_response():
    mock_tweet = MagicMock()
    mock_tweet.text = "Empty test"
    mock_tweet.author.username = "test_user"
    
    with patch('twitter.mentions.generate_response') as mock_ai, \
         patch('twitter.mentions.send_tweet') as mock_send:
        
        mock_ai.return_value = None  # Simulate failed response
        
        handle_mention(mock_tweet)
        
        mock_send.assert_not_called()  # Shouldn't send empty response