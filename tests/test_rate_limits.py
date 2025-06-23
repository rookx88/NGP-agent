from datetime import datetime
from unittest.mock import MagicMock
from twitter.utils import check_rate_limits
import time
import tweepy

def test_rate_limit_handling():
    # Test OAuth 1.0a response
    mock_client_oauth = MagicMock()
    mock_response_oauth = MagicMock()
    mock_httpx_response = MagicMock()
    mock_httpx_response.headers = {
        'x-rate-limit-remaining': '4',
        'x-rate-limit-reset': str(int(time.time()) + 300)
    }
    mock_response_oauth.response = mock_httpx_response
    mock_client_oauth.get_me.return_value = mock_response_oauth
    
    limits = check_rate_limits(mock_client_oauth)
    assert limits['remaining'] == 4

    # Test Bearer token response
    mock_client_bearer = MagicMock()
    
    # Create proper mock response for Unauthorized error
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_client_bearer.get_me.side_effect = tweepy.Unauthorized(mock_response)
    
    # Mock search response
    mock_search_response = MagicMock()
    mock_search_response.headers = {
        'x-rate-limit-remaining': '2',
        'x-rate-limit-reset': str(int(time.time()) + 600)
    }
    mock_client_bearer.search_recent_tweets.return_value = mock_search_response
    
    bearer_limits = check_rate_limits(mock_client_bearer)
    assert bearer_limits['remaining'] == 2

def test_response_structures():
    # Test OAuth 1.0a response with nested response
    mock_client = MagicMock()
    mock_nested_response = MagicMock()
    mock_headers = {'x-rate-limit-remaining': '15', 'x-rate-limit-reset': '1234567890'}
    mock_nested_response.response.headers = mock_headers
    mock_client.get_me.return_value = mock_nested_response
    assert check_rate_limits(mock_client)['remaining'] == 15

    # Test Bearer token flat response
    mock_client = MagicMock()
    mock_flat_response = MagicMock()
    mock_flat_response.headers = mock_headers
    
    # Create proper mock response for Unauthorized error
    mock_unauth_response = MagicMock()
    mock_unauth_response.status_code = 401
    mock_client.get_me.side_effect = tweepy.Unauthorized(mock_unauth_response)
    
    mock_client.search_recent_tweets.return_value = mock_flat_response
    assert check_rate_limits(mock_client)['remaining'] == 15 