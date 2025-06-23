import pytest
from unittest.mock import Mock, patch
from ai.news_processor import NewsDigester
import os

def test_semantic_detection():
    digester = NewsDigester()
    
    # Test semantic matching
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'articles': [{'title': 'Test'}]}
        
        # Known analogy match
        result = digester.get_event_context("International trade agreements")
        assert "Smoot-Hawley" in result
        
        # Semantic similarity test
        result = digester.get_event_context("Cross-border commerce policies")
        assert "Smoot-Hawley" in result

def test_llm_fallback():
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
         patch('numpy.dot', return_value=0.2), \
         patch('ai.news_processor.OpenAI') as mock_openai:

        # Setup mock client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_message = Mock(content="immigration")
        mock_choice = Mock(message=mock_message)
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        digester = NewsDigester()
        result = digester.get_event_context("Border security measures")

        assert "1924 Immigration Act" in result
        mock_openai.assert_called_once_with(api_key='test-key')
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify this text into 'trade', 'immigration' or 'technology':"},
                {"role": "user", "content": "Border security measures"}
            ],
            temperature=0.0
        )

def test_response_formatting():
    digester = NewsDigester()
    
    # Test with mock news results
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            'articles': [{'title': 'AI Regulation News'}, {'title': 'Tech Policy Update'}]
        }
        
        result = digester.get_event_context("Technology regulations")
        assert " | " in result
        assert "Historical context" in result
        assert "Current developments" in result

def test_error_handling():
    digester = NewsDigester()
    
    # Test API failure
    with patch('requests.get', side_effect=Exception("API down")):
        result = digester.get_event_context("Test")
        assert "historical records" in result 