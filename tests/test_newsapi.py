# Test script
from ai.news_processor import NewsDigester

def test_trade_conflict():
    digester = NewsDigester()
    result = digester.get_event_context("Trade war intensifies with new tariffs")
    print(f"Trade Conflict Result: {result}")
    assert "Smoot-Hawley" in result or "news" in result

def test_fallback():
    digester = NewsDigester()
    result = digester.get_event_context("Random topic")
    print(f"Fallback Result: {result}")
    assert "historical" in result

def test_api_failure():
    digester = NewsDigester()
    digester.api_key = "invalid_key"
    result = digester.get_event_context("trade war")
    assert "Smoot-Hawley" in result, f"Expected Smoot-Hawley, got: {result}"

def test_immigration_context():
    digester = NewsDigester()
    result = digester.get_event_context("Border policy changes")
    assert "1924 Immigration Act" in result, f"Expected 1924 Act, got: {result}"

if __name__ == "__main__":
    test_trade_conflict()
    test_fallback()
    test_api_failure()
    test_immigration_context()
