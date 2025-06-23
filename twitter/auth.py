import os

def get_bot_user_id(client=None):
    """Handle dict response from Tweepy"""
    if not os.getenv('BOT_USER_ID'):
        from .core import get_client
        client = client or get_client()
        bot_username = os.getenv('BOT_HANDLE').replace('@', '')
        user_response = client.get_user(username=bot_username)
        
        # Access dictionary response
        user_data = user_response.get('data', {})
        os.environ['BOT_USER_ID'] = str(user_data.get('id'))
        
    return os.getenv('BOT_USER_ID') 