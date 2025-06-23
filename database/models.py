from sqlalchemy import Column, Integer, String, DateTime, Index, Text
from database.base import Base

class Interaction(Base):
    __tablename__ = 'interactions'
    
    id = Column(Integer, primary_key=True)
    tweet_id = Column(String(50), unique=True)
    user_handle = Column(String(100))
    original_content = Column(String(1000))  # Original tweets can be 280 chars
    bot_response = Column(String(4000))     # Match GPT-4's 4k token limit
    timestamp = Column(DateTime)
    # Removed redundant 'response' and 'sentiment' columns

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not all([self.tweet_id, self.original_content, self.bot_response]):
            raise ValueError("Missing required interaction fields")

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    username = Column(String)
    last_interaction = Column(DateTime)
    # Removed 'interaction_count' as it can be queried 

Index('ix_interactions_tweet_id', Interaction.tweet_id, unique=True) 