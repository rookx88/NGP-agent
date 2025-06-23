import os
import requests
import numpy as np
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Tuple  # For return type hints

logger = logging.getLogger(__name__)
load_dotenv()

class NewsDigester:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.historical_analogues = {
            'trade': '1930 Smoot-Hawley Tariff Act',
            'immigration': '1924 Immigration Act',
            'technology': '1938 Radio Act',
            'default': 'historical records'
        }
        
        # Precompute topic embeddings
        self.topic_embeddings = {
            topic: self.model.encode(analogy)
            for topic, analogy in self.historical_analogues.items()
        }
        
        # Confidence threshold for LLM fallback
        self.similarity_threshold = 0.3

        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def _get_analogy(self, topic: str) -> str:
        return self.historical_analogues.get(topic.lower(), self.historical_analogues['default'])

    def _semantic_topic_detection(self, text: str) -> tuple:
        """Return (topic, confidence_score) using embeddings"""
        text_embed = self.model.encode(text)
        similarities = {
            topic: np.dot(text_embed, topic_embed)
            for topic, topic_embed in self.topic_embeddings.items()
            if topic != 'default'
        }
        max_topic = max(similarities, key=similarities.get)
        return max_topic, similarities[max_topic]

    def _llm_topic_detection(self, text: str) -> str:
        """Fallback topic detection using GPT-3.5"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "Classify this text into 'trade', 'immigration' or 'technology':"
                }, {
                    "role": "user",
                    "content": text
                }],
                temperature=0.0
            )
            return response.choices[0].message.content.lower()
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return 'default'

    def get_event_context(self, tweet_text: str) -> str:
        """Hybrid approach to get news context"""
        try:
            # First stage: Semantic detection
            detected_topic, confidence = self._semantic_topic_detection(tweet_text)
            
            # Second stage: LLM fallback for low confidence
            if confidence < self.similarity_threshold:
                detected_topic = self._llm_topic_detection(tweet_text)
                logger.info(f"Used LLM fallback for topic: {detected_topic}")

            # News API call
            response = self.session.get(
                "https://newsapi.org/v2/everything",
                params={
                    'q': detected_topic,
                    'apiKey': self.api_key,
                    'pageSize': 2,
                    'sortBy': 'relevancy'
                },
                timeout=10
            )
            
            articles = response.json().get('articles', [])
            historical = f"Historical context: {self._get_analogy(detected_topic)}"
            
            if articles:
                current = ' | '.join(a['title'] for a in articles)
                return f"{historical} | Current developments: {current}"
            
            return historical
            
        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            return f"Historical perspective: {self._get_analogy('default')}" 