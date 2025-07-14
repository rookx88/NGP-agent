import logging
from twitter.client import get_twitter_client
from twitter.core import get_client
from rapidfuzz import fuzz
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import spacy
import os
from collections import Counter, defaultdict
from openai import OpenAI
import time
import json
from datetime import datetime, timedelta

# Set up logging to print to console at INFO level
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# User's personal brand themes (same as assessor.py)
THEMES = {
    'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'automation', 'neural network', 'deep learning', 'algorithm', 'data science', 'robotics', 'automation', 'intelligence', 'smart', 'predictive', 'analytics'],
    'Sports': ['soccer', 'football', 'basketball', 'nba', 'nfl', 'fifa', 'premier league', 'champions league', 'world cup', 'player', 'team', 'match', 'game', 'goal', 'score', 'transfer', 'coach', 'tactics', 'league', 'cup', 'tournament', 'championship', 'playoff', 'draft', 'trade', 'athlete', 'sport'],
    'Technology': ['tech', 'technology', 'software', 'hardware', 'startup', 'innovation', 'digital', 'app', 'platform', 'software', 'coding', 'programming', 'development', 'cybersecurity', 'cloud', 'blockchain', 'vr', 'ar', 'iot', 'mobile', 'web', 'api', 'database'],
    'Gaming': ['gaming', 'game', 'esports', 'streaming', 'twitch', 'youtube', 'console', 'pc', 'mobile game', 'rpg', 'fps', 'battle royale', 'mmo', 'indie game', 'game development', 'gamer', 'playstation', 'xbox', 'nintendo', 'steam', 'discord']
}

# Configuration for influencer analysis
INFLUENCER_CONFIG = {
    'max_users': 5,                    # Limit number of users to analyze
    'posts_per_user': 20,              # How many posts to fetch per user
    'min_post_quality': 0.3,           # Lowered minimum quality score
    'prefer_recent_days': 7,           # Prefer posts from last X days
    'engagement_types': ['reply', 'quote_tweet', 'mention'],  # Response types
    'rate_limit_delay': 3.0,           # More conservative for user API calls
    'max_trends_per_theme': 2,         # Number of top posts to process per theme
    'enable_quality_fallback': True,   # Use alternative metrics when engagement data is missing
    'user_weights': {                  # Weight users by relevance/importance
        'verified': 1.0,
        'high_followers': 0.9,
        'medium_followers': 0.7,
        'low_followers': 0.5
    }
}

# Quality scoring weights for posts
QUALITY_WEIGHTS = {
    'engagement_rate': 0.30,           # Likes, retweets, replies relative to followers
    'content_length': 0.15,            # Longer posts often indicate more thought
    'recency_score': 0.20,             # Recent posts are more relevant
    'originality_score': 0.15,         # Original content vs retweets
    'theme_relevance': 0.20            # How well it matches your themes
}

# Rate limiting configuration (more conservative for user API calls)
RATE_LIMIT_CONFIG = {
    'requests_per_15min': 100,         # More conservative for user timeline API
    'requests_per_hour': 300,          # More conservative for user timeline API
    'min_delay_between_requests': 3.0, # Increased delay between requests
    'max_delay_between_requests': 15.0, # Increased maximum delay for backoff
    'exponential_backoff_base': 3.0,   # More aggressive backoff
}

# Global rate limiting state
rate_limit_state = {
    'last_request_time': 0,
    'requests_this_hour': 0,
    'requests_this_15min': 0,
    'hour_start': time.time(),
    'minute_start': time.time()
}

# Confidence scores for theme matching (same as assessor.py)
LAYER_CONFIDENCE = {
    'manual': 1.0,
    'direct': 0.9,
    'fuzzy': 0.7,
    'semantic': 0.6,
    'ner': 0.5,
}

# Manual overrides for known influencers and their typical themes
INFLUENCER_THEME_OVERRIDES = {
    'elonmusk': ['AI', 'Technology'],
    'naval': ['AI', 'Technology'],
    'sama': ['AI', 'Technology'],
    'patrickc': ['Technology', 'AI'],
    'balajis': ['Technology', 'AI'],
    'pmarca': ['Technology', 'AI'],
    'paulg': ['Technology', 'AI'],
    'dhh': ['Technology'],
    'jason': ['Technology'],
    'shl': ['Technology'],
    'tferriss': ['Technology', 'AI'],
    'jaltucher': ['Technology', 'AI'],
    'naval': ['Technology', 'AI'],
    'sivers': ['Technology', 'AI'],
    'jamesclear': ['Technology', 'AI'],
    'calnewport': ['Technology', 'AI'],
    'shane_parrish': ['Technology', 'AI'],
    'morganhousel': ['Technology', 'AI'],
    'david_perell': ['Technology', 'AI'],
    'sahilbloom': ['Technology', 'AI'],
    'alexhormozi': ['Technology', 'AI'],
    'thedankoe': ['Technology', 'AI'],
    'dickiebush': ['Technology', 'AI'],
    'niche': ['Technology', 'AI'],
    'julian': ['Technology', 'AI'],
    'jackbutcher': ['Technology', 'AI'],
    'khehy': ['Technology', 'AI'],
    'gregisenberg': ['Technology', 'AI'],
    'davidgoggins': ['Sports'],
    'jockowillink': ['Sports'],
    'andrew_huberman': ['Technology', 'AI'],
    'lexfridman': ['AI', 'Technology'],
    'joerogan': ['Sports', 'Technology'],
    'tferriss': ['Technology', 'AI'],
    'naval': ['Technology', 'AI'],
    'sama': ['AI', 'Technology'],
    'patrickc': ['Technology', 'AI'],
    'balajis': ['Technology', 'AI'],
    'pmarca': ['Technology', 'AI'],
    'paulg': ['Technology', 'AI'],
    'dhh': ['Technology'],
    'jason': ['Technology'],
    'shl': ['Technology'],
    'tferriss': ['Technology', 'AI'],
    'jaltucher': ['Technology', 'AI'],
    'naval': ['Technology', 'AI'],
    'sivers': ['Technology', 'AI'],
    'jamesclear': ['Technology', 'AI'],
    'calnewport': ['Technology', 'AI'],
    'shane_parrish': ['Technology', 'AI'],
    'morganhousel': ['Technology', 'AI'],
    'david_perell': ['Technology', 'AI'],
    'sahilbloom': ['Technology', 'AI'],
    'alexhormozi': ['Technology', 'AI'],
    'thedankoe': ['Technology', 'AI'],
    'dickiebush': ['Technology', 'AI'],
    'niche': ['Technology', 'AI'],
    'julian': ['Technology', 'AI'],
    'jackbutcher': ['Technology', 'AI'],
    'khehy': ['Technology', 'AI'],
    'gregisenberg': ['Technology', 'AI'],
    'davidgoggins': ['Sports'],
    'jockowillink': ['Sports'],
    'andrew_huberman': ['Technology', 'AI'],
    'lexfridman': ['AI', 'Technology'],
    'joerogan': ['Sports', 'Technology']
}

# Load required models (same as assessor.py)
try:
    import nltk
    nltk.data.find('corpora/wordnet')
except LookupError:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

MODEL_NAME = os.environ.get('SENTENCE_TRANSFORMERS_MODEL', 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_NAME)
theme_sentences = {theme: " ".join(keywords) for theme, keywords in THEMES.items()}
theme_embeddings = {theme: model.encode(sentence) for theme, sentence in theme_sentences.items()}


def update_rate_limit_state():
    """Update rate limiting state and enforce delays."""
    current_time = time.time()
    
    # Reset hourly counter if needed
    if current_time - rate_limit_state['hour_start'] >= 3600:
        rate_limit_state['requests_this_hour'] = 0
        rate_limit_state['hour_start'] = current_time
    
    # Reset 15-minute counter if needed
    if current_time - rate_limit_state['minute_start'] >= 900:
        rate_limit_state['requests_this_15min'] = 0
        rate_limit_state['minute_start'] = current_time
    
    # Enforce minimum delay between requests
    time_since_last = current_time - rate_limit_state['last_request_time']
    min_delay = RATE_LIMIT_CONFIG['min_delay_between_requests']
    
    if time_since_last < min_delay:
        sleep_time = min_delay - time_since_last
        logger.info(f"Rate limiting: waiting {sleep_time:.2f}s between requests")
        time.sleep(sleep_time)
    
    rate_limit_state['last_request_time'] = time.time()
    rate_limit_state['requests_this_hour'] += 1
    rate_limit_state['requests_this_15min'] += 1


def should_skip_user_due_to_rate_limit():
    """Check if we should skip users due to rate limiting."""
    hour_limit = RATE_LIMIT_CONFIG['requests_per_hour'] * 0.8
    minute_limit = RATE_LIMIT_CONFIG['requests_per_15min'] * 0.8
    
    if (rate_limit_state['requests_this_hour'] >= hour_limit or
        rate_limit_state['requests_this_15min'] >= minute_limit):
        
        logger.warning(f"Approaching rate limits - Hour: {rate_limit_state['requests_this_hour']}/{hour_limit:.0f}, "
                      f"15min: {rate_limit_state['requests_this_15min']}/{minute_limit:.0f}")
        return True
    return False


def log_rate_limit_status():
    """Log current rate limiting status for debugging."""
    hour_limit = RATE_LIMIT_CONFIG['requests_per_hour']
    minute_limit = RATE_LIMIT_CONFIG['requests_per_15min']
    
    hour_usage = (rate_limit_state['requests_this_hour'] / hour_limit) * 100
    minute_usage = (rate_limit_state['requests_this_15min'] / minute_limit) * 100
    
    logger.info(f"Rate limit status - Hour: {rate_limit_state['requests_this_hour']}/{hour_limit} ({hour_usage:.1f}%), "
                f"15min: {rate_limit_state['requests_this_15min']}/{minute_limit} ({minute_usage:.1f}%)")


def validate_usernames(usernames):
    """
    Validate that usernames exist and are accessible.
    Returns list of valid usernames.
    """
    client = get_client()
    valid_usernames = []
    
    for username in usernames:
        try:
            update_rate_limit_state()
            logger.info(f"Validating username: {username}")
            
            # Remove @ if present
            clean_username = username.lstrip('@')
            
            # Try to get user info using Tweepy's get_user method
            user_info = client.get_user(username=clean_username)
            
            if user_info and user_info.get('data'):
                valid_usernames.append(clean_username)
                logger.info(f"✅ Validated: {clean_username}")
            else:
                logger.warning(f"❌ User not found: {clean_username}")
                
        except Exception as e:
            logger.error(f"❌ Error validating {username}: {e}")
            continue
    
    logger.info(f"Validated {len(valid_usernames)} out of {len(usernames)} usernames")
    return valid_usernames


def fetch_user_posts(username, max_posts=20):
    """
    Fetch recent posts from a user with enhanced error handling.
    """
    client = get_client()
    logger.info(f"Fetching {max_posts} recent posts from @{username}...")
    
    try:
        update_rate_limit_state()
        
        # Get user ID first
        user_info = client.get_user(username=username)
        if not user_info or not user_info.get('data'):
            logger.error(f"Could not get user info for {username}")
            return []
        
        user_id = user_info['data']['id']
        if not user_id:
            logger.error(f"Could not get user ID for {username}")
            return []
        
        # Fetch user's tweets
        tweets = client.get_users_tweets(
            user_id, 
            max_results=max_posts,
            tweet_fields=['public_metrics', 'created_at', 'text', 'referenced_tweets']
        )
        
        if not tweets or not tweets.get('data'):
            logger.warning(f"No tweets found for {username}")
            return []
        
        # Process tweets
        processed_posts = []
        for tweet in tweets['data']:
            # Skip retweets and replies to focus on original content
            if tweet.get('referenced_tweets'):
                ref_type = tweet['referenced_tweets'][0].get('type', '')
                if ref_type in ['retweeted', 'replied_to']:
                    continue
            
            processed_post = {
                'id': tweet.get('id'),
                'text': tweet.get('text', ''),
                'created_at': tweet.get('created_at'),
                'public_metrics': tweet.get('public_metrics', {}),
                'username': username,
                'user_id': user_id
            }
            
            # Skip empty or very short posts
            if len(processed_post['text'].strip()) < 10:
                continue
                
            processed_posts.append(processed_post)
        
        logger.info(f"Fetched {len(processed_posts)} original posts from @{username}")
        return processed_posts
        
    except Exception as e:
        logger.error(f"Failed to fetch posts for {username}: {e}")
        return []


def calculate_post_quality_score(post, user_history=None):
    """
    Calculate quality score for a post based on multiple factors.
    """
    scores = {}
    
    # 1. Engagement rate (0-1)
    metrics = post.get('public_metrics', {})
    total_engagement = (
        metrics.get('like_count', 0) + 
        metrics.get('retweet_count', 0) * 2 +  # Retweets worth more
        metrics.get('reply_count', 0) * 3      # Replies worth most
    )
    
    # Normalize by follower count (estimate if not available)
    estimated_followers = 10000  # Default estimate
    if user_history and len(user_history) > 0:
        # Calculate average engagement to estimate followers
        avg_engagement = sum(
            p.get('public_metrics', {}).get('like_count', 0) 
            for p in user_history
        ) / len(user_history)
        estimated_followers = max(1000, avg_engagement * 100)  # Rough estimate
    
    engagement_rate = min(total_engagement / max(estimated_followers, 1), 1.0)
    scores['engagement_rate'] = engagement_rate
    
    # 2. Content length score (0-1)
    text_length = len(post.get('text', ''))
    # Optimal length is 100-200 characters, with diminishing returns
    if text_length < 50:
        length_score = text_length / 50
    elif text_length < 200:
        length_score = 1.0
    else:
        length_score = max(0.5, 1.0 - (text_length - 200) / 800)
    scores['content_length'] = length_score
    
    # 3. Recency score (0-1)
    created_at = post.get('created_at')
    if created_at:
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        days_old = (datetime.now(created_at.tzinfo) - created_at).days
        recency_score = max(0, 1.0 - (days_old / INFLUENCER_CONFIG['prefer_recent_days']))
    else:
        recency_score = 0.5  # Default if no date
    scores['recency_score'] = recency_score
    
    # 4. Originality score (0-1)
    # Already filtered out retweets, so give high score for original content
    scores['originality_score'] = 1.0
    
    # 5. Theme relevance score (0-1)
    themes = match_themes(post.get('text', ''))
    theme_relevance = len(themes) / len(THEMES) if themes else 0
    scores['theme_relevance'] = theme_relevance
    
    # Calculate weighted total score
    total_score = sum(
        scores[factor] * QUALITY_WEIGHTS[factor]
        for factor in QUALITY_WEIGHTS.keys()
    )
    
    post['quality_score'] = total_score
    post['quality_breakdown'] = scores
    
    logger.info(f"Quality score for @{post['username']}: {total_score:.3f} "
                f"(engagement: {scores['engagement_rate']:.3f}, "
                f"length: {scores['content_length']:.3f}, "
                f"recency: {scores['recency_score']:.3f}, "
                f"themes: {themes})")
    
    return total_score


def select_best_post_per_user(user_posts):
    """
    Select the best post from each user based on quality scoring.
    """
    best_posts = {}
    
    for username, posts in user_posts.items():
        if not posts:
            continue
        
        # Calculate quality scores for all posts
        for post in posts:
            calculate_post_quality_score(post, posts)
        
        # Select the best post
        best_post = max(posts, key=lambda p: p.get('quality_score', 0))
        
        # Only include if quality meets minimum threshold
        if best_post.get('quality_score', 0) >= INFLUENCER_CONFIG['min_post_quality']:
            best_posts[username] = best_post
            logger.info(f"Selected best post for @{username}: "
                       f"score {best_post['quality_score']:.3f}, "
                       f"text: {best_post['text'][:100]}...")
        else:
            logger.warning(f"No posts met quality threshold for @{username}")
    
    return best_posts


def fetch_user_posts_multi_user(usernames, max_users=None):
    """
    Fetch posts from multiple users with intelligent rate limiting.
    """
    logger.info(f"Fetching posts from {len(usernames)} users...")
    
    # Limit users if specified
    if max_users:
        usernames = usernames[:max_users]
    
    user_posts = {}
    successful_users = 0
    failed_users = 0
    
    for username in usernames:
        # Check rate limits before each request
        if should_skip_user_due_to_rate_limit():
            logger.warning(f"Skipping {username} due to rate limiting")
            break
        
        try:
            posts = fetch_user_posts(username, INFLUENCER_CONFIG['posts_per_user'])
            if posts:
                user_posts[username] = posts
                successful_users += 1
                logger.info(f"Successfully fetched {len(posts)} posts from @{username}")
            else:
                failed_users += 1
                logger.warning(f"No posts returned for @{username}")
                
        except Exception as e:
            logger.error(f"Error fetching posts for @{username}: {e}")
            failed_users += 1
            continue
    
    logger.info(f"User fetch summary: {successful_users} successful, {failed_users} failed")
    return user_posts


# Theme matching functions (adapted from assessor.py)
def expand_synonyms(keywords):
    expanded = set(keywords)
    for word in keywords:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return list(expanded)

THEMES_EXPANDED = {theme: expand_synonyms(keywords) for theme, keywords in THEMES.items()}


def direct_keyword_match(text):
    matches = []
    text_lower = text.lower()
    for theme, keywords in THEMES_EXPANDED.items():
        if any(kw in text_lower for kw in keywords):
            matches.append(theme)
    if matches:
        logger.info(f"Direct keyword match: {matches}")
    return matches


def fuzzy_match(text, threshold=82):
    matches = []
    text_lower = text.lower()
    for theme, keywords in THEMES_EXPANDED.items():
        for kw in keywords:
            score = fuzz.partial_ratio(kw, text_lower)
            if score >= threshold:
                matches.append(theme)
                break
    if matches:
        logger.info(f"Fuzzy match: {matches}")
    return matches


def semantic_similarity_match(text, threshold=0.45):
    matches = []
    text_emb = model.encode(text)
    for theme, emb in theme_embeddings.items():
        score = util.cos_sim(text_emb, emb).item()
        if score > threshold:
            matches.append(theme)
    if matches:
        logger.info(f"Semantic similarity match: {matches}")
    return matches


def ner_match(text):
    matches = []
    doc = nlp(text)
    for ent in doc.ents:
        ent_text = ent.text.lower()
        for theme, keywords in THEMES_EXPANDED.items():
            if any(kw in ent_text for kw in keywords):
                matches.append(theme)
    if matches:
        logger.info(f"NER match: {matches}")
    return matches


def manual_override(username, text):
    # Check for influencer-specific theme overrides
    if username in INFLUENCER_THEME_OVERRIDES:
        themes = INFLUENCER_THEME_OVERRIDES[username]
        logger.info(f"Manual override for @{username}: {themes}")
        return themes
    
    # Check for manual overrides in text content
    manual_overrides = {
        'Red Planet': 'AI',
        'ISS': 'Technology',
        'Elon Musk': 'AI',
        'ChatGPT': 'AI',
        'OpenAI': 'AI',
        'GPT': 'AI',
        'Machine Learning': 'AI',
        'Deep Learning': 'AI',
        'Neural Network': 'AI',
        'Algorithm': 'AI',
        'Data Science': 'AI',
        'Robotics': 'AI',
        'Automation': 'AI',
        'Intelligence': 'AI',
        'Smart': 'AI',
        'Predictive': 'AI',
        'Analytics': 'AI',
        'Soccer': 'Sports',
        'Football': 'Sports',
        'Basketball': 'Sports',
        'NBA': 'Sports',
        'NFL': 'Sports',
        'FIFA': 'Sports',
        'Premier League': 'Sports',
        'Champions League': 'Sports',
        'World Cup': 'Sports',
        'Player': 'Sports',
        'Team': 'Sports',
        'Match': 'Sports',
        'Game': 'Sports',
        'Goal': 'Sports',
        'Score': 'Sports',
        'Transfer': 'Sports',
        'Coach': 'Sports',
        'Tactics': 'Sports',
        'League': 'Sports',
        'Cup': 'Sports',
        'Tournament': 'Sports',
        'Championship': 'Sports',
        'Playoff': 'Sports',
        'Draft': 'Sports',
        'Trade': 'Sports',
        'Athlete': 'Sports',
        'Sport': 'Sports',
        'Tech': 'Technology',
        'Technology': 'Technology',
        'Software': 'Technology',
        'Hardware': 'Technology',
        'Startup': 'Technology',
        'Innovation': 'Technology',
        'Digital': 'Technology',
        'App': 'Technology',
        'Platform': 'Technology',
        'Coding': 'Technology',
        'Programming': 'Technology',
        'Development': 'Technology',
        'Cybersecurity': 'Technology',
        'Cloud': 'Technology',
        'Blockchain': 'Technology',
        'VR': 'Technology',
        'AR': 'Technology',
        'IoT': 'Technology',
        'Mobile': 'Technology',
        'Web': 'Technology',
        'API': 'Technology',
        'Database': 'Technology',
        'Gaming': 'Gaming',
        'Game': 'Gaming',
        'Esports': 'Gaming',
        'Streaming': 'Gaming',
        'Twitch': 'Gaming',
        'YouTube': 'Gaming',
        'Console': 'Gaming',
        'PC': 'Gaming',
        'Mobile Game': 'Gaming',
        'RPG': 'Gaming',
        'FPS': 'Gaming',
        'Battle Royale': 'Gaming',
        'MMO': 'Gaming',
        'Indie Game': 'Gaming',
        'Game Development': 'Gaming',
        'Gamer': 'Gaming',
        'PlayStation': 'Gaming',
        'Xbox': 'Gaming',
        'Nintendo': 'Gaming',
        'Steam': 'Gaming',
        'Discord': 'Gaming'
    }
    
    for key, theme in manual_overrides.items():
        if key.lower() in text.lower():
            logger.info(f"Manual override in text: {theme}")
            return [theme]
    
    return []


def match_themes(text):
    """
    Match themes using multiple layers (adapted from assessor.py).
    """
    # 1. Manual override
    manual = manual_override("", text)  # We'll handle username separately
    if manual:
        return manual
    
    # 2. Direct keyword match
    direct = direct_keyword_match(text)
    if direct:
        return direct
    
    # 3. Fuzzy match
    fuzzy = fuzzy_match(text)
    if fuzzy:
        return fuzzy
    
    # 4. Semantic similarity
    semantic = semantic_similarity_match(text)
    if semantic:
        return semantic
    
    # 5. NER
    ner = ner_match(text)
    if ner:
        return ner
    
    return []


def match_themes_with_confidence(text):
    """
    Match themes with confidence scores (adapted from assessor.py).
    """
    # 1. Manual override
    manual = manual_override("", text)
    if manual:
        return manual, LAYER_CONFIDENCE['manual']
    
    # 2. Direct keyword match
    direct = direct_keyword_match(text)
    if direct:
        return direct, LAYER_CONFIDENCE['direct']
    
    # 3. Fuzzy match
    fuzzy = fuzzy_match(text)
    if fuzzy:
        return fuzzy, LAYER_CONFIDENCE['fuzzy']
    
    # 4. Semantic similarity
    semantic = semantic_similarity_match(text)
    if semantic:
        return semantic, LAYER_CONFIDENCE['semantic']
    
    # 5. NER
    ner = ner_match(text)
    if ner:
        return ner, LAYER_CONFIDENCE['ner']
    
    return [], 0.0


def generate_response_angles(post_content, username, theme, n_angles=4):
    """
    Generate response angles for engaging with an influencer's post.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    prompt = f'''
You're a {theme} expert looking to engage with @{username}'s post. Create {n_angles} unique response angles.

Post: "{post_content}"

Each angle should be:
- Provocative and polarizing when appropriate
- Direct and confrontational
- Designed to spark heated debate
- Under 80 characters
- No hashtags or excessive punctuation
- Short and punchy

Response angle types:
- Contrarian: Challenge their assumptions
- Questioning: Ask tough questions
- Disagree: Respectfully but firmly disagree
- Challenge: Push back on their logic
- Provocative: Make a bold statement

Format: Return only the angles, one per line, no numbering.
'''
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a social media engagement strategist."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.9,
            max_tokens=400
        )
        
        text = response.choices[0].message.content.strip()
        angles = [line.strip() for line in text.split('\n') if line.strip()]
        
        return angles[:n_angles]
        
    except Exception as e:
        logger.error(f"Error generating response angles: {e}")
        return []


def draft_influencer_response(angle, post_content, username, theme, format="reply", max_attempts=3):
    """
    Draft a response to an influencer's post.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_attempts):
        if format == "reply":
            prompt = f'''
Write a reply to @{username}'s post from this angle: "{angle}"

Original post: "{post_content}"

Style guidelines:
- Direct and to the point
- Provocative when appropriate
- Minimal punctuation
- No hashtags
- Under 200 characters
- Use 1 emoji max if needed

Tone: Confident, direct, sometimes confrontational.
'''
        elif format == "quote_tweet":
            prompt = f'''
Write a quote tweet about @{username}'s post from this angle: "{angle}"

Original post: "{post_content}"

Style guidelines:
- Direct and to the point
- Provocative when appropriate
- Minimal punctuation
- No hashtags
- Under 200 characters
- Use 1 emoji max if needed

Tone: Confident, direct, sometimes confrontational.
'''
        else:  # mention
            prompt = f'''
Write a tweet mentioning @{username} from this angle: "{angle}"

Original post: "{post_content}"

Style guidelines:
- Direct and to the point
- Provocative when appropriate
- Minimal punctuation
- No hashtags
- Under 200 characters
- Use 1 emoji max if needed

Tone: Confident, direct, sometimes confrontational.
'''
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": "You are a social media copywriter."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.9,
                max_tokens=200
            )
            
            text = response.choices[0].message.content.strip()
            
            # Clean up the response
            if len(text) > 280:
                text = text[:277] + '...'
            
            return text
            
        except Exception as e:
            logger.error(f"Error drafting response (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)
    
    return "Great post! Thanks for sharing this perspective."


def analyze_engagement_strategy(post, username, theme):
    """
    Analyze and suggest the best engagement strategy for a post.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    prompt = f'''
Analyze this post from @{username} and suggest the best engagement strategy.

Post: "{post.get('text', '')}"
Theme: {theme}
Engagement metrics: {post.get('public_metrics', {})}

Suggest the best approach:
1. Reply - for direct conversation
2. Quote tweet - for broader reach
3. Mention - for casual engagement

Consider:
- Post engagement level
- Content type (thoughtful vs casual)
- User's typical response patterns
- Your {theme} expertise

Return: "REPLY", "QUOTE_TWEET", or "MENTION" with a brief reason.
'''
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        if "REPLY" in result:
            return "reply"
        elif "QUOTE" in result:
            return "quote_tweet"
        elif "MENTION" in result:
            return "mention"
        else:
            return "reply"  # Default
            
    except Exception as e:
        logger.error(f"Error analyzing engagement strategy: {e}")
        return "reply"


def create_visual_asset(angle, post_content, username, theme):
    """
    Create a visual asset for the engagement response.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    prompt_gen = f'''
Post: "{post_content}"
User: @{username}
Theme: {theme}
Response Angle: {angle}

Write a descriptive prompt for an image that represents this engagement opportunity.
Focus on the {theme} theme and the conversation context.

Format: Image prompt: ...
'''
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a creative visual director."
            }, {
                "role": "user",
                "content": prompt_gen
            }],
            temperature=0.8,
            max_tokens=200
        )
        
        text = response.choices[0].message.content.strip()
        image_prompt = text
        if 'Image prompt:' in text:
            image_prompt = text.split('Image prompt:')[-1].strip()
        
        # Generate image
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        
        return image_response.data[0].url
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return None


def select_best_response_angle(angles, post_content, username, theme):
    """
    Select the best response angle using AI.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return angles[0] if angles else ""
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Given these response angles for @{username}'s post (theme: {theme}):
    
    {chr(10).join(f"{i+1}. {angle}" for i, angle in enumerate(angles))}
    
    Original post: "{post_content}"
    
    Select the angle that is:
    1. Most likely to spark meaningful conversation
    2. Most authentic and genuine
    3. Best aligned with {theme} expertise
    4. Most likely to get a response from @{username}
    5. Most valuable to the community
    
    Return ONLY the selected angle text (not the number).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        selected_angle = response.choices[0].message.content.strip()
        logger.info(f"AI selected angle: {selected_angle}")
        
        # Validate that the selected angle is actually in our list
        if selected_angle in angles:
            return selected_angle
        else:
            # Find closest match
            for angle in angles:
                if fuzz.ratio(selected_angle.lower(), angle.lower()) > 80:
                    return angle
            
            # Fallback to first angle
            return angles[0] if angles else ""
            
    except Exception as e:
        logger.error(f"Error selecting best angle: {e}")
        return angles[0] if angles else ""


def main(usernames):
    """
    Main workflow: analyze influencer posts and generate engagement strategies.
    """
    logger.info("Starting influencer analysis and engagement strategy generation...")
    
    # Reset rate limiting state
    global rate_limit_state
    rate_limit_state = {
        'last_request_time': 0,
        'requests_this_hour': 0,
        'requests_this_15min': 0,
        'hour_start': time.time(),
        'minute_start': time.time()
    }
    
    # Validate usernames
    valid_usernames = validate_usernames(usernames)
    if not valid_usernames:
        logger.error("No valid usernames provided")
        return []
    
    # Limit to max users
    valid_usernames = valid_usernames[:INFLUENCER_CONFIG['max_users']]
    
    logger.info(f"Analyzing {len(valid_usernames)} users")
    log_rate_limit_status()
    
    # Fetch posts from all users
    user_posts = fetch_user_posts_multi_user(valid_usernames)
    
    # Log final rate limit status
    log_rate_limit_status()
    
    if not user_posts:
        logger.error("No posts fetched from any user")
        return []
    
    # Select best post from each user
    best_posts = select_best_post_per_user(user_posts)
    
    if not best_posts:
        logger.warning("No posts met quality threshold")
        return []
    
    # Generate engagement packages
    engagement_packages = []
    
    for username, post in best_posts.items():
        logger.info(f"\n=== Processing @{username} ===")
        
        try:
            # Match themes
            themes = match_themes(post['text'])
            if not themes:
                logger.warning(f"No theme match for @{username}'s post")
                continue
            
            # Use first theme for now (could be enhanced to handle multiple)
            theme = themes[0]
            
            # Generate response angles
            angles = generate_response_angles(post['text'], username, theme)
            if not angles:
                logger.warning(f"No angles generated for @{username}")
                continue
            
            # Select best angle
            chosen_angle = select_best_response_angle(angles, post['text'], username, theme)
            
            # Analyze engagement strategy
            strategy = analyze_engagement_strategy(post, username, theme)
            
            # Draft response
            response = draft_influencer_response(chosen_angle, post['text'], username, theme, strategy)
            
            # Create visual asset
            visual_url = create_visual_asset(chosen_angle, post['text'], username, theme)
            
            # Create engagement package
            engagement_package = {
                'username': username,
                'post_id': post['id'],
                'post_text': post['text'],
                'post_created_at': post['created_at'],
                'post_metrics': post['public_metrics'],
                'quality_score': post['quality_score'],
                'quality_breakdown': post['quality_breakdown'],
                'theme': theme,
                'all_angles': angles,
                'chosen_angle': chosen_angle,
                'engagement_strategy': strategy,
                'drafted_response': response,
                'visual_url': visual_url
            }
            
            engagement_packages.append(engagement_package)
            
            # Print results
            print(f"\n🎯 ENGAGEMENT PACKAGE: @{username}")
            print(f"📊 Quality Score: {post['quality_score']:.3f}")
            print(f"🎨 Theme: {theme}")
            print(f"📝 Original Post: {post['text'][:100]}...")
            print(f"💡 Chosen Angle: {chosen_angle}")
            print(f"📱 Strategy: {strategy.upper()}")
            print(f"💬 Response: {response}")
            if visual_url:
                print(f"🖼️  Visual: {visual_url}")
            print("-" * 80)
            
        except Exception as e:
            logger.error(f"Error processing @{username}: {e}")
            continue
    
    logger.info(f"Generated {len(engagement_packages)} engagement packages")
    return engagement_packages


def export_results(engagement_packages, filename=None):
    """
    Export results to JSON file with timestamp.
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"influencer_engagement_results_{timestamp}.json"
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'config': INFLUENCER_CONFIG,
        'engagement_packages': engagement_packages,
        'summary': {
            'total_packages': len(engagement_packages),
            'themes_covered': list(set(pkg['theme'] for pkg in engagement_packages)),
            'strategies_used': list(set(pkg['engagement_strategy'] for pkg in engagement_packages))
        }
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results exported to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    usernames = ["elonmusk", "naval", "sama"]  # Replace with actual usernames
    
    print("🚀 Starting Influencer Analysis and Engagement Strategy Generation")
    print(f"📋 Configuration: Max users={INFLUENCER_CONFIG['max_users']}, Posts per user={INFLUENCER_CONFIG['posts_per_user']}")
    
    results = main(usernames)
    
    if results:
        # Export results
        export_filename = export_results(results)
        
        # Print final summary
        print(f"\n✅ Analysis complete! Generated {len(results)} engagement packages")
        if export_filename:
            print(f"📁 Results saved to: {export_filename}")
        
        # Print JSON output for programmatic use
        print("\n" + "="*80)
        print("JSON OUTPUT:")
        print("="*80)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("❌ No results generated") 