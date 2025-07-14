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

# User's personal brand themes
THEMES = {
    'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'automation', 'neural network', 'deep learning', 'algorithm', 'data science', 'robotics', 'automation', 'intelligence', 'smart', 'predictive', 'analytics'],
    'Sports': ['soccer', 'football', 'basketball', 'nba', 'nfl', 'fifa', 'premier league', 'champions league', 'world cup', 'player', 'team', 'match', 'game', 'goal', 'score', 'transfer', 'coach', 'tactics', 'league', 'cup', 'tournament', 'championship', 'playoff', 'draft', 'trade', 'athlete', 'sport'],
    'Technology': ['tech', 'technology', 'software', 'hardware', 'startup', 'innovation', 'digital', 'app', 'platform', 'software', 'coding', 'programming', 'development', 'cybersecurity', 'cloud', 'blockchain', 'vr', 'ar', 'iot', 'mobile', 'web', 'api', 'database'],
    'Gaming': ['gaming', 'game', 'esports', 'streaming', 'twitch', 'youtube', 'console', 'pc', 'mobile game', 'rpg', 'fps', 'battle royale', 'mmo', 'indie game', 'game development', 'gamer', 'playstation', 'xbox', 'nintendo', 'steam', 'discord']
}

# Multi-region WOEIDs for comprehensive trend coverage
REGIONS = {
    'US': 23424977,
    'UK': 23424975,
    'Canada': 23424775,
    'Australia': 23424748,
    'Germany': 23424829,
    'France': 23424819,
    'Japan': 23424856,
    'Brazil': 23424768,
    'India': 23424848,
    'Mexico': 23424900,
    'Spain': 23424950,
    'Italy': 23424853,
    'Netherlands': 23424909,
    'South Korea': 23424868,
    'Turkey': 23424969
}

# Configuration settings
CONFIG = {
    'use_multi_region': True,      # Set to False for single-region (US only)
    'max_regions': 5,              # Conservative: limit to 5 regions to avoid rate limiting
    'max_trends_per_theme': 2,     # Number of top trends to process per theme
    'min_engagement_score': 0.03,  # Lowered further to account for missing volume data
    'rate_limit_delay': 2.0,       # Increased delay between API calls (seconds)
    'enable_volume_fallback': True, # Use alternative metrics when volume data is missing
    'region_weights': {            # Weight regions by relevance/importance
        'US': 1.0,
        'UK': 0.9,
        'Canada': 0.8,
        'Australia': 0.7,
        'Germany': 0.6,
        'France': 0.6,
        'Japan': 0.5,
        'Brazil': 0.5,
        'India': 0.4,
        'Mexico': 0.4,
        'Spain': 0.5,
        'Italy': 0.5,
        'Netherlands': 0.4,
        'South Korea': 0.4,
        'Turkey': 0.3
    }
}

# Engagement scoring weights (updated for better balance)
ENGAGEMENT_WEIGHTS = {
    'cross_region_presence': 0.25,  # Reduced from 0.3
    'volume_score': 0.20,          # Reduced from 0.25
    'controversy_score': 0.20,     # Same
    'cross_theme_appeal': 0.15,    # Same
    'velocity_score': 0.10,        # Same
    'region_weight_score': 0.10    # NEW: Weighted by region importance
}

# Manual override for known ambiguous or recurring trends
MANUAL_OVERRIDES = {
    'Red Planet': 'Mars/Space',
    'ISS': 'Mars/Space',
    'Elon Musk': 'Mars/Space',
    'Backcountry': 'Camping',
    'BBQ': 'Cooking',
}

# Confidence scores for each layer
LAYER_CONFIDENCE = {
    'manual': 1.0,
    'direct': 0.9,
    'fuzzy': 0.7,
    'semantic': 0.6,
    'ner': 0.5,
}

# Blacklist of ambiguous terms that require tweet validation
AMBIGUOUS_TERMS = {'Poles', 'Hill', 'Mitchell', 'Jones', 'Phil', 'Simpson', 'Dobbins', 'Holland', 'Metcalf', 'Dilbert', 'Bondi', 'Serrano', 'Taylor', 'Paul', 'Jake', 'Reed', 'Sheppard', 'Dustin', 'May', 'Gerard', 'Vientos', 'Alycia', 'Nicolandria', 'Chelley', 'Bryan', 'Huda', 'Chris', 'Iris'}

# Synonym expansion using WordNet
try:
    import nltk
    nltk.data.find('corpora/wordnet')
except LookupError:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load sentence-transformers model for semantic similarity
MODEL_NAME = os.environ.get('SENTENCE_TRANSFORMERS_MODEL', 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_NAME)
theme_sentences = {theme: " ".join(keywords) for theme, keywords in THEMES.items()}
theme_embeddings = {theme: model.encode(sentence) for theme, sentence in theme_sentences.items()}


def expand_synonyms(keywords):
    expanded = set(keywords)
    for word in keywords:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return list(expanded)

# Precompute expanded theme keywords
THEMES_EXPANDED = {theme: expand_synonyms(keywords) for theme, keywords in THEMES.items()}


# Configuration for rate limiting and API management
RATE_LIMIT_CONFIG = {
    'requests_per_15min': 150,  # More conservative estimate
    'requests_per_hour': 500,   # More conservative estimate
    'min_delay_between_requests': 2.0,  # Increased minimum delay between requests
    'max_delay_between_requests': 10.0, # Increased maximum delay for backoff
    'exponential_backoff_base': 2.5,    # More aggressive backoff
    'region_priority': {
        'US': 1, 'UK': 2, 'Canada': 3, 'Australia': 4, 'Germany': 5,
        'France': 6, 'Japan': 7, 'Brazil': 8, 'India': 9, 'Mexico': 10,
        'Spain': 11, 'Italy': 12, 'Netherlands': 13, 'South Korea': 14, 'Turkey': 15
    }
}

# Global rate limiting state
rate_limit_state = {
    'last_request_time': 0,
    'requests_this_hour': 0,
    'requests_this_15min': 0,
    'hour_start': time.time(),
    'minute_start': time.time()
}

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

def should_skip_region_due_to_rate_limit():
    """Check if we should skip regions due to rate limiting."""
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

def get_region_delay(region_name, attempt=1):
    """Get appropriate delay for a region based on priority and attempt number."""
    base_delay = RATE_LIMIT_CONFIG['min_delay_between_requests']
    priority = RATE_LIMIT_CONFIG['region_priority'].get(region_name, 10)
    
    # Higher priority regions get shorter delays
    priority_multiplier = 1.0 + (priority - 1) * 0.2
    
    # Exponential backoff for retries
    retry_multiplier = RATE_LIMIT_CONFIG['exponential_backoff_base'] ** (attempt - 1)
    
    delay = base_delay * priority_multiplier * retry_multiplier
    return min(delay, RATE_LIMIT_CONFIG['max_delay_between_requests'])

def fetch_trends_multi_region(use_multi_region=True, max_regions=None):
    """
    Fetch trends from multiple regions with intelligent rate limiting and prioritization.
    """
    if not use_multi_region:
        logger.info("Fetching trends from US region only...")
        update_rate_limit_state()
        trends = fetch_trends_with_retry(REGIONS['US'])
        if trends:
            for trend in trends:
                trend['regions'] = ['US']
                trend['region_count'] = 1
                trend['total_volume'] = trend.get('tweet_volume', 0)
        return trends or []
    
    logger.info("Fetching trends from multiple regions with intelligent rate limiting...")
    
    all_trends = []
    region_trends = {}
    successful_regions = 0
    failed_regions = 0
    
    # Sort regions by priority
    regions_to_fetch = sorted(REGIONS.items(), 
                            key=lambda x: RATE_LIMIT_CONFIG['region_priority'].get(x[0], 999))
    
    # Limit regions if specified
    if max_regions:
        regions_to_fetch = regions_to_fetch[:max_regions]
    
    for region_name, woeid in regions_to_fetch:
        # Check rate limits before each request
        if should_skip_region_due_to_rate_limit():
            logger.warning(f"Skipping {region_name} due to rate limiting")
            break
        
        try:
            logger.info(f"Fetching trends for {region_name} (WOEID: {woeid})")
            update_rate_limit_state()
            
            trends = fetch_trends_with_retry(woeid, region_name)
            if trends:
                region_trends[region_name] = trends
                for trend in trends:
                    trend['regions'] = [region_name]
                    all_trends.append(trend)
                successful_regions += 1
                logger.info(f"Successfully fetched {len(trends)} trends from {region_name}")
            else:
                failed_regions += 1
                logger.warning(f"No trends returned for {region_name}")
                
        except Exception as e:
            logger.error(f"Error fetching trends for {region_name}: {e}")
            failed_regions += 1
            continue
    
    logger.info(f"Region fetch summary: {successful_regions} successful, {failed_regions} failed")
    
    if not all_trends:
        logger.warning("No trends fetched from any region, falling back to US only")
        return fetch_trends_multi_region(use_multi_region=False)
    
    # Aggregate and deduplicate trends
    aggregated_trends = aggregate_trends(all_trends)
    
    logger.info(f"Fetched trends from {len(region_trends)} regions, aggregated to {len(aggregated_trends)} unique trends")
    return aggregated_trends


def fetch_trends_with_retry(woeid, region_name="Unknown", max_retries=3):
    """
    Fetch trends with intelligent retry logic and rate limiting.
    """
    for attempt in range(max_retries):
        try:
            # Get appropriate delay for this region and attempt
            delay = get_region_delay(region_name, attempt + 1)
            if attempt > 0:
                logger.info(f"Retry {attempt + 1}/{max_retries} for {region_name}, waiting {delay:.2f}s")
                time.sleep(delay)
            
            trends = fetch_trends(woeid)
            if trends:
                logger.info(f"Successfully fetched {len(trends)} trends for {region_name}")
                return trends
            else:
                logger.warning(f"No trends returned for {region_name} (attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {region_name}: {e}")
            
            # Handle rate limiting specifically
            if "429" in error_msg or "Too Many Requests" in error_msg:
                logger.warning(f"Rate limited for {region_name}, implementing longer backoff")
                backoff_time = RATE_LIMIT_CONFIG['exponential_backoff_base'] ** (attempt + 2) * 5
                logger.info(f"Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)
                # Reset rate limit counters since we're backing off
                rate_limit_state['requests_this_hour'] = max(0, rate_limit_state['requests_this_hour'] - 1)
                rate_limit_state['requests_this_15min'] = max(0, rate_limit_state['requests_this_15min'] - 1)
            elif attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for {region_name}")
    
    return []


def aggregate_trends(trends):
    """
    Aggregate trends by name, combining regions and volumes with volume estimation.
    """
    trend_dict = {}
    
    for trend in trends:
        name = trend['trend_name']
        if name not in trend_dict:
            trend_dict[name] = {
                'trend_name': name,
                'regions': set(),
                'total_volume': 0,
                'region_volumes': {},
                'first_seen': datetime.now(),
                'promoted_content': trend.get('promoted_content', None),
                'query': trend.get('query', name)
            }
        
        # Add region
        trend_dict[name]['regions'].add(trend['regions'][0])
        
        # Add volume
        volume = trend.get('tweet_volume', 0)
        if volume:
            trend_dict[name]['total_volume'] += volume
            trend_dict[name]['region_volumes'][trend['regions'][0]] = volume
    
    # Convert sets to lists for JSON serialization
    aggregated = []
    for trend_data in trend_dict.values():
        trend_data['regions'] = list(trend_data['regions'])
        trend_data['region_count'] = len(trend_data['regions'])
        aggregated.append(trend_data)
    
    # Estimate missing volumes if needed
    if CONFIG['enable_volume_fallback']:
        estimated_trends = estimate_missing_volumes(aggregated)
        return estimated_trends
    
    return aggregated


def estimate_missing_volumes(trends):
    """
    Estimate volume data for trends that don't have it using various heuristics.
    """
    # Count trends with and without volume data
    trends_with_volume = [t for t in trends if t.get('total_volume', 0) > 0]
    trends_without_volume = [t for t in trends if t.get('total_volume', 0) == 0]
    
    if not trends_without_volume:
        return trends
    
    logger.info(f"Estimating volumes for {len(trends_without_volume)} trends without volume data")
    
    # Calculate base volume from trends that have it
    if trends_with_volume:
        avg_volume = sum(t['total_volume'] for t in trends_with_volume) / len(trends_with_volume)
        min_volume = min(t['total_volume'] for t in trends_with_volume)
    else:
        # Fallback values if no volume data available
        avg_volume = 10000
        min_volume = 1000
    
    # Estimate volumes based on multiple factors
    for trend in trends_without_volume:
        estimated_volume = estimate_trend_volume(trend, avg_volume, min_volume)
        trend['total_volume'] = estimated_volume
        trend['volume_estimated'] = True  # Flag to indicate this is estimated
    
    logger.info(f"Volume estimation complete. Estimated volumes range from {min(t['total_volume'] for t in trends_without_volume):,} to {max(t['total_volume'] for t in trends_without_volume):,}")
    
    return trends


def estimate_trend_volume(trend, avg_volume, min_volume):
    """
    Estimate volume for a single trend using multiple heuristics.
    """
    base_volume = avg_volume
    multiplier = 1.0
    
    # Factor 1: Region count (more regions = higher volume)
    region_count = trend.get('region_count', 1)
    multiplier *= (0.5 + (region_count * 0.3))
    
    # Factor 2: Region importance (weighted regions = higher volume)
    region_weights = CONFIG['region_weights']
    if trend.get('regions'):
        avg_region_weight = sum(region_weights.get(region, 0.5) for region in trend['regions']) / len(trend['regions'])
        multiplier *= (0.7 + (avg_region_weight * 0.6))
    
    # Factor 3: Controversy potential
    controversy_keywords = ['breaking', 'exclusive', 'scandal', 'controversy', 'debate', 'protest', 'backlash']
    trend_lower = trend['trend_name'].lower()
    controversy_count = sum(1 for word in controversy_keywords if word in trend_lower)
    multiplier *= (1.0 + (controversy_count * 0.2))
    
    # Factor 4: Cross-theme appeal
    theme_matches = match_themes(trend['trend_name'])
    cross_theme_multiplier = 1.0 + (len(theme_matches) * 0.15)
    multiplier *= cross_theme_multiplier
    
    # Factor 5: Trend name characteristics
    name = trend['trend_name']
    if name.startswith('#'):  # Hashtags often have higher engagement
        multiplier *= 1.2
    if any(char in name for char in ['!', '?']):  # Excitement/controversy indicators
        multiplier *= 1.1
    if len(name.split()) > 2:  # Longer names might be more specific/engaging
        multiplier *= 1.05
    
    estimated_volume = int(base_volume * multiplier)
    
    # Ensure reasonable bounds
    estimated_volume = max(min_volume, min(estimated_volume, avg_volume * 3))
    
    return estimated_volume


def calculate_engagement_score(trend, all_trends):
    """
    Calculate engagement score based on multiple factors with fallback metrics.
    """
    scores = {}
    
    # 1. Cross-region presence (0-1)
    max_regions = max(len(REGIONS), 1)
    scores['cross_region_presence'] = trend['region_count'] / max_regions
    
    # 2. Volume score (0-1) - relative to other trends with fallback
    volumes = [t.get('total_volume', 0) for t in all_trends if t.get('total_volume', 0) > 0]
    if volumes and max(volumes) > 0:
        max_volume = max(volumes)
        trend_volume = trend.get('total_volume', 0)
        scores['volume_score'] = trend_volume / max_volume
    else:
        # Fallback: use region count as proxy for volume when volume data is missing
        if CONFIG['enable_volume_fallback']:
            max_region_count = max(t.get('region_count', 1) for t in all_trends)
            scores['volume_score'] = trend.get('region_count', 1) / max_region_count if max_region_count > 0 else 0
            logger.info(f"Using region count fallback for volume score: {scores['volume_score']:.3f}")
        else:
            scores['volume_score'] = 0
    
    # 3. Controversy score (0-1) - based on keywords that suggest debate
    controversy_keywords = ['debate', 'controversy', 'scandal', 'accusation', 'allegation', 'investigation', 'lawsuit', 'protest', 'boycott', 'backlash', 'outrage', 'fury', 'anger', 'dispute', 'conflict', 'war', 'battle', 'fight', 'clash', 'breaking', 'exclusive', 'leak', 'exposed', 'revealed']
    trend_lower = trend['trend_name'].lower()
    controversy_count = sum(1 for word in controversy_keywords if word in trend_lower)
    scores['controversy_score'] = min(controversy_count / 3, 1.0)  # Cap at 1.0
    
    # 4. Cross-theme appeal (0-1) - how many themes this trend could appeal to
    theme_matches = match_themes(trend['trend_name'])
    scores['cross_theme_appeal'] = len(theme_matches) / len(THEMES)
    
    # 5. Velocity score (0-1) - enhanced heuristic
    scores['velocity_score'] = (scores['volume_score'] + scores['cross_region_presence']) / 2
    
    # 6. Region weight score (0-1) - NEW: weighted by region importance
    region_weights = CONFIG['region_weights']
    if trend.get('regions'):
        avg_region_weight = sum(region_weights.get(region, 0.5) for region in trend['regions']) / len(trend['regions'])
        scores['region_weight_score'] = avg_region_weight
    else:
        scores['region_weight_score'] = 0.5  # Default weight
    
    # Calculate weighted total score
    total_score = sum(
        scores[factor] * ENGAGEMENT_WEIGHTS[factor]
        for factor in ENGAGEMENT_WEIGHTS.keys()
    )
    
    trend['engagement_score'] = total_score
    trend['engagement_breakdown'] = scores
    
    # Enhanced logging with fallback indicators
    volume_source = "API" if trend.get('total_volume', 0) > 0 else "fallback"
    logger.info(f"Engagement score for '{trend['trend_name']}': {total_score:.3f} (regions: {trend['region_count']}, volume: {trend.get('total_volume', 0)} [{volume_source}])")
    
    return total_score


def match_themes_with_engagement_boost(trend_name, engagement_score):
    """
    Enhanced theme matching that considers engagement potential.
    Returns (themes, confidence, engagement_boost).
    """
    themes, confidence = match_themes_with_confidence(trend_name)
    
    # Apply engagement boost to confidence
    engagement_boost = min(engagement_score * 0.2, 0.3)  # Max 30% boost
    boosted_confidence = min(confidence + engagement_boost, 1.0)
    
    logger.info(f"Engagement boost for '{trend_name}': {engagement_boost:.3f} (original: {confidence:.3f}, boosted: {boosted_confidence:.3f})")
    
    return themes, boosted_confidence, engagement_boost


def select_top_trends_by_theme_with_engagement(trends, max_per_theme=2):
    """
    Select top trends for each theme considering both relevance and engagement.
    """
    logger.info("Selecting top trends by theme with engagement scoring...")
    
    # Calculate engagement scores for all trends
    for trend in trends:
        calculate_engagement_score(trend, trends)
    
    # Filter trends by minimum engagement score
    min_engagement = CONFIG['min_engagement_score']
    filtered_trends = [t for t in trends if t.get('engagement_score', 0) >= min_engagement]
    
    if len(filtered_trends) < len(trends):
        logger.info(f"Filtered out {len(trends) - len(filtered_trends)} trends below engagement threshold {min_engagement}")
    
    # Group trends by theme with engagement consideration
    theme_trends = defaultdict(list)
    
    for trend in filtered_trends:
        themes, confidence, engagement_boost = match_themes_with_engagement_boost(
            trend['trend_name'], 
            trend['engagement_score']
        )
        
        for theme in themes:
            # Create trend entry with all relevant information
            trend_entry = {
                'trend': trend,
                'confidence': confidence,
                'engagement_score': trend['engagement_score'],
                'engagement_boost': engagement_boost,
                'combined_score': confidence + (trend['engagement_score'] * 0.3)  # Weight engagement
            }
            theme_trends[theme].append(trend_entry)
    
    # Select top trends for each theme
    top_trends = {}
    
    for theme, trend_list in theme_trends.items():
        if trend_list:
            # Sort by combined score (relevance + engagement)
            trend_list.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Take top trends up to max_per_theme
            top_trends[theme] = trend_list[:max_per_theme]
            
            logger.info(f"Selected {len(top_trends[theme])} trends for {theme}:")
            for i, entry in enumerate(top_trends[theme]):
                trend = entry['trend']
                logger.info(f"  {i+1}. {trend['trend_name']} (score: {entry['combined_score']:.3f}, regions: {trend['region_count']}, volume: {trend.get('total_volume', 0)})")
    
    return top_trends


def analyze_trend_performance(trends):
    """
    Analyze overall trend performance and provide insights.
    """
    if not trends:
        return {}
    
    analysis = {
        'total_trends': len(trends),
        'regions_covered': len(REGIONS),
        'avg_engagement_score': 0,
        'top_engaging_trends': [],
        'theme_distribution': defaultdict(int),
        'region_distribution': defaultdict(int),
        'volume_stats': {
            'min': float('inf'),
            'max': 0,
            'avg': 0
        }
    }
    
    total_engagement = 0
    volumes = []
    
    for trend in trends:
        # Engagement
        engagement = trend.get('engagement_score', 0)
        total_engagement += engagement
        
        # Volume stats
        volume = trend.get('total_volume', 0)
        if volume > 0:
            volumes.append(volume)
            analysis['volume_stats']['min'] = min(analysis['volume_stats']['min'], volume)
            analysis['volume_stats']['max'] = max(analysis['volume_stats']['max'], volume)
        
        # Theme distribution
        themes = match_themes(trend['trend_name'])
        for theme in themes:
            analysis['theme_distribution'][theme] += 1
        
        # Region distribution
        for region in trend.get('regions', []):
            analysis['region_distribution'][region] += 1
    
    # Calculate averages
    if trends:
        analysis['avg_engagement_score'] = total_engagement / len(trends)
    
    if volumes:
        analysis['volume_stats']['avg'] = sum(volumes) / len(volumes)
    else:
        analysis['volume_stats']['min'] = 0
    
    # Get top engaging trends
    sorted_trends = sorted(trends, key=lambda x: x.get('engagement_score', 0), reverse=True)
    analysis['top_engaging_trends'] = [
        {
            'name': t['trend_name'],
            'engagement': t.get('engagement_score', 0),
            'regions': t.get('region_count', 0),
            'volume': t.get('total_volume', 0)
        }
        for t in sorted_trends[:5]
    ]
    
    return analysis


def print_phase1_summary(trends):
    """
    Print a comprehensive summary of Phase 1 implementation results.
    """
    analysis = analyze_trend_performance(trends)
    
    # Calculate volume estimation stats
    trends_with_volume = [t for t in trends if t.get('total_volume', 0) > 0 and not t.get('volume_estimated', False)]
    trends_with_estimated_volume = [t for t in trends if t.get('volume_estimated', False)]
    trends_without_volume = [t for t in trends if t.get('total_volume', 0) == 0 and not t.get('volume_estimated', False)]
    
    print("\n" + "="*80)
    print("🎯 PHASE 1 IMPLEMENTATION SUMMARY")
    print("="*80)
    print(f"📊 Total Trends Analyzed: {analysis['total_trends']}")
    print(f"🌍 Regions Covered: {analysis['regions_covered']}")
    print(f"📈 Average Engagement Score: {analysis['avg_engagement_score']:.3f}")
    
    print(f"\n📊 Volume Statistics:")
    print(f"   Min: {analysis['volume_stats']['min']:,}")
    print(f"   Max: {analysis['volume_stats']['max']:,}")
    print(f"   Avg: {analysis['volume_stats']['avg']:.0f}")
    
    print(f"\n📈 Volume Data Quality:")
    print(f"   API Volume Data: {len(trends_with_volume)} trends")
    print(f"   Estimated Volumes: {len(trends_with_estimated_volume)} trends")
    print(f"   No Volume Data: {len(trends_without_volume)} trends")
    
    if trends_with_estimated_volume:
        estimated_range = f"{min(t['total_volume'] for t in trends_with_estimated_volume):,} - {max(t['total_volume'] for t in trends_with_estimated_volume):,}"
        print(f"   Estimated Range: {estimated_range}")
    
    print(f"\n🏆 Top 5 Most Engaging Trends:")
    for i, trend in enumerate(analysis['top_engaging_trends'], 1):
        volume_source = "EST" if any(t.get('volume_estimated', False) for t in trends if t['trend_name'] == trend['name']) else "API"
        print(f"   {i}. {trend['name']} (Score: {trend['engagement']:.3f}, Regions: {trend['regions']}, Volume: {trend['volume']:,} [{volume_source}])")
    
    print(f"\n🎨 Theme Distribution:")
    for theme, count in analysis['theme_distribution'].items():
        percentage = (count / analysis['total_trends']) * 100
        print(f"   {theme}: {count} trends ({percentage:.1f}%)")
    
    print(f"\n🌍 Region Distribution (Top 10):")
    sorted_regions = sorted(analysis['region_distribution'].items(), key=lambda x: x[1], reverse=True)
    for region, count in sorted_regions[:10]:
        percentage = (count / analysis['total_trends']) * 100
        print(f"   {region}: {count} trends ({percentage:.1f}%)")
    
    print(f"\n⚙️  System Configuration:")
    print(f"   Multi-region: {CONFIG['use_multi_region']}")
    print(f"   Max regions: {CONFIG['max_regions'] or 'All'}")
    print(f"   Volume fallback: {CONFIG['enable_volume_fallback']}")
    print(f"   Min engagement threshold: {CONFIG['min_engagement_score']}")
    
    print("="*80)


def direct_keyword_match(trend_name):
    matches = []
    name = trend_name.lower()
    for theme, keywords in THEMES_EXPANDED.items():
        if any(kw in name for kw in keywords):
            matches.append(theme)
    if matches:
        logger.info(f"Direct keyword match for '{trend_name}': {matches}")
    return matches


def fuzzy_match(trend_name, threshold=82):
    matches = []
    name = trend_name.lower()
    for theme, keywords in THEMES_EXPANDED.items():
        for kw in keywords:
            score = fuzz.partial_ratio(kw, name)
            if score >= threshold:
                matches.append(theme)
                break
    if matches:
        logger.info(f"Fuzzy match for '{trend_name}': {matches}")
    return matches


def semantic_similarity_match(trend_name, threshold=0.45):
    matches = []
    trend_emb = model.encode(trend_name)
    for theme, emb in theme_embeddings.items():
        score = util.cos_sim(trend_emb, emb).item()
        if score > threshold:
            matches.append(theme)
    if matches:
        logger.info(f"Semantic similarity match for '{trend_name}': {matches}")
    return matches


def ner_match(trend_name):
    matches = []
    doc = nlp(trend_name)
    for ent in doc.ents:
        ent_text = ent.text.lower()
        for theme, keywords in THEMES_EXPANDED.items():
            if any(kw in ent_text for kw in keywords):
                matches.append(theme)
    if matches:
        logger.info(f"NER match for '{trend_name}': {matches}")
    return matches


def manual_override(trend_name):
    # Manual overrides for known good matches
    for key, theme in MANUAL_OVERRIDES.items():
        if key.lower() in trend_name.lower():
            logger.info(f"Manual override for '{trend_name}': {theme}")
            return [theme]
    
    # Manual exclusions for known bad matches
    political_figures = ['bondi', 'bongino', 'trump', 'biden', 'harris', 'pence', 'clinton', 'obama']
    if any(politician in trend_name.lower() for politician in political_figures):
        logger.info(f"Manual exclusion for political figure '{trend_name}'")
        return []
    
    return []


def match_themes(trend_name):
    # 1. Manual override
    manual = manual_override(trend_name)
    if manual:
        return manual
    # 2. Direct keyword match
    direct = direct_keyword_match(trend_name)
    if direct:
        return direct
    # 3. Fuzzy match
    fuzzy = fuzzy_match(trend_name)
    if fuzzy:
        return fuzzy
    # 4. Synonym expansion is included in THEMES_EXPANDED
    # 5. Semantic similarity
    semantic = semantic_similarity_match(trend_name)
    if semantic:
        return semantic
    # 6. NER
    ner = ner_match(trend_name)
    if ner:
        return ner
    return []


def match_themes_with_confidence(trend_name):
    # 1. Manual override
    manual = manual_override(trend_name)
    if manual:
        logger.info(f"Manual override for '{trend_name}': {manual} (confidence {LAYER_CONFIDENCE['manual']})")
        return manual, LAYER_CONFIDENCE['manual']
    # 2. Direct keyword match
    direct = direct_keyword_match(trend_name)
    if direct:
        logger.info(f"Direct keyword match for '{trend_name}': {direct} (confidence {LAYER_CONFIDENCE['direct']})")
        return direct, LAYER_CONFIDENCE['direct']
    # 3. Fuzzy match
    fuzzy = fuzzy_match(trend_name)
    if fuzzy:
        logger.info(f"Fuzzy match for '{trend_name}': {fuzzy} (confidence {LAYER_CONFIDENCE['fuzzy']})")
        return fuzzy, LAYER_CONFIDENCE['fuzzy']
    # 4. Semantic similarity
    semantic = semantic_similarity_match(trend_name)
    if semantic:
        logger.info(f"Semantic similarity match for '{trend_name}': {semantic} (confidence {LAYER_CONFIDENCE['semantic']})")
        return semantic, LAYER_CONFIDENCE['semantic']
    # 5. NER
    ner = ner_match(trend_name)
    if ner:
        logger.info(f"NER match for '{trend_name}': {ner} (confidence {LAYER_CONFIDENCE['ner']})")
        return ner, LAYER_CONFIDENCE['ner']
    return [], 0.0


def deduplicate_tweets(tweets):
    seen = set()
    deduped = []
    for tweet in tweets:
        text = tweet.get('text', '').strip()
        if text and text not in seen:
            deduped.append(tweet)
            seen.add(text)
    return deduped


THEME_CONTEXT = {
    'AI': [
        'ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'automation', 'neural network', 'deep learning', 'algorithm', 'data science', 'robotics', 'intelligence', 'smart', 'predictive', 'analytics', 'model', 'training', 'dataset', 'neural', 'automated', 'intelligent', 'cognitive', 'learning', 'prediction', 'analysis'
    ],
    'Sports': [
        'soccer', 'football', 'basketball', 'nba', 'nfl', 'fifa', 'premier league', 'champions league', 'world cup', 'player', 'team', 'match', 'game', 'goal', 'score', 'transfer', 'coach', 'tactics', 'league', 'cup', 'tournament', 'championship', 'playoff', 'draft', 'trade', 'athlete', 'sport', 'win', 'lose', 'draw', 'season', 'title', 'stadium', 'fans', 'supporters', 'club', 'manager', 'formation', 'strategy', 'point', 'assist', 'rebound', 'touchdown', 'field goal', 'quarterback', 'running back', 'defense', 'offense'
    ],
    'Technology': [
        'tech', 'technology', 'software', 'hardware', 'startup', 'innovation', 'digital', 'app', 'platform', 'coding', 'programming', 'development', 'cybersecurity', 'cloud', 'blockchain', 'vr', 'ar', 'iot', 'mobile', 'web', 'api', 'database', 'code', 'developer', 'engineer', 'programmer', 'hacker', 'bug', 'feature', 'update', 'release', 'version', 'beta', 'alpha', 'prototype', 'product', 'service', 'company', 'funding', 'investment', 'venture', 'capital'
    ],
    'Gaming': [
        'gaming', 'game', 'esports', 'streaming', 'twitch', 'youtube', 'console', 'pc', 'mobile game', 'rpg', 'fps', 'battle royale', 'mmo', 'indie game', 'game development', 'gamer', 'playstation', 'xbox', 'nintendo', 'steam', 'discord', 'server', 'lobby', 'match', 'round', 'level', 'quest', 'mission', 'character', 'avatar', 'skin', 'weapon', 'item', 'inventory', 'skill', 'ability', 'rank', 'leaderboard', 'tournament', 'championship', 'streamer', 'content creator', 'mod', 'dlc', 'expansion'
    ]
}


def theme_context_check(text, theme):
    """Return True if any theme-specific context word is in the text."""
    text_l = text.lower()
    for word in THEME_CONTEXT[theme]:
        if word in text_l:
            return True
    return False


def ai_validate_trend_relevance(trend_name, theme, tweets, summary, backstory):
    """
    Use AI to validate if a trend is actually relevant to the theme.
    Returns True if relevant, False if not.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("No OpenAI API key found, defaulting to True")
        return True
    
    client = OpenAI(api_key=api_key)
    
    # Sample a few tweets for context
    sample_tweets = [t['text'] for t in tweets[:3]]
    tweet_context = "\n".join([f"- {tweet}" for tweet in sample_tweets])
    
    prompt = f"""
    You are a content strategist evaluating if a trending topic is relevant for content creation.
    
    Trend: {trend_name}
    Theme: {theme}
    Summary: {summary}
    Backstory: {backstory}
    Sample tweets:
    {tweet_context}
    
    Question: Is this trend genuinely relevant to the {theme} theme for creating engaging content?
    
    Consider:
    1. Is there a natural, authentic connection to {theme}?
    2. Can you create compelling content about this trend from a {theme} perspective?
    3. Would {theme} enthusiasts find this content valuable and interesting?
    4. Is this a forced connection or a genuine opportunity?
    
    Respond with ONLY "YES" or "NO" and a brief reason.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip().lower()
        logger.info(f"AI validation for '{trend_name}' ({theme}): {result}")
        
        return result.startswith('yes')
        
    except Exception as e:
        logger.error(f"Error in AI validation: {e}")
        return True  # Default to True if AI fails


def validate_trend_with_tweets(trend_name, theme, max_results=10, min_ratio=0.2):
    tweets = fetch_top_tweets_for_trend(trend_name, max_results)
    tweets = deduplicate_tweets(tweets)
    
    # Get summary and backstory for AI validation
    summary, backstory = per_trend_summary_with_backstory(trend_name, tweets)
    
    # Let AI decide if this trend is relevant
    if not ai_validate_trend_relevance(trend_name, theme, tweets, summary, backstory):
        logger.info(f"AI rejected trend '{trend_name}' for theme '{theme}': not genuinely relevant")
        return False
    
    # Basic validation as backup
    match_count = 0
    context_match_count = 0
    
    for tweet in tweets:
        tweet_text = tweet['text']
        tweet_themes, conf = match_themes_with_confidence(tweet_text)
        if theme in tweet_themes:
            match_count += 1
        if theme_context_check(tweet_text, theme):
            context_match_count += 1
    
    ratio = match_count / max(1, len(tweets))
    logger.info(f"Basic validation for trend '{trend_name}' and theme '{theme}': {match_count}/{len(tweets)} tweets matched ({ratio:.2f}), {context_match_count} context matches")
    
    # Simple threshold check as backup
    if ratio < min_ratio and context_match_count == 0:
        logger.info(f"Trend '{trend_name}' failed basic validation backup check")
        return False
    
    return True


def fetch_trends(woeid=23424977):
    """
    Fetch Twitter trends with enhanced volume data handling and proper response parsing.
    """
    client = get_twitter_client()
    logger.info(f"Fetching Twitter trends for WOEID {woeid}...")
    
    try:
        response = client.get_trends(woeid)
        logger.info(f"Raw trends API response: {response}")
        
        # Handle different response formats and extract trends properly
        trends = []
        if isinstance(response, dict):
            # Handle dict response format
            if 'data' in response:
                trends = response['data']
            elif 'trends' in response:
                trends = response['trends']
        elif hasattr(response, 'data'):
            # Handle object with data attribute
            trends = response.data
        elif hasattr(response, 'trends'):
            # Handle object with trends attribute
            trends = response.trends
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            trends = []
        
        # Enhanced trend processing with proper volume data extraction
        processed_trends = []
        for trend in trends:
            if isinstance(trend, dict):
                # Extract trend name from various possible fields
                trend_name = (trend.get('name') or 
                            trend.get('trend_name') or 
                            trend.get('query', ''))
                
                # Extract volume from various possible fields
                tweet_volume = (trend.get('tweet_volume') or 
                              trend.get('tweet_count') or 
                              trend.get('volume', 0))
                
                processed_trend = {
                    'trend_name': trend_name,
                    'tweet_volume': tweet_volume,
                    'query': trend.get('query', trend_name),
                    'promoted_content': trend.get('promoted_content'),
                    'woeid': woeid
                }
            else:
                # Handle object format
                trend_name = (getattr(trend, 'name', '') or 
                            getattr(trend, 'trend_name', '') or 
                            getattr(trend, 'query', ''))
                
                tweet_volume = (getattr(trend, 'tweet_volume', 0) or 
                              getattr(trend, 'tweet_count', 0) or 
                              getattr(trend, 'volume', 0))
                
                processed_trend = {
                    'trend_name': trend_name,
                    'tweet_volume': tweet_volume,
                    'query': getattr(trend, 'query', trend_name),
                    'promoted_content': getattr(trend, 'promoted_content', None),
                    'woeid': woeid
                }
            
            # Skip empty trend names
            if processed_trend['trend_name']:
                processed_trends.append(processed_trend)
        
        # Log detailed volume data statistics
        trends_with_volume = sum(1 for t in processed_trends if t.get('tweet_volume', 0) > 0)
        total_volume = sum(t.get('tweet_volume', 0) for t in processed_trends)
        
        logger.info(f"Processed {len(processed_trends)} trends")
        logger.info(f"Volume data available for {trends_with_volume}/{len(processed_trends)} trends")
        if trends_with_volume > 0:
            avg_volume = total_volume / trends_with_volume
            logger.info(f"Average volume: {avg_volume:,.0f}, Total volume: {total_volume:,.0f}")
        
        return processed_trends
        
    except Exception as e:
        logger.error(f"Failed to fetch trends for WOEID {woeid}: {e}")
        return []


def find_top_trends_by_theme(trends):
    theme_trends = {theme: [] for theme in THEMES}
    for trend in trends:
        name = trend.get('name') or trend.get('trend_name')
        if not name:
            continue
        themes, confidence = match_themes_with_confidence(name)
        for theme in themes:
            # If ambiguous or low confidence, require tweet validation
            is_ambiguous = any(term.lower() in name.lower() for term in AMBIGUOUS_TERMS)
            if confidence < 0.8 or is_ambiguous:
                logger.info(f"Trend '{name}' matched theme '{theme}' with confidence {confidence} (ambiguous: {is_ambiguous}), validating with tweets...")
                if not validate_trend_with_tweets(name, theme):
                    logger.info(f"Trend '{name}' rejected for theme '{theme}' after tweet validation.")
                    continue
            theme_trends[theme].append({
                'name': name,
                'tweet_volume': trend.get('tweet_volume') or trend.get('tweet_count', 0)
            })
    for theme in theme_trends:
        theme_trends[theme].sort(key=lambda t: t['tweet_volume'] or 0, reverse=True)
        logger.info(f"Theme '{theme}' has {len(theme_trends[theme])} matching trends.")
    return theme_trends


def fetch_top_tweets_for_trend(trend_name, max_results=3):
    client = get_client()
    api_max = max(10, min(100, max_results))  # Ensure within allowed range
    logger.info(f"Searching for top {max_results} tweets for trend: '{trend_name}' (API will fetch {api_max})...")
    try:
        query = trend_name
        tweets = client.search_recent_tweets(query=query, max_results=api_max, tweet_fields=['public_metrics', 'author_id'])
        logger.info(f"Raw tweets API response for '{trend_name}': {tweets}")
        if isinstance(tweets, dict):
            tweet_data = tweets.get('data', [])
        elif hasattr(tweets, 'data'):
            tweet_data = tweets.data
        else:
            tweet_data = []
        logger.info(f"Found {len(tweet_data)} tweets for trend '{trend_name}'.")
        return tweet_data[:max_results]  # Return only the top N requested
    except Exception as e:
        logger.error(f"Failed to fetch tweets for trend '{trend_name}': {e}")
        return []


def fetch_long_tweets_for_trend(trend_name, min_length=200, max_results=10):
    """Fetch longer-format tweets for a given trend name using Twitter API v2 search."""
    client = get_client()
    api_max = max(10, min(100, max_results))
    logger.info(f"Searching for long tweets for trend: '{trend_name}' (min_length={min_length}, API will fetch {api_max})...")
    try:
        query = f"{trend_name} lang:en -is:retweet"  # English, no retweets
        tweets = client.search_recent_tweets(query=query, max_results=api_max, tweet_fields=['public_metrics', 'author_id', 'text'])
        if isinstance(tweets, dict):
            tweet_data = tweets.get('data', [])
        elif hasattr(tweets, 'data'):
            tweet_data = tweets.data
        else:
            tweet_data = []
        # Filter for longer tweets
        long_tweets = [t for t in tweet_data if len(t.get('text', '')) >= min_length]
        logger.info(f"Found {len(long_tweets)} long tweets for trend '{trend_name}'.")
        return long_tweets
    except Exception as e:
        logger.error(f"Failed to fetch long tweets for trend '{trend_name}': {e}")
        return []


def summarize_tweets_with_openai(trend, tweets, purpose="conversation"):
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    if not tweets:
        return "No relevant tweets found."
    prompt = f"Trend: {trend}\nSample tweets for {purpose}:\n"
    for tweet in tweets[:5]:
        prompt += f"- {tweet.get('text', '[No text]')}\n"
    if purpose == "conversation":
        prompt += "\nPlease summarize the main conversation and sentiment about this trend."
    else:
        prompt += "\nPlease provide a concise backstory or context for why this is trending. If unclear, make an educated guess based on the tweets and your knowledge."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You are a social media analyst."
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def per_trend_summary_with_backstory(trend, tweets):
    # Summarize conversation
    summary = summarize_tweets_with_openai(trend, tweets, purpose="conversation")
    # Fetch long tweets for backstory
    long_tweets = fetch_long_tweets_for_trend(trend, min_length=200, max_results=15)
    backstory = summarize_tweets_with_openai(trend, long_tweets, purpose="backstory")
    # If backstory is vague, try again with a refined search (e.g., add 'explained' or 'news')
    vague_phrases = ["unclear", "not sure", "unknown", "no information", "no context", "cannot determine", "guess"]
    if any(phrase in backstory.lower() for phrase in vague_phrases):
        logger.info(f"Backstory for trend '{trend}' was unclear, refining search...")
        refined_long_tweets = fetch_long_tweets_for_trend(trend + " explained OR news", min_length=180, max_results=15)
        if refined_long_tweets:
            backstory = summarize_tweets_with_openai(trend, refined_long_tweets, purpose="backstory")
    return summary, backstory


def summarize_trends_with_openai(trend_summaries):
    """
    Given a list of (trend, [sample_tweets]), return an overall summary using OpenAI.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    prompt = "Here are some current Twitter trends and sample tweets:\n"
    for trend, tweets in trend_summaries:
        prompt += f"\nTrend: {trend}\nSample tweets:\n"
        for tweet in tweets[:3]:  # Limit to 3 tweets per trend for brevity
            prompt += f"- {tweet}\n"
    prompt += "\nPlease provide a concise summary of the overall themes, public sentiment, and any notable patterns you observe."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You are a social media analyst."
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def generate_content_angles(summary, backstory, theme, trend_name, n_angles=4):
    """
    Use OpenAI to brainstorm 3-5 unique, polarizing, and thought-provoking content angles for a trend.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    prompt = f'''
You're a {theme} expert with a bold, modern personality. Create {n_angles} unique, polarizing content angles for the trend "{trend_name}".

Context: {summary}
Backstory: {backstory}

Each angle should be:
- Controversial or contrarian (but not offensive)
- Forward-thinking or predictive
- Designed to spark heated discussion
- Relevant to {theme} enthusiasts
- Under 100 characters
- Casual, modern tone (like a confident friend sharing a hot take)

Examples of good angles:
- "Everyone's wrong about this trend"
- "This is actually a game-changer for [theme]"
- "Why this will fail spectacularly"
- "The hidden opportunity nobody sees"
- "This changes everything for [theme]"
- "Why the hype is actually justified"

Format: Return only the angles, one per line, no numbering.
'''
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "You are a creative strategist."
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.9,
        max_tokens=400
    )
    text = response.choices[0].message.content.strip()
    # Parse numbered list
    angles = []
    for line in text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and (line[1] == '.' or line[1] == ')')):
            angle = line[2:].strip()
            if angle:
                angles.append(angle)
        elif line and not angles and len(line) > 10:
            angles.append(line)
    if not angles:
        # fallback: split by lines
        angles = [l.strip('- ').strip() for l in text.split('\n') if l.strip()]
    return angles[:n_angles]


def ai_validate_content_quality(content, angle, trend_name, theme):
    """
    Use AI to validate if the generated content is high quality and relevant.
    Returns True if good, False if needs improvement.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return True
    
    client = OpenAI(api_key=api_key)
    
    tweets_text = "\n".join([f"Tweet {i+1}: {tweet}" for i, tweet in enumerate(content['tweets'])])
    
    prompt = f"""
    You are a content quality reviewer evaluating a social media post.
    
    Trend: {trend_name}
    Theme: {theme}
    Angle: {angle}
    Generated Content:
    {tweets_text}
    
    Evaluate this content on:
    1. Relevance: Does it genuinely connect to the {theme} theme?
    2. Authenticity: Does it sound natural, not forced?
    3. Engagement: Would it spark conversation?
    4. Quality: Is it well-written and compelling?
    
    Respond with ONLY "GOOD" or "NEEDS_IMPROVEMENT" and a brief reason.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip().lower()
        logger.info(f"Content quality validation: {result}")
        
        return result.startswith('good')
        
    except Exception as e:
        logger.error(f"Error in content validation: {e}")
        return True


def draft_content(angle, summary, trend_name, format="thread", max_attempts=3):
    """
    Use OpenAI to draft a Twitter thread or single tweet based on the chosen angle.
    Includes self-validation and retry logic.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_attempts):
        if format == "thread":
            prompt = f'''
Write a 3-part Twitter thread about "{trend_name}" from this angle: "{angle}"

Context: {summary}

Style guidelines:
- Sound like a confident, knowledgeable friend sharing a hot take
- Use casual, modern language (but not unprofessional)
- Include emojis strategically (1-2 per tweet max)
- Make it punchy and memorable
- Each tweet should be under 280 characters
- Tweet 1: Hook with controversy or bold statement
- Tweet 2: Back it up with reasoning or evidence
- Tweet 3: Call to action or prediction

IMPORTANT: Only write about the actual topic. Don't force connections to unrelated themes. If the trend is about politics, sports, or entertainment, write about that authentically.

Tone: Confident, slightly controversial, but not mean-spirited. Like someone who knows their stuff and isn't afraid to say it.

Format: Return exactly 3 tweets, one per line, no numbering.
'''
        else:
            prompt = f'''
Write a single, punchy tweet about "{trend_name}" from this angle: "{angle}"

Context: {summary}

Style guidelines:
- Sound like a confident friend sharing a hot take
- Use casual, modern language
- Include 1-2 strategic emojis
- Make it memorable and shareable
- Under 280 characters
- Controversial but not offensive

IMPORTANT: Only write about the actual topic. Don't force connections to unrelated themes.

Tone: Confident, slightly controversial, but not mean-spirited.
'''
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a social media copywriter."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.8,
            max_tokens=400
        )
        
        text = response.choices[0].message.content.strip()
        
        if format == "thread":
            # Parse numbered list
            tweets = []
            for line in text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() and (line[1] == '.' or line[1] == ')')):
                    tweet = line[2:].strip()
                    if tweet:
                        tweets.append(tweet)
                elif line and not tweets and len(line) > 10:
                    tweets.append(line)
            if not tweets:
                tweets = [l.strip('- ').strip() for l in text.split('\n') if l.strip()]
            
            content = {'type': 'thread', 'tweets': tweets[:3]}
        else:
            # Single tweet
            tweet = text.strip()
            if len(tweet) > 280:
                tweet = tweet[:277] + '...'
            content = {'type': 'single_tweet', 'tweets': [tweet]}
        
        # Validate content quality
        if ai_validate_content_quality(content, angle, trend_name, "AI onboarding" if "ai" in trend_name.lower() else "Soccer" if any(sport in trend_name.lower() for sport in ['soccer', 'football', 'match']) else "Life improvement"):
            logger.info(f"Content validation passed on attempt {attempt + 1}")
            return content
        else:
            logger.info(f"Content validation failed on attempt {attempt + 1}, retrying...")
            if attempt < max_attempts - 1:
                # Add feedback for next attempt
                prompt += f"\n\nPrevious attempt was rejected. Make it more authentic and relevant to the actual topic."
    
    # If all attempts failed, return the last attempt
    logger.warning(f"All {max_attempts} content generation attempts failed, returning last attempt")
    return content


def create_visual_asset(angle, trend_name, theme):
    """
    Use OpenAI to generate a DALL-E 3 image for the content angle.
    Step 1: Generate a metaphorical, descriptive image prompt.
    Step 2: Generate the image and return the URL.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    # Step 1: Generate image prompt
    prompt_gen = f'''
Trend: {trend_name}
Theme: {theme}
Angle: {angle}

Write a highly descriptive, metaphorical prompt for an image generator (like DALL-E 3). Translate the conceptual angle into a visual scene. Specify a style (e.g., 'photorealistic', 'cinematic lighting').

Format:
Image prompt: ...
'''
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "You are a creative visual director."
        }, {
            "role": "user",
            "content": prompt_gen
        }],
        temperature=0.85,
        max_tokens=200
    )
    text = response.choices[0].message.content.strip()
    # Extract the image prompt
    image_prompt = text
    if 'Image prompt:' in text:
        image_prompt = text.split('Image prompt:')[-1].strip()
    # Step 2: Generate image
    try:
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            n=1,
            size="1024x1024",
            quality="standard"
        )
        image_url = image_response.data[0].url
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        image_url = None
    return image_url


def select_best_angle(angles, summary, backstory, theme, trend_name):
    """
    Use OpenAI to select the most relevant and engaging angle from the list.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("No OpenAI API key found, using first angle")
        return angles[0]
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Given these content angles for trend '{trend_name}' (theme: {theme}):
    
    {chr(10).join(f"{i+1}. {angle}" for i, angle in enumerate(angles))}
    
    Context: {summary}
    Backstory: {backstory}
    
    Select the angle that is:
    1. Most relevant to the current conversation and trend
    2. Most likely to spark engagement and discussion
    3. Best aligned with a {theme} brand/personality
    4. Most unique and thought-provoking
    5. Most likely to generate shares and comments
    
    Return ONLY the selected angle text (not the number). If you can't decide, return the first angle.
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
            # If AI returned something not in our list, find the closest match
            for angle in angles:
                if fuzz.ratio(selected_angle.lower(), angle.lower()) > 80:
                    logger.info(f"Using closest match: {angle}")
                    return angle
            
            # Fallback to first angle
            logger.warning("AI selection not found in angles list, using first angle")
            return angles[0]
            
    except Exception as e:
        logger.error(f"Error selecting best angle: {e}")
        return angles[0]


def main():
    """
    Main workflow: identify trends, generate content angles, draft content, and create visual assets.
    Enhanced with multi-region trend fetching and engagement-based selection.
    """
    logger.info("Starting enhanced trend analysis and content generation with multi-region support...")
    
    # System health check
    system_health = check_system_health()
    if not system_health['healthy']:
        logger.warning(f"System health issues detected: {system_health['issues']}")
    
    # Reset rate limiting state at start
    global rate_limit_state
    rate_limit_state = {
        'last_request_time': 0,
        'requests_this_hour': 0,
        'requests_this_15min': 0,
        'hour_start': time.time(),
        'minute_start': time.time()
    }
    
    # Fetch trends from multiple regions with engagement scoring
    # Start conservatively with fewer regions to avoid rate limiting
    initial_max_regions = min(CONFIG['max_regions'] or 5, 5)  # Start with max 5 regions
    
    logger.info(f"Starting with {initial_max_regions} regions to avoid rate limiting")
    log_rate_limit_status()
    
    trends = fetch_trends_multi_region(
        use_multi_region=CONFIG['use_multi_region'],
        max_regions=initial_max_regions
    )
    
    # Log final rate limit status
    log_rate_limit_status()
    
    if not trends:
        logger.warning("No trends found from initial regions, trying US only")
        trends = fetch_trends_multi_region(use_multi_region=False)
        if not trends:
            logger.error("No trends found even from US region")
            return []
    
    # Print Phase 1 summary
    print_phase1_summary(trends)
    
    # Select top trends by theme with engagement consideration
    top_trends_by_theme = select_top_trends_by_theme_with_engagement(
        trends, 
        max_per_theme=CONFIG['max_trends_per_theme']
    )


def check_system_health():
    """
    Check system health and configuration validity.
    """
    health = {
        'healthy': True,
        'issues': [],
        'warnings': []
    }
    
    # Check required environment variables
    required_env_vars = ['OPENAI_API_KEY']
    for var in required_env_vars:
        if not os.getenv(var):
            health['issues'].append(f"Missing environment variable: {var}")
            health['healthy'] = False
    
    # Check Twitter API access
    try:
        client = get_twitter_client()
        # Try a simple API call to test connectivity
        test_response = client.get_trends(23424977)  # US WOEID
        if not test_response:
            health['warnings'].append("Twitter API returned empty response")
    except Exception as e:
        health['issues'].append(f"Twitter API connectivity issue: {e}")
        health['healthy'] = False
    
    # Check configuration validity
    if CONFIG['min_engagement_score'] < 0 or CONFIG['min_engagement_score'] > 1:
        health['issues'].append("Invalid min_engagement_score (must be 0-1)")
        health['healthy'] = False
    
    if CONFIG['max_regions'] is not None and CONFIG['max_regions'] <= 0:
        health['issues'].append("Invalid max_regions (must be > 0)")
        health['healthy'] = False
    
    # Check region weights
    for region in REGIONS.keys():
        if region not in CONFIG['region_weights']:
            health['warnings'].append(f"Missing region weight for: {region}")
    
    # Log health status
    if health['healthy']:
        logger.info("System health check passed")
    else:
        logger.error(f"System health check failed: {health['issues']}")
    
    if health['warnings']:
        logger.warning(f"System health warnings: {health['warnings']}")
    
    return health
    
    if not top_trends_by_theme:
        logger.warning("No relevant trends found for any theme")
        return []
    
    # Generate content packages for each top trend
    content_packages = []
    
    for theme, trend_entries in top_trends_by_theme.items():
        logger.info(f"\n=== Processing {theme} trends ===")
        
        for entry in trend_entries:
            trend = entry['trend']
            confidence = entry['confidence']
            engagement_score = entry['engagement_score']
            combined_score = entry['combined_score']
            
            logger.info(f"\n--- Processing trend: {trend['trend_name']} ---")
            logger.info(f"Confidence: {confidence:.3f}, Engagement: {engagement_score:.3f}, Combined: {combined_score:.3f}")
            logger.info(f"Regions: {trend['region_count']}, Total Volume: {trend.get('total_volume', 0)}")
            
            try:
                # Get summary and backstory
                summary, backstory = per_trend_summary_with_backstory(trend['trend_name'], fetch_top_tweets_for_trend(trend['trend_name'], max_results=3))
                
                # Generate content angles
                angles = generate_content_angles(summary, backstory, theme, trend['trend_name'])
                if not angles:
                    logger.warning(f"No angles generated for {trend['trend_name']}")
                    continue
                
                # Select the best angle using AI
                chosen_angle = select_best_angle(angles, summary, backstory, theme, trend['trend_name'])
                
                # Draft content
                content = draft_content(chosen_angle, summary, trend['trend_name'])
                if not content:
                    logger.warning(f"No content drafted for {trend['trend_name']}")
                    continue
                
                # Create visual asset
                visual_url = create_visual_asset(chosen_angle, trend['trend_name'], theme)
                
                # Create enhanced content package
                content_package = {
                    'trend_name': trend['trend_name'],
                    'theme': theme,
                    'summary': summary,
                    'backstory': backstory,
                    'all_angles': angles,
                    'chosen_angle': chosen_angle,
                    'content': content,
                    'visual_url': visual_url,
                    'trend_volume': trend.get('total_volume', 0),
                    'region_count': trend['region_count'],
                    'regions': trend['regions'],
                    'engagement_score': engagement_score,
                    'confidence_score': confidence,
                    'combined_score': combined_score,
                    'engagement_breakdown': trend.get('engagement_breakdown', {})
                }
                
                content_packages.append(content_package)
                
                # Print enhanced results
                print(f"\n🎯 ENHANCED CONTENT PACKAGE: {trend['trend_name']} ({theme})")
                print(f"📊 Engagement Score: {engagement_score:.3f}")
                print(f"🎯 Combined Score: {combined_score:.3f}")
                print(f"🌍 Regions: {trend['region_count']} ({', '.join(trend['regions'])})")
                print(f"📈 Total Volume: {trend.get('total_volume', 'N/A')}")
                print(f"📝 Summary: {summary}")
                print(f"📖 Backstory: {backstory}")
                print(f"💡 Chosen Angle: {chosen_angle}")
                print(f"📱 Content:")
                for i, tweet in enumerate(content['tweets'], 1):
                    print(f"   {i}. {tweet}")
                if visual_url:
                    print(f"🖼️  Visual: {visual_url}")
                print("-" * 80)
                
            except Exception as e:
                logger.error(f"Error processing {trend['trend_name']}: {e}")
                continue
    
    logger.info(f"Generated {len(content_packages)} enhanced content packages")
    return content_packages


def export_results(content_packages, filename=None):
    """
    Export results to JSON file with timestamp.
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trend_analysis_results_{timestamp}.json"
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'content_packages': content_packages,
        'summary': {
            'total_packages': len(content_packages),
            'themes_covered': list(set(pkg['theme'] for pkg in content_packages)),
            'total_trends_processed': sum(len(pkg.get('all_angles', [])) for pkg in content_packages)
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
    print("🚀 Starting Enhanced Trend Analysis with Phase 1 Implementation")
    print(f"📋 Configuration: Multi-region={CONFIG['use_multi_region']}, Max regions={CONFIG['max_regions']}, Max per theme={CONFIG['max_trends_per_theme']}")
    
    results = main()
    
    if results:
        # Export results
        export_filename = export_results(results)
        
        # Print final summary
        print(f"\n✅ Analysis complete! Generated {len(results)} content packages")
        if export_filename:
            print(f"📁 Results saved to: {export_filename}")
        
        # Print JSON output for programmatic use
        print("\n" + "="*80)
        print("JSON OUTPUT:")
        print("="*80)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("❌ No results generated") 