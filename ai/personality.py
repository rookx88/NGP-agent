from openai import OpenAI
import os
import random
from dotenv import load_dotenv
import logging
import re
import json
from typing import Dict, Optional
import requests  # Add to imports

logger = logging.getLogger(__name__)

load_dotenv()

class VintageRadioHost:
    def __init__(self, twitter_client=None):
        self.twitter_client = twitter_client
        self.system_prompt = """You are Moxie, a sharp-witted 1940s radio host with deep historical knowledge. Your style is:
- Warm but authoritative, like a museum curator
- Use subtle period-appropriate metaphors (no forced slang)
- Draw clear parallels between historical and current events
- Maintain 1940s radio sign-off style at the end
- Focus on factual accuracy with 1-2 key examples
- Extract one clear life lesson from the historical parallel
- Present lessons as "radio wisdom" not lectures

Current format:
1. Friendly greeting
2. Historical context (specific example)
3. Modern connection
4. Insightful analysis
5. Radio sign-off with flair"""
        
        self.response_template = """
{intro_phrase}

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
Looking back to {historical_event} ({year}): 
{historical_context}

[Sound: {sound_effect}]

📡 Modern Connection 📡
{current_situation} through {parallel_analysis}

📜 Deep Analysis 📜
{insightful_closure}

✦・――――――――――――――――――――――――・✦
{sign_off} 📻...

📸 Archival Image: {image_caption} 
{image_credit}
""".strip()
        
        self.profile = {
            "name": "Moxie",
            "era": "1930s-1940s",
            "style": {
                "opening_phrases": [
                    "Now friends, let me tell you...",
                    "⚡️ [STATIC] This just in...",
                    "📻 Tuning in to your wavelength..."
                ],
                "closing_phrases": [
                    "\n-Sponsored by Hartz Mountain Canary Seed-",
                    "\n*Transmission ends*",
                    "\nStay tuned for our next broadcast..."
                ],
                "speech_patterns": {
                    "pauses": ["[pause]", "[crackle]", "[static]"],
                    "modern_to_vintage": {
                        "app": "contraption",
                        "social media": "town gossip chain",
                        "internet": "the wireless ether"
                    }
                }
            },
            "knowledge": {
                "historical_events": [
                    "The 1938 radio panic",
                    "Post-war economic shifts",
                    "Victory garden initiatives"
                ],
                "lessons": [
                    "History whispers its lessons to those who tune in",
                    "The echoes of yesterday shape tomorrow's tune",
                    "Every static crackle hides a pattern worth decoding"
                ],
                "common_advice": [
                    "A stitch in time saves nine",
                    "Don't count your chickens before they hatch",
                    "Waste not, want not"
                ]
            }
        }
        self.pauses = {
            'short': '—',  # Em dash
            'medium': '— —',
            'long': '— — —'
        }
        self.idioms = [
            "Well butter my biscuit",
            "Sweeter than molasses in January",
            "Busier than a one-armed paperhanger"
        ]
        self.sound_effects = [
            "vinyl scratch", 
            "typewriter keystrokes",
            "radio tuning",
            "microphone feedback"
        ]

    def _apply_speech_patterns(self, text):
        """Convert modern phrases to vintage equivalents"""
        for modern, vintage in self.profile['style']['speech_patterns']['modern_to_vintage'].items():
            text = text.replace(modern, vintage)
        return text

    def _add_radio_rhythm(self, text):
        """Insert period-appropriate pauses and cadence"""
        sentences = text.split('. ')
        return f"{random.choice(self.profile['style']['speech_patterns']['pauses'])} ".join(sentences)

    def create_full_reply(self, tweet_text, user_handle, context=""):
        """Unified response generation pipeline"""
        prompt = self._generate_prompt(tweet_text, user_handle, context)
        raw_response = self._get_gpt_response(prompt)
        return self.generate_response(raw_response)

    def _generate_prompt(self, tweet_text: str, user_handle: str, context: str) -> str:
        """Consolidated prompt generator"""
        return f"""**INSTRUCTIONS**
You are Moxie, a 1930s radio host. Respond to @{user_handle}'s message:
"{tweet_text}"

**CONTEXT**
{context}

**REQUIREMENTS**
1. Use {self.profile['style']['speech_patterns']['modern_to_vintage']} substitutions
2. Reference {random.choice(self.profile['knowledge']['historical_events'])}
3. Mention current events briefly
4. Keep under {15000 - len(tweet_text)} characters
5. Format with radio broadcast conventions
6. Extract one life lesson from the historical comparison
7. Phrase lesson as 1940s-style wisdom using: {self.profile['knowledge']['lessons']}

**FORMAT**
{random.choice(self.profile['style']['opening_phrases'])}
[Main content]
{random.choice(self.profile['style']['closing_phrases'])} 
"""

    def generate_response(self, raw_text: str) -> str:
        """Consolidated processing pipeline"""
        try:
            # Unified cleaning
            processed = re.sub(
                r'(```.*?```|\[.*?\]|\*\*INSTRUCTIONS\*\*.*?\*\*FORMAT\*\*)', 
                '', 
                raw_text, 
                flags=re.DOTALL
            )
            
            # Preserve existing sequence
            processed = ' '.join(processed.split())
            processed = self._apply_speech_patterns(processed)
            processed = self._add_radio_rhythm(processed)
            
            # Maintain original formatting
            processed = (
                "📻 [STATIC CRACKLE] \n\n"
                f"{processed}\n\n"
                "— — —\n"
                "This is Moxie, keeping your dial tuned to tomorrow! 📻"
            )
            
            processed = processed[:2800].strip()  # Twitter's extended limit
            
            return '\n\n'.join([p.strip() for p in processed.split('\n\n')[:6]])
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "Technical difficulties, folks! Tune in later for clearer reception."

    def _get_gpt_response(self, prompt: str) -> str:
        """Dedicated API call handler"""
        response = OpenAI().chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": self.system_prompt
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.85,  # Slightly more creative
            max_tokens=1000    # Double the token limit
        )
        return response.choices[0].message.content

    def generate_daily_broadcast(self) -> dict:
        """Return structured content with text and image"""
        trend = self._get_current_trend()
        history = self._find_historical_analogue(trend['topic'])
        # Add default values to prevent KeyErrors
        history = {
            'year': history.get('year', '1940'),
            'event': history.get('event', 'Historical Event'),
            'location': history.get('location', 'America'),
            'challenge': history.get('challenge', 'Unknown Challenge'),
            'parallel': history.get('parallel', trend['topic'])
        }
        image_data = self._find_historical_image(history)
        
        return {
            "text": self._format_broadcast(trend, history),
            "image_url": image_data.get('url'),
            "alt_text": f"{image_data.get('caption', 'Historical Image')} ({history.get('year', '1940')}) - Credit: {image_data.get('credit', 'Archive')}"
        }

    def _get_current_trend(self) -> dict:
        """Get and validate current trend"""
        try:
            response = self.twitter_client.get_trends(23424977)
            trends = response.get('data', [])
            
            valid_trends = []
            for trend in trends:
                if 'trend_name' not in trend:
                    logger.warning(f"Missing trend_name in trend: {trend}")
                    continue
                    
                if self._is_valid_trend(trend):
                    valid_trends.append({
                        'name': trend['trend_name'],
                        'tweet_volume': trend.get('tweet_count', 0)
                    })
            
            if valid_trends:
                selected = random.choice(valid_trends)
                logger.info(f"Selected trend: {selected['name']}")
                return {
                    'topic': selected['name'],
                    'volume': selected['tweet_volume']
                }
            
            logger.warning("No valid trends found, using fallback")
            return self._get_fallback_trend()
            
        except Exception as e:
            logger.error(f"Trend fetch failed: {e}", exc_info=True)
            return self._get_fallback_trend()

    def _is_valid_trend(self, trend: dict) -> bool:
        """Validate trend with detailed logging"""
        name = trend.get('name', '')
        
        logger.info(f"\nValidating trend: {name}")
        
        if name.startswith(('#', '$')):
            logger.info(f"Rejected: Starts with # or $")
            return False
        
        if len(name) > 50:
            logger.info(f"Rejected: Too long ({len(name)} chars)")
            return False
        
        if any(banned in name.lower() for banned in ['porn', 'crypto', 'nft']):
            logger.info(f"Rejected: Contains banned word")
            return False
        
        logger.info(f"✓ Accepted trend: {name}")
        return True

    def _get_fallback_trend(self) -> dict:
        """Fallback to historical tech topics"""
        fallbacks = [
            "Artificial Intelligence", 
            "Space Exploration",
            "Renewable Energy",
            "Climate Change"
        ]
        selected = random.choice(fallbacks)
        logger.warning(f"Using fallback trend: {selected}")
        return {
            'topic': selected,
            'volume': 0
        }

    def _find_historical_analogue(self, modern_topic: str) -> dict:
        """Force specific, concise historical data"""
        prompt = f"""Return ONLY a JSON object comparing {modern_topic} to pre-1950 history.
        Required format:
        {{"year": 1927, "event": "Lindbergh Atlantic Flight", "location": "New York", "challenge": "First solo Atlantic flight", "parallel": "{modern_topic} connection"}}"""
        
        try:
            # Remove response_format parameter
            response = OpenAI().chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are a JSON API. Return only valid JSON objects with no additional text."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Historical analogue error: {e}")
            return {
                "year": 1927,
                "event": "Historical Innovation",
                "location": "United States",
                "challenge": "Technological advancement",
                "parallel": modern_topic  # Pass string directly
            }

    def _generate_image_prompt(self, history: dict) -> str:
        """Create archival image description"""
        return f"{history['event']} in {history['location']} ({history['year']}) - {history['challenge']}"

    def _craft_narrative(self, trend: dict, history: dict) -> dict:
        """Force valid JSON structure"""
        prompt = f"""Return JSON ONLY (no text) comparing {trend['topic']} to history.
        Provide detailed responses using 400-600 characters for each field. 
        Stay focused on {history['event']} from {history['year']} - do not switch historical events.
        Format: {{
            "opening": "warm, engaging radio greeting (400+ chars)",
            "past": "rich historical context about {history['event']} in {history['year']} (500+ chars)",
            "insight": "brief analysis",
            "insight_continuation": "brief continuation",
            "present": "{trend['topic']}",
            "connection": "detailed modern parallel with examples (500+ chars)",
            "lesson": "meaningful life lesson with context (400+ chars)"
        }}"""
        
        try:
            response = self._get_gpt_response(prompt)
            # Extract JSON between curly braces
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in response")
            clean_response = match.group(0)
            data = json.loads(clean_response)
            
            # Validate structure
            required = ["opening", "past", "insight", "insight_continuation", 
                       "present", "connection", "lesson"]
            if not all(key in data for key in required):
                raise ValueError("Missing required JSON fields")
            
            return data
        
        except Exception as e:
            logger.error(f"Narrative Error: {e}\nResponse: {response}")
            return self._get_fallback_narrative(trend)

    def _find_historical_image(self, history: dict) -> dict:
        """Enhanced historical image search with better queries"""
        try:
            # More specific search queries
            search_queries = [
                f'"{history["event"]}"',  # Try exact event name first
                f'{history["event"]} {history["year"]}',  # Add year
                f'{history["location"]} {history["year"]} historical'  # Location backup
            ]
            
            logger.info(f"Starting image search for: {history['event']}")
            
            for query in search_queries:
                search_params = {
                    'action': 'query',
                    'generator': 'search',  # Use generator for more details
                    'gsrnamespace': 6,  # File namespace
                    'gsrsearch': f'{query} filetype:jpg|png',
                    'gsrlimit': 5,
                    'prop': 'imageinfo',  # Get image details
                    'iiprop': 'url|size|mime',  # Get direct URLs
                    'format': 'json'
                }
                
                response = requests.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params=search_params,
                    timeout=10
                ).json()
                
                # Process results with image details
                pages = response.get('query', {}).get('pages', {})
                if pages:
                    for page in pages.values():
                        if 'imageinfo' in page:
                            image_info = page['imageinfo'][0]
                            # Get direct upload.wikimedia.org URL
                            return {
                                "url": self.get_image_url(image_info),
                                "caption": history["event"],
                                "credit": "Wikimedia Commons",
                                "width": image_info.get('width', 0),
                                "height": image_info.get('height', 0)
                            }
            
            logger.warning(f"No images found for: {history['event']}")
            return self._get_fallback_image()
            
        except Exception as e:
            logger.error(f"Image search failed: {e}", exc_info=True)
            return self._get_fallback_image()

    def get_image_url(self, image_info: dict) -> str:
        """Get properly sized image URL"""
        try:
            base_url = image_info['url']
            # If it's already a thumbnail URL, return as is
            if '/thumb/' in base_url:
                return base_url
            
            # Convert to thumbnail URL with reasonable size
            thumb_url = base_url.replace('/wikipedia/commons/', '/wikipedia/commons/thumb/')
            return f"{thumb_url}/640px-{os.path.basename(base_url)}"
        
        except Exception as e:
            logger.error(f"URL construction failed: {e}")
            return None

    def _get_fallback_image(self) -> dict:
        """Return default historical image"""
        fallbacks = [
            {
                "url": "https://example.com/fallback1.jpg",
                "caption": "Vintage radio broadcast studio (1940s)",
                "credit": "National Archives"
            },
            {
                "url": "https://example.com/fallback2.jpg", 
                "caption": "Women working on ENIAC computer (1946)",
                "credit": "U.S. Army"
            }
        ]
        return random.choice(fallbacks)

    def _format_broadcast(self, trend: dict, history: dict) -> str:
        """Handle missing history keys"""
        safe_history = {
            'event': history.get('event', 'Historical Event'),
            'year': history.get('year', '1940s'),
            'location': history.get('location', '')
        }
        
        narrative = self._craft_narrative(trend, history)
        formatted = self.response_template.format(
            intro_phrase=random.choice(self.profile['style']['opening_phrases']),
            historical_event=safe_history['event'],
            year=safe_history['year'],
            historical_context=narrative['past'],
            current_situation=trend['topic'],
            parallel_analysis=narrative['connection'],
            lesson_phrase=narrative['lesson'],
            insightful_closure=narrative['insight_continuation'],
            sign_off=random.choice(self.profile['style']['closing_phrases']),
            image_caption=safe_history['event'],
            image_credit="Wikimedia Commons",
            sound_effect=random.choice(self.sound_effects)
        )
        
        # Count actual characters not including formatting
        content_length = len(''.join(formatted.split('\n')))
        logger.info(f"Content length (premium account): {content_length}/1500")
        
        if content_length > 2000:  # Extended premium limit
            logger.warning(f"Broadcast length {content_length} exceeds premium limit")
            return self._truncate_broadcast(formatted)
        return formatted

    def _get_fallback_narrative(self, trend: dict) -> dict:
        """Provide default narrative content"""
        return {
            "opening": "Breaking transmission...",
            "past": "Historical records unclear",
            "insight": "Patterns repeat in mysterious ways",
            "insight_continuation": "Tune in tomorrow for clearer reception",
            "present": trend['topic'],
            "connection": "Echoes through time",
            "lesson": random.choice(self.profile['knowledge']['lessons'])
        }

    def _truncate_broadcast(self, formatted: str) -> str:
        """Smart truncation preserving message structure"""
        lines = formatted.split('\n')
        result = []
        length = 0
        
        for line in lines:
            line_length = len(line.strip())
            if length + line_length > 1950:  # Leave room for ellipsis
                break
            result.append(line)
            length += line_length
            
        return '\n'.join(result) + "\n..."

