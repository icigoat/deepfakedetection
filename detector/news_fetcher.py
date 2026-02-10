"""
Real-time AI News Fetcher
Fetches live AI news headlines from various sources
"""

import requests
from datetime import datetime, timedelta
import random

def fetch_ai_news_headlines(max_headlines=20):
    """
    Fetch real-time AI news headlines from NewsAPI
    
    Returns:
        List of AI news headlines (strings)
    """
    headlines = []
    
    try:
        # NewsAPI - Free tier (get your API key from https://newsapi.org/)
        # For now, using a fallback method with RSS feeds
        
        # Try multiple sources
        sources = [
            fetch_from_techcrunch(),
            fetch_from_verge(),
            fetch_from_hackernews(),
        ]
        
        for source_headlines in sources:
            headlines.extend(source_headlines)
            if len(headlines) >= max_headlines:
                break
        
        # Remove duplicates and limit
        headlines = list(set(headlines))[:max_headlines]
        
    except Exception as e:
        print(f"Error fetching AI news: {e}")
        # Return fallback headlines if fetch fails
        headlines = get_fallback_headlines()
    
    return headlines if headlines else get_fallback_headlines()


def fetch_from_techcrunch():
    """Fetch AI news from TechCrunch RSS"""
    headlines = []
    try:
        import feedparser
        feed = feedparser.parse('https://techcrunch.com/tag/artificial-intelligence/feed/')
        for entry in feed.entries[:10]:
            title = entry.title.strip()
            if len(title) > 10:  # Valid headline
                headlines.append(title)
    except:
        pass
    return headlines


def fetch_from_verge():
    """Fetch AI news from The Verge RSS"""
    headlines = []
    try:
        import feedparser
        feed = feedparser.parse('https://www.theverge.com/rss/ai-artificial-intelligence/index.xml')
        for entry in feed.entries[:10]:
            title = entry.title.strip()
            if len(title) > 10:
                headlines.append(title)
    except:
        pass
    return headlines


def fetch_from_hackernews():
    """Fetch AI-related news from Hacker News API"""
    headlines = []
    try:
        # Get top stories
        response = requests.get('https://hacker-news.firebaseio.com/v0/topstories.json', timeout=5)
        story_ids = response.json()[:30]
        
        # Fetch story details
        for story_id in story_ids[:15]:
            story_response = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json', timeout=3)
            story = story_response.json()
            
            if story and 'title' in story:
                title = story['title']
                # Filter for AI-related content
                ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'neural', 'gpt', 'llm', 'deepfake', 'chatgpt', 'openai']
                if any(keyword in title.lower() for keyword in ai_keywords):
                    headlines.append(title)
                    
            if len(headlines) >= 10:
                break
    except:
        pass
    return headlines


def get_fallback_headlines():
    """Fallback headlines if API fails"""
    return [
        "AI breakthrough in natural language processing",
        
    ]


def get_random_ai_news():
    """Get a single random AI news headline"""
    headlines = fetch_ai_news_headlines()
    return random.choice(headlines) if headlines else "AI continues to evolve rapidly"


# Cache headlines for 1 hour to avoid excessive API calls
_cached_headlines = None
_cache_time = None

def get_cached_ai_news():
    """Get cached AI news (refreshes every hour)"""
    global _cached_headlines, _cache_time
    
    now = datetime.now()
    
    # Refresh cache if older than 1 hour or doesn't exist
    if _cached_headlines is None or _cache_time is None or (now - _cache_time) > timedelta(hours=1):
        _cached_headlines = fetch_ai_news_headlines()
        _cache_time = now
    
    return random.choice(_cached_headlines) if _cached_headlines else "AI technology advances daily"
