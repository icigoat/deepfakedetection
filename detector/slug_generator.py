"""
Anime & Motivational Quote Slug Generator
Generates unique, memorable slugs from popular anime, motivational quotes, and LIVE AI news
"""

import random
from django.utils.text import slugify

# One Piece Quotes
ONE_PIECE_QUOTES = [
    "I will become the pirate king",
    "I want to live take me to the sea",
    "Nothing happened",
    "The one piece is real",
    "A man dies when he is forgotten",
    "People's dreams never end",
]

# Naruto Quotes
NARUTO_QUOTES = [
    "I will become hokage believe it",
    "Those who break the rules are trash but those who abandon friends are worse",
    "The pain of being alone is not easy to bear",
    "Hard work is worthless for those who dont believe in themselves",
    "If you dont like your destiny dont accept it",
    "When people protect something precious they become truly strong",
]

# Bleach Quotes
BLEACH_QUOTES = [
    "Bankai",
    "I will protect everyone with this blade",
    "Abandon your fear turn and face him",
    "Do not live bowing down die standing up",
    "We are drawn to each other like drops of water",
    "If I were the rain could I connect with someone",
]

# Jujutsu Kaisen Quotes
JJK_QUOTES = [
    "Throughout heaven and earth I alone am the honored one",
    "Stand proud you are strong",
    "Nah id win",
    "Domain expansion infinite void",
    "You are my special",
    "Love is the most twisted curse of all",
]

# Attack on Titan Quotes
AOT_QUOTES = [
    "Tatakae tatakae ",
    "Shinzou wo sasageyo",
    "If you win you live if you lose you die",
    "The world is cruel but also very beautiful",
    "Freedom is what I sought",
    "Give up on your dreams and die",
]

# Solo Leveling Quotes
SOLO_LEVELING_QUOTES = [
    "Arise",
    "The weak have no right to choose how they die",
    "Shadow monarch awakens now",
    "I am the player of this system",
    "You should have run when you had the chance",
]

# Demon Slayer Quotes
DEMON_SLAYER_QUOTES = [
    "Hinokami kagura",
    "Set your heart ablaze",
    "Growing stronger is not a sin",
    "Go beyond your limits ",
    "Those with regret could not be called demons",
]

# Death Note Quotes
DEATH_NOTE_QUOTES = [
    "I am justice ",
    "I will become the god of this new world",
    "Exactly as planned ",
    "This world is rotten and those making it rot deserve to die",
    "I am L and I will find you",
    "The one who wins is the one who is prepared",
]

# Game of Thrones Quotes
GOT_QUOTES = [
    "Winter is coming ",
    "When you play the game of thrones you win or you die",
    "A lannister always pays his debts",
    "Chaos is a ladder",
    "I drink and I know things",
    "Valar morghulis",
    "The north remembers always",
    "Dracarys",
    "You know nothing jon snow",
    "A man has no name ",
    "Fire cannot kill a dragon ",
    "Hold the door",
    "The night is dark and full of terrors",
    "Not today ",
    "I am the sword in the darkness",
]

# Motivational / Cross-Series Aura Quotes
MOTIVATIONAL_QUOTES = [
    "Surpass your limits right here right now",
    "Power comes in response to a need",
    "The moment you give up is the moment you lose",
    "A lesson without pain is meaningless",
    "If you dont take risks you cant create a future",
    "Believe in yourself and create your own destiny",
    "The duty of a king is to love his people",
    "Where I walk becomes the path",
]

# Combine all quotes
ALL_QUOTES = (
    ONE_PIECE_QUOTES +
    JJK_QUOTES +
    SOLO_LEVELING_QUOTES +
    NARUTO_QUOTES +
    GOT_QUOTES +
    AOT_QUOTES +
    DEMON_SLAYER_QUOTES +
    MOTIVATIONAL_QUOTES
)

def generate_unique_slug(existing_slugs=None):
    """
    Generate a unique slug from anime/motivational quotes/LIVE AI NEWS with random suffix
    Format: quote-or-news/randomid (with slash separator)
    
    Args:
        existing_slugs: Set of already used slugs to avoid duplicates
    
    Returns:
        A unique, URL-safe slug like "gpt-5-achieves-breakthrough/a3f9b2c1"
    """
    if existing_slugs is None:
        existing_slugs = set()
    
    # 30% chance to use live AI news, 70% anime/motivational quotes
    use_live_news = random.random() < 0.3
    
    if use_live_news:
        try:
            from .news_fetcher import get_cached_ai_news
            quote = get_cached_ai_news()
        except:
            # Fallback to regular quotes if news fetch fails
            quote = random.choice(ALL_QUOTES)
    else:
        quote = random.choice(ALL_QUOTES)
    
    # Try to find an unused quote
    max_attempts = 100
    for _ in range(max_attempts):
        slug_base = slugify(quote)
        
        # Limit slug base length to 50 characters for readability
        if len(slug_base) > 50:
            slug_base = slug_base[:50].rstrip('-')
        
        # Add random suffix for uniqueness (shorter: 8 chars)
        import uuid
        random_suffix = uuid.uuid4().hex[:8]  # 8 character random string
        
        # Use slash as separator for better readability
        slug = f"{slug_base}/{random_suffix}"
        
        if slug not in existing_slugs:
            return slug
        
        # Try a different quote if this one is taken
        if use_live_news:
            try:
                from .news_fetcher import get_cached_ai_news
                quote = get_cached_ai_news()
            except:
                quote = random.choice(ALL_QUOTES)
        else:
            quote = random.choice(ALL_QUOTES)
    
    # Fallback (should never reach here)
    import uuid
    quote = random.choice(ALL_QUOTES)
    slug_base = slugify(quote)[:50].rstrip('-')
    slug = f"{slug_base}/{uuid.uuid4().hex[:8]}"
    return slug

def get_random_quote():
    """Get a random quote (not slugified)"""
    return random.choice(ALL_QUOTES)
