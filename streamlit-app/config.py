"""Configuration settings for the application."""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
UNSPLASH_ACCESS_KEY = os.getenv('UNSPLASH_ACCESS_KEY')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# API Base URLs
SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"

# Validate required keys
def validate_config():
    """Validate that all required API keys are present."""
    missing_keys = []
    
    if not SAMBANOVA_API_KEY:
        missing_keys.append("SAMBANOVA_API_KEY")
    if not UNSPLASH_ACCESS_KEY:
        missing_keys.append("UNSPLASH_ACCESS_KEY")
    if not REPLICATE_API_TOKEN:
        missing_keys.append("REPLICATE_API_TOKEN")
    
    return missing_keys

# Image Search Configuration
MAX_IMAGES_PER_SEARCH = 6
SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
CACHE_DURATION = 3600  # Cache duration in seconds

# Directory Configuration
SCRAPED_IMAGES_DIR = 'scraped_images'
CACHE_DIR = '.cache' 