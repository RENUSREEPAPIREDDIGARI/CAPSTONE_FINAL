"""Configuration file for API keys and settings

To set up your Unsplash API key:
1. Go to https://unsplash.com/developers
2. Register as a developer
3. Create a new application
4. Copy your Access Key
5. Replace the value below with your key
"""

# Unsplash API Configuration
UNSPLASH_ACCESS_KEY = "mDJF7QkPyyz8OcFsVcmOFnUN82_Y7p4GX3Ud8JOO_N4"

# Image Search Configuration
MAX_IMAGES_PER_SEARCH = 6
SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
CACHE_DURATION = 3600  # Cache duration in seconds

# Directory Configuration
SCRAPED_IMAGES_DIR = 'scraped_images'
CACHE_DIR = '.cache' 
