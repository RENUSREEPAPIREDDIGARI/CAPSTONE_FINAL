"""Social Media Integration Module"""

import logging
import os
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SocialMediaManager:
    def __init__(self):
        pass
    
    def optimize_for_instagram(self, image_path: str) -> Dict[str, any]:
        """Optimize image specifically for Instagram"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if image is in RGBA mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Instagram recommended sizes
            sizes = {
                'square': (1080, 1080),  # Square post
                'portrait': (1080, 1350),  # Portrait/vertical post
                'landscape': (1080, 608)  # Landscape/horizontal post
            }
            
            optimized_images = {}
            
            for format_name, target_size in sizes.items():
                try:
                    # Create a copy of the image
                    img_copy = image.copy()
                    
                    # Resize image maintaining aspect ratio
                    img_copy.thumbnail(target_size, Image.LANCZOS)
                    
                    # Create white background for Instagram
                    background = Image.new('RGB', target_size, (255, 255, 255))
                    
                    # Calculate position to paste the image (center)
                    paste_x = (target_size[0] - img_copy.size[0]) // 2
                    paste_y = (target_size[1] - img_copy.size[1]) // 2
                    
                    # Paste the image onto the background
                    background.paste(img_copy, (paste_x, paste_y))
                    
                    # Save optimized image
                    optimized_path = f"{os.path.splitext(image_path)[0]}_instagram_{format_name}.jpg"
                    background.save(optimized_path, 'JPEG', quality=95, optimize=True)
                    
                    optimized_images[format_name] = optimized_path
                    
                except Exception as format_error:
                    logger.error(f"Error processing {format_name} format: {str(format_error)}")
                    continue
            
            if not optimized_images:
                return {
                    'success': False,
                    'message': 'Failed to optimize image for any format'
                }
            
            return {
                'success': True,
                'images': optimized_images,
                'message': 'Images optimized for Instagram in multiple formats'
            }
            
        except Exception as e:
            logger.error(f"Error optimizing image for Instagram: {str(e)}")
            return {
                'success': False,
                'message': f"Error optimizing image: {str(e)}"
            }
    
    def prepare_instagram_caption(self, caption: str, hashtags: List[str] = None) -> str:
        """Prepare caption for Instagram with proper formatting"""
        # Instagram caption limit is 2,200 characters
        MAX_CAPTION_LENGTH = 2200
        MAX_HASHTAGS = 30
        
        # Clean and format hashtags
        if hashtags:
            hashtags = hashtags[:MAX_HASHTAGS]  # Limit to 30 hashtags
            hashtag_text = "\n.\n.\n." + "\n" + " ".join([f"#{tag.strip('#')}" for tag in hashtags])
        else:
            hashtag_text = ""
            
        # Format main caption
        formatted_caption = caption.strip()
        
        # Combine caption with hashtags
        full_caption = formatted_caption + hashtag_text
        
        # Truncate if exceeds limit
        if len(full_caption) > MAX_CAPTION_LENGTH:
            available_space = MAX_CAPTION_LENGTH - len(hashtag_text) - 3
            truncated_caption = formatted_caption[:available_space] + "..."
            full_caption = truncated_caption + hashtag_text
            
        return full_caption
    
    def get_instagram_guidelines(self) -> Dict[str, any]:
        """Get comprehensive Instagram posting guidelines"""
        return {
            'image_formats': {
                'square': '1080x1080 pixels (1:1 ratio)',
                'portrait': '1080x1350 pixels (4:5 ratio)',
                'landscape': '1080x608 pixels (1.91:1 ratio)'
            },
            'caption_rules': {
                'max_length': '2,200 characters',
                'hashtag_limit': '30 hashtags',
                'line_breaks': 'Use . or - for clean line breaks',
                'mentions': 'Tag up to 20 accounts'
            },
            'best_practices': [
                'Post high-quality, well-lit images',
                'Use relevant hashtags',
                'Engage with your audience',
                'Post consistently',
                'Add location tags when relevant',
                'Write engaging captions'
            ],
            'posting_steps': [
                "1. Open Instagram app",
                "2. Tap '+' to create new post",
                "3. Select the optimized image",
                "4. Tap 'Next' and edit if needed",
                "5. Paste your prepared caption",
                "6. Add location and tags if desired",
                "7. Share your post"
            ]
        } 