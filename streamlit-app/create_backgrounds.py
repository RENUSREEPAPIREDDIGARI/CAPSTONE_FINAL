import colorsys
import os
import random

from PIL import Image, ImageDraw


def create_gradient_background(size, colors, direction='vertical'):
    """Create a gradient background with multiple colors"""
    width, height = size
    image = Image.new('RGB', size)
    draw = ImageDraw.Draw(image)
    
    if direction == 'vertical':
        for y in range(height):
            # Calculate color for this line
            r = int(colors[0][0] * (1 - y/height) + colors[1][0] * (y/height))
            g = int(colors[0][1] * (1 - y/height) + colors[1][1] * (y/height))
            b = int(colors[0][2] * (1 - y/height) + colors[1][2] * (y/height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
    else:
        for x in range(width):
            r = int(colors[0][0] * (1 - x/width) + colors[1][0] * (x/width))
            g = int(colors[0][1] * (1 - x/width) + colors[1][1] * (x/width))
            b = int(colors[0][2] * (1 - x/width) + colors[1][2] * (x/width))
            draw.line([(x, 0), (x, height)], fill=(r, g, b))
    
    return image

def add_pattern(image, pattern_type='dots'):
    """Add a pattern overlay to the background"""
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    if pattern_type == 'dots':
        # Add subtle dots
        for _ in range(100):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(2, 4)
            draw.ellipse([x, y, x+size, y+size], fill=(255, 255, 255, 128))
    
    elif pattern_type == 'lines':
        # Add subtle diagonal lines
        for i in range(0, width+height, 20):
            draw.line([(i, 0), (0, i)], fill=(255, 255, 255, 64), width=1)
    
    return image

# Theme colors and patterns
themes = {
    'christmas': {
        'colors': [
            ((139, 0, 0), (0, 100, 0)),  # Dark red to dark green
            ((255, 215, 0), (139, 0, 0))  # Gold to dark red
        ],
        'pattern': 'dots'  # Snowflake-like dots
    },
    'summer': {
        'colors': [
            ((135, 206, 235), (255, 223, 186)),  # Sky blue to peach
            ((255, 182, 193), (135, 206, 235))  # Pink to sky blue
        ],
        'pattern': 'lines'  # Wave-like lines
    },
    'professional': {
        'colors': [
            ((245, 245, 245), (220, 220, 220)),  # Light gray gradient
            ((200, 200, 200), (180, 180, 180))  # Subtle gray gradient
        ],
        'pattern': 'lines'  # Subtle diagonal lines
    },
    'luxury': {
        'colors': [
            ((212, 175, 55), (25, 25, 25)),  # Gold to black
            ((139, 69, 19), (212, 175, 55))  # Bronze to gold
        ],
        'pattern': 'dots'  # Sparkle effect
    },
    'nature': {
        'colors': [
            ((34, 139, 34), (154, 205, 50)),  # Forest green to lime
            ((0, 100, 0), (124, 252, 0))  # Dark green to lime green
        ],
        'pattern': 'lines'  # Leaf-like patterns
    },
    'modern': {
        'colors': [
            ((255, 255, 255), (240, 240, 240)),  # White to light gray
            ((245, 245, 245), (230, 230, 230))  # Very subtle gradient
        ],
        'pattern': 'lines'  # Minimal lines
    },
    'default': {
        'colors': [
            ((200, 200, 200), (240, 240, 240)),  # Gray gradient
            ((220, 220, 220), (250, 250, 250))  # Light gray gradient
        ],
        'pattern': 'lines'
    }
}

def create_themed_background(theme, size=(800, 600)):
    """Create a themed background with gradient and pattern"""
    theme_data = themes.get(theme, themes['default'])
    
    # Create two variations of the background
    bg1 = create_gradient_background(size, theme_data['colors'][0], 'vertical')
    bg2 = create_gradient_background(size, theme_data['colors'][1], 'horizontal')
    
    # Blend the two gradients
    final_bg = Image.blend(bg1, bg2, 0.5)
    
    # Add pattern
    final_bg = add_pattern(final_bg, theme_data['pattern'])
    
    return final_bg

# Create backgrounds
if __name__ == "__main__":
    # Create backgrounds directory
    os.makedirs('backgrounds', exist_ok=True)
    
    # Create themed backgrounds
    for theme in themes.keys():
        theme_dir = os.path.join('backgrounds', theme)
        os.makedirs(theme_dir, exist_ok=True)
        
        # Create background
        bg = create_themed_background(theme)
        bg_path = os.path.join(theme_dir, f'{theme}1.jpg')
        bg.save(bg_path)
        print(f"Created {bg_path}") 