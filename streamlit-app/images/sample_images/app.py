import csv
import io
import json
import os
import random
import tempfile
import time
from datetime import datetime
from io import BytesIO
from urllib.parse import urljoin

import config
import facebook
import numpy as np
import openai
import pandas as pd
import replicate
import requests
import streamlit as st
import twitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance
from rembg import remove

# Load environment variables
load_dotenv()

def scrape_images(search_query, num_images=6):
    """
    Scrape images from Unsplash based on the search query
    """
    try:
        # Create cache and scraped images directories if they don't exist
        os.makedirs('scraped_images', exist_ok=True)
        os.makedirs('.cache', exist_ok=True)
        
        # Cache key for this search
        cache_key = f"{search_query}_{num_images}"
        cache_file = os.path.join('.cache', f"{hash(cache_key)}.json")
        
        # Check cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                if time.time() - cache_data['timestamp'] < 3600:  # 1 hour cache
                    return cache_data['images']
        
        # Fetch from Unsplash API
        unsplash_api_url = "https://api.unsplash.com/search/photos"
        headers = {
            'Authorization': f'Client-ID {config.UNSPLASH_ACCESS_KEY}'
        }
        
        params = {
            'query': search_query,
            'per_page': num_images,
            'orientation': 'landscape'
        }
        
        response = requests.get(unsplash_api_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            downloaded_images = []
            
            for i, photo in enumerate(data['results']):
                try:
                    img_url = photo['urls']['regular']
                    img_response = requests.get(img_url)
                    
                    if img_response.status_code == 200:
                        # Generate a unique filename using photo ID
                        photo_id = photo['id']
                        img_path = os.path.join('scraped_images', f'{photo_id}.jpg')
                        
                        # Save the image
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        
                        # Add image metadata
                        image_data = {
                            'path': img_path,
                            'photographer': photo['user']['name'],
                            'description': photo['description'] or photo['alt_description'] or '',
                            'url': photo['links']['html']
                        }
                        downloaded_images.append(image_data)
                        
                except Exception as e:
                    st.warning(f"Error downloading image {i}: {str(e)}")
                    continue
            
            # Cache the results
            cache_data = {
                'timestamp': time.time(),
                'images': downloaded_images
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            return downloaded_images
        else:
            st.error(f"Failed to fetch images from Unsplash API: {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"Error in web scraping: {str(e)}")
        return []

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="GenovateAI - Social Media Content Generator",
    page_icon="🎯",
    layout="wide"
)

# Add custom CSS with updated logo design
st.markdown("""
<style>
    /* Base styling */
    .stApp {
        background: #000000;
        color: #FFFFFF;
    }
    
    /* Logo styling */
    .logo-container {
        text-align: center;
        margin: 2rem auto;
        padding: 2rem;
        max-width: 600px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 102, 255, 0.1);
    }
    
    .logo {
        width: 80px;
        height: 80px;
        position: relative;
        margin: 0 auto 1rem;
        background: linear-gradient(45deg, #0066ff, #00ccff);
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(0, 102, 255, 0.3);
        transform: rotate(45deg);
        transition: all 0.3s ease;
    }
    
    .logo:hover {
        transform: rotate(45deg) scale(1.05);
        box-shadow: 0 0 40px rgba(0, 102, 255, 0.5);
    }
    
    .logo::after {
        content: '';
        position: absolute;
        width: 70%;
        height: 70%;
        top: 15%;
        left: 15%;
        background: #000000;
        border-radius: 8px;
    }
    
    .brand-title {
        font-family: 'Arial Black', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #0066ff, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        letter-spacing: -1px;
        text-shadow: 0 0 20px rgba(0, 102, 255, 0.5);
    }
    
    .tagline {
        font-size: 1.2rem;
        color: #ffffff;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 10px rgba(0, 102, 255, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #0066ff, #00ccff) !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        color: white !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(0, 102, 255, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 30px rgba(0, 102, 255, 0.5) !important;
    }

    /* Input fields and other elements */
    .stTextInput > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(0, 102, 255, 0.2) !important;
        color: white !important;
    }

    .stTextInput > div > div:hover {
        border-color: rgba(0, 102, 255, 0.5) !important;
    }

    .stTextArea > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(0, 102, 255, 0.2) !important;
        color: white !important;
    }

    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(0, 102, 255, 0.2) !important;
        color: white !important;
    }

    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    p, span, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Footer styling */
    .footer {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border-top: 1px solid rgba(0, 102, 255, 0.1);
    }
</style>

<div class="logo-container">
    <div class="logo"></div>
    <h1 class="brand-title">GenovateAI</h1>
    <div class="tagline">Next-Gen Content Creation</div>
</div>
""", unsafe_allow_html=True)

# Initialize SambaNova client
client = OpenAI(
    api_key="635091b0-62d7-4e56-a89f-a8f37080e6a6",
    base_url="https://api.sambanova.ai/v1",
)

# Initialize OpenAI client
openai_client = OpenAI(
    api_key="635091b0-62d7-4e56-a89f-a8f37080e6a6",
    base_url="https://api.sambanova.ai/v1",
)

# Load data mapping
@st.cache_data
def load_data_mapping():
    try:
        # Use the correct path to data_mapping.csv in the embedding-generation folder
        df = pd.read_csv('../embedding-generation/data_mapping.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data mapping: {str(e)}")
        return pd.DataFrame()

def find_similar_posts(product_type, tone, n=3):
    """Find similar posts based on product type and tone"""
    try:
        df = load_data_mapping()
        if df.empty:
            return []
        
        # Clean the data - fill NaN values and ensure string type
        df['file_name'] = df['file_name'].fillna('').astype(str)
        df['text'] = df['text'].fillna('').astype(str)
        df['local_path'] = df['local_path'].fillna('').astype(str)
        
        # Filter posts by product type
        product_posts = df[df['file_name'].str.contains(product_type, case=False, na=False)]
        
        # Score posts based on tone similarity
        def calculate_similarity_score(row):
            text = str(row['text']).lower()
            return 1 if tone.lower() in text else 0
        
        # Calculate similarity scores
        similarity_scores = product_posts.apply(calculate_similarity_score, axis=1)
        product_posts = product_posts.copy()  # Create a copy to avoid SettingWithCopyWarning
        product_posts['similarity_score'] = similarity_scores
        
        # Sort by similarity score and return top n posts
        similar_posts = product_posts.nlargest(n, 'similarity_score')
        # Include only necessary columns
        return similar_posts[['text', 'file_name', 'local_path']].to_dict('records')
    except Exception as e:
        st.error(f"Error finding similar posts: {str(e)}")
        return []

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """Enhance image using Pillow"""
    try:
        # Convert to RGB if image is in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Create an ImageEnhance object
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
        
        return image
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return image

def load_image_from_path(post):
    """Load image from local path only"""
    try:
        if 'local_path' in post and post['local_path']:
            # Get the absolute path of the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Handle paths that start with ./
            local_path = post['local_path']
            if local_path.startswith('./'):
                local_path = local_path[2:]  # Remove the ./
            
            # Join with the current directory
            full_path = os.path.join(current_dir, local_path)
            
            if os.path.exists(full_path):
                image = Image.open(full_path)
                # Enhance the image
                enhanced_image = enhance_image(image, brightness=1.2, contrast=1.2, sharpness=1.5)
                return enhanced_image
            else:
                st.warning(f"Image file not found at: {full_path}")
        else:
            st.warning("No local path provided in the post data")
        return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def generate_content_from_patterns(similar_posts, product, brand, tone, description=None):
    """Generate new content using patterns from similar posts and product description"""
    try:
        if not similar_posts:
            return None
            
        # Extract patterns from similar posts
        patterns = {
            'openings': [],
            'hashtags': [],
            'calls_to_action': [],
            'descriptive_phrases': []
        }
        
        for post in similar_posts:
            text = post['text'].lower()
            sentences = text.split('.')
            
            # Extract opening
            if sentences:
                patterns['openings'].append(sentences[0].strip())
            
            # Extract hashtags
            hashtags = [word for word in text.split() if word.startswith('#')]
            patterns['hashtags'].extend(hashtags)
            
            # Extract calls to action
            action_verbs = ['discover', 'experience', 'shop', 'explore', 'find', 'get', 'try']
            for sentence in sentences:
                if any(verb in sentence.lower() for verb in action_verbs):
                    patterns['calls_to_action'].append(sentence.strip())
        
        # Remove duplicates
        patterns['hashtags'] = list(set(patterns['hashtags']))[:3]
        patterns['calls_to_action'] = list(set(patterns['calls_to_action']))[:2]
        
        # Generate new content
        opening = patterns['openings'][0] if patterns['openings'] else f"Discover our amazing {product}"
        
        # Incorporate product description if provided
        if description:
            product_intro = f"Experience the essence of {brand} with our exceptional {product} - {description.strip()}."
        else:
            product_intro = f"Experience the essence of {brand} with our exceptional {product}."
            
        cta = patterns['calls_to_action'][0] if patterns['calls_to_action'] else f"Experience the {brand} difference today"
        hashtags = ' '.join(patterns['hashtags']) if patterns['hashtags'] else f"#{brand.replace(' ', '')} #{product.replace(' ', '')}"
        
        content = f"{opening.capitalize()}\n\n"
        content += f"{product_intro}\n"
        content += f"{cta.capitalize()}\n\n"
        content += hashtags
        
        return content
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return None

def generate_content_with_llm(similar_posts, product, brand, tone, description=None):
    """Generate new content using SambaNova LLM with similar posts as context"""
    try:
        # Extract patterns from similar posts
        patterns = {
            'openings': [],
            'hashtags': [],
            'calls_to_action': [],
            'descriptive_phrases': []
        }
        
        # Collect example posts
        example_posts = []
        for post in similar_posts:
            example_posts.append(post['text'])
            
            # Also collect patterns for reference
            text = post['text'].lower()
            sentences = text.split('.')
            if sentences:
                patterns['openings'].append(sentences[0].strip())
            hashtags = [word for word in text.split() if word.startswith('#')]
            patterns['hashtags'].extend(hashtags)
        
        # Create prompt for LLM
        prompt = f"""Generate a creative social media post for a {product} with these specifications:

Brand: {brand}
Tone: {tone}
Product Description: {description if description else 'A high-quality ' + product}

Here are some successful similar posts for reference:
{chr(10).join(['- ' + post for post in example_posts])}

Common hashtags used: {', '.join(list(set(patterns['hashtags']))[:5])}

Please create a unique post that:
1. Has an attention-grabbing opening
2. Incorporates the product description naturally
3. Maintains the specified tone ({tone})
4. Includes a call to action
5. Ends with 2-3 relevant hashtags
6. Matches the brand voice ({brand})

Format the post with appropriate line breaks and spacing."""

        # Call SambaNova LLM for content generation
        response = client.chat.completions.create(
            model="Llama-3.1-Swallow-8B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a professional social media content creator who specializes in crafting engaging product posts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            top_p=0.1
        )
        
        # Get the generated content
        generated_content = response.choices[0].message.content.strip()
        
        return generated_content
    except Exception as e:
        st.error(f"Error generating content with LLM: {str(e)}")
        # Fallback to pattern-based generation if LLM fails
        return generate_content_from_patterns(similar_posts, product, brand, tone, description)

def remove_background(image):
    """Remove background from image using rembg"""
    try:
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Remove background
        output = remove(img_byte_arr)
        
        # Convert back to PIL Image
        return Image.open(BytesIO(output))
    except Exception as e:
        st.error(f"Error removing background: {str(e)}")
        return image

def add_background(image, background_color=(255, 255, 255)):
    """Add a solid color background to an image"""
    try:
        # Ensure the image has an alpha channel
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        # Create a new image with the background color
        background = Image.new('RGBA', image.size, background_color + (255,))  # Add alpha value
        
        # Create a composite image
        composite = Image.alpha_composite(background, image)
        
        # Convert back to RGB for final output
        return composite.convert('RGB')
    except Exception as e:
        st.error(f"Error adding background: {str(e)}")
        return image

def get_background_image(theme):
    """Get background image path for the selected theme"""
    # Try to get AI-generated background first
    ai_bg_path = os.path.join('backgrounds', theme, f'{theme}_ai_generated.jpg')
    if os.path.exists(ai_bg_path):
        return ai_bg_path
        
    # Generate new AI background
    generated_path = generate_themed_background(theme)
    if generated_path:
        return generated_path
        
    # Fallback to default backgrounds if AI generation fails
    backgrounds_dir = 'backgrounds'
    if theme in os.listdir(backgrounds_dir):
        theme_dir = os.path.join(backgrounds_dir, theme)
        background_files = [f for f in os.listdir(theme_dir) if f.endswith(('.jpg', '.png'))]
        if background_files:
            return os.path.join(backgrounds_dir, theme, background_files[0])
    
    # Default fallback
    return os.path.join(backgrounds_dir, 'modern', 'modern1.jpg')

def generate_themed_background(theme):
    """Generate an AI background based on the theme using Replicate API"""
    try:
        # You'll need to set your REPLICATE_API_TOKEN environment variable
        if 'REPLICATE_API_TOKEN' not in os.environ:
            st.warning("Please set your REPLICATE_API_TOKEN to use AI-generated backgrounds")
            return None

        # Create a prompt based on the theme
        prompts = {
            'modern': "A modern, minimalist abstract background with subtle gradients in blue and green, ultra high quality, 4k",
            'creative': "A creative, artistic background with flowing shapes and vibrant colors, ultra high quality, 4k",
            'professional': "An elegant, professional background with subtle patterns in navy blue and gray, ultra high quality, 4k",
            'tech': "A futuristic technology background with glowing circuits and digital patterns, ultra high quality, 4k",
            'nature': "A serene nature-inspired background with organic patterns and calming colors, ultra high quality, 4k"
        }
        
        prompt = prompts.get(theme, prompts['modern'])
        
        # Generate image using Stable Diffusion
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": prompt,
                "width": 1024,
                "height": 576,
                "refine": "expert_ensemble_refiner",
                "scheduler": "K_EULER",
                "num_inference_steps": 25
            }
        )
        
        if output and isinstance(output, list) and len(output) > 0:
            # Download the generated image
            response = requests.get(output[0])
            if response.status_code == 200:
                # Save the image to the backgrounds directory
                os.makedirs(os.path.join('backgrounds', theme), exist_ok=True)
                img_path = os.path.join('backgrounds', theme, f'{theme}_ai_generated.jpg')
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                return img_path
    except Exception as e:
        st.error(f"Error generating background: {str(e)}")
    return None

# First create the backgrounds directory and default backgrounds
def ensure_backgrounds_exist():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backgrounds_dir = os.path.join(current_dir, 'backgrounds')
        
        # Create main backgrounds directory
        os.makedirs(backgrounds_dir, exist_ok=True)
        
        # Create theme directories and default backgrounds
        themes = ['christmas', 'summer', 'professional', 'luxury', 'nature', 'modern', 'default']
        colors = {
            'christmas': (255, 0, 0),  # Red
            'summer': (135, 206, 235),  # Sky Blue
            'professional': (245, 245, 245),  # Light Gray
            'luxury': (212, 175, 55),  # Gold
            'nature': (34, 139, 34),  # Forest Green
            'modern': (240, 240, 240),  # White
            'default': (200, 200, 200)  # Gray
        }
        
        for theme in themes:
            theme_dir = os.path.join(backgrounds_dir, theme)
            os.makedirs(theme_dir, exist_ok=True)
            
            # Create default background if it doesn't exist
            bg_path = os.path.join(theme_dir, f'{theme}1.jpg')
            if not os.path.exists(bg_path):
                color = colors.get(theme, (240, 240, 240))
                img = Image.new('RGB', (800, 600), color)
                img.save(bg_path)
                
    except Exception as e:
        st.error(f"Error ensuring backgrounds exist: {str(e)}")

# Call this when the app starts
ensure_backgrounds_exist()

def add_custom_background(image, background_image):
    """Add a custom background image to the product image"""
    try:
        if background_image is None:
            st.error("No background image was generated")
            return add_background(image, (255, 255, 255))  # Fallback to white background
            
        # Ensure both images are in RGBA mode
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        # Resize background to match product image size
        background_image = background_image.resize(image.size, Image.LANCZOS)
        
        # Convert background to RGBA
        if background_image.mode != 'RGBA':
            background_image = background_image.convert('RGBA')
        
        # Create a new blank image with the same size
        final_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        # Paste background first
        final_image.paste(background_image, (0, 0))
        
        # Paste the product image on top
        final_image.paste(image, (0, 0), image)
        
        return final_image.convert('RGB')
    except Exception as e:
        st.error(f"Error adding custom background: {str(e)}")
        # Fallback to white background if there's an error
        return add_background(image, (255, 255, 255))

# Main content area
st.subheader("🖼️ Image Selection & Processing")

# Create tabs for different image sources
image_source = st.tabs(["Upload Image", "Search Web Images"])

with image_source[0]:
    # Upload Image Tab
    uploaded_file = st.file_uploader("Upload your product image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.session_state.current_image = uploaded_file
        st.session_state.image_source = "upload"

with image_source[1]:
    # Web Search Tab
    product_search = st.text_input("Search for product images:", placeholder="e.g., leather bag, smart watch, perfume")
    
    if product_search:
        with st.spinner("Searching for images..."):
            try:
                product_images = scrape_images(f"{product_search} product", num_images=6)
                if product_images:
                    st.success(f"Found {len(product_images)} images!")
                    
                    # Display images in a grid
                    cols = st.columns(2)
                    for idx, img_data in enumerate(product_images):
                        with cols[idx % 2]:
                            try:
                                img = Image.open(img_data['path'])
                                st.image(img, use_column_width=True)
                                st.caption(f"📸 {img_data['description'] or 'Product ' + str(idx + 1)}")
                                if st.button(f"Select Image ✨", key=f"select_product_{idx}"):
                                    st.session_state.current_image = img_data['path']
                                    st.session_state.image_source = "web"
                                    st.session_state.image_attribution = f"Image by [{img_data['photographer']}]({img_data['url']}) on Unsplash"
                            except Exception as e:
                                st.error(f"Error displaying image {idx}")
            except Exception as e:
                st.error(f"Error searching images: {str(e)}")

# Only show image processing if an image is selected
if 'current_image' in st.session_state:
    st.divider()
    st.subheader("🎨 Image Enhancement & Background")
    
    # Create two columns for controls and preview
    control_col, preview_col = st.columns([0.4, 0.6])
    
    with control_col:
        # Image Enhancement Controls
        st.write("📊 Enhancement Settings")
        with st.container():
            brightness = st.slider("Brightness", 0.5, 2.0, 1.2, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.2, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 2.0, 1.5, 0.1)
        
        # Background Options
        st.write("🎯 Background Options")
        background_option = st.radio(
            "Choose background type:",
            ["Solid Color", "Search Background Images"],
            horizontal=True
        )
        
        if background_option == "Solid Color":
            bg_color = st.color_picker("Select background color", "#FFFFFF")
            bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        else:
            bg_search_query = st.text_input(
                "Search for backgrounds:",
                placeholder="e.g., abstract, gradient, nature"
            )
            
            if bg_search_query:
                with st.spinner("Searching backgrounds..."):
                    try:
                        bg_images = scrape_images(f"{bg_search_query} background", num_images=4)
                        if bg_images:
                            cols = st.columns(2)
                            for idx, img_data in enumerate(bg_images):
                                with cols[idx % 2]:
                                    try:
                                        img = Image.open(img_data['path'])
                                        st.image(img, use_column_width=True)
                                        if st.button(f"Use Background ✨", key=f"use_bg_{idx}"):
                                            st.session_state.selected_background = img_data['path']
                                            st.session_state.background_attribution = f"Background by [{img_data['photographer']}]({img_data['url']}) on Unsplash"
                                    except Exception as e:
                                        st.error(f"Error displaying background {idx}")
                    except Exception as e:
                        st.error(f"Error searching backgrounds: {str(e)}")
    
    with preview_col:
        st.write("👁️ Preview")
        try:
            # Load and process the current image
            if st.session_state.image_source == "upload":
                current_image = Image.open(st.session_state.current_image)
            else:
                current_image = Image.open(st.session_state.current_image)
            
            # Remove background
            processed_image = remove_background(current_image)
            
            # Add background
            if background_option == "Solid Color":
                final_image = add_background(processed_image, bg_color_rgb)
            else:
                if 'selected_background' in st.session_state:
                    background_img = Image.open(st.session_state.selected_background)
                    final_image = add_custom_background(processed_image, background_img)
                    st.caption(st.session_state.get('background_attribution', ''))
                else:
                    final_image = add_background(processed_image, (255, 255, 255))
            
            # Enhance the image
            enhanced_image = enhance_image(final_image, brightness, contrast, sharpness)
            st.image(enhanced_image, caption='Final Image', use_column_width=True)
            
            # Show attribution if image is from web
            if st.session_state.get('image_attribution'):
                st.caption(st.session_state.image_attribution)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Product Details Section
if 'current_image' in st.session_state:
    st.divider()
    st.subheader("📝 Product Details & Content Generation")
    
    # Create three columns for product details
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
    
    with col1:
        product = st.text_input("Product Name:", placeholder="Enter product name")
        description = st.text_area("Product Description:", 
                                 placeholder="Enter product description",
                                 height=100)
    
    with col2:
        brand = st.selectbox("Brand Type:", ["Luxury Brand", "Casual Brand", "Sports Brand"])
        tone = st.selectbox("Content Tone:", ["Professional", "Casual", "Enthusiastic", "Sophisticated"])
    
    with col3:
        if product and brand and tone:
            st.write("Generate Content")
            if st.button("✨ Create Post", type="primary"):
                with st.spinner("Creating your content..."):
                    similar_posts = find_similar_posts(product, tone)
                    new_content = generate_content_with_llm(similar_posts, product, brand, tone, description)
                    if new_content:
                        st.session_state.generated_content = new_content
                        st.text_area("Generated Content", value=new_content, height=200)
                        st.info("✨ Content generated using AI with inspiration from similar successful posts.")

# Social Media Posting Section
if 'current_image' in st.session_state and 'generated_content' in st.session_state:
    st.divider()
    st.subheader("📱 Share on Social Media")
    
    social_col1, social_col2 = st.columns([0.6, 0.4])
    
    with social_col1:
        st.write("Preview Post")
        if enhanced_image:
            st.image(enhanced_image, use_column_width=True)
        st.text_area("Caption", value=st.session_state.generated_content, height=100, disabled=True)
        
        suggested_hashtags = st.text_input("Add Hashtags (comma-separated)",
                                         placeholder="e.g., #fashion, #style, #trending")
    
    with social_col2:
        st.write("Choose Platforms")
        platforms = []
        if st.checkbox("Post to Facebook"):
            platforms.append('facebook')
        if st.checkbox("Post to Twitter"):
            platforms.append('twitter')
        if st.checkbox("Save for Instagram"):
            platforms.append('instagram')
        
        if st.button("🚀 Share Now", type="primary"):
            if not platforms:
                st.warning("Please select at least one platform to share.")
            else:
                # Save the final image for uploading
                image_path = save_uploadable_image(enhanced_image)
                if image_path:
                    # Prepare caption with hashtags
                    caption = st.session_state.generated_content
                    if suggested_hashtags:
                        hashtags = [tag.strip() for tag in suggested_hashtags.split(',')]
                        caption += "\n\n" + " ".join(hashtags)
                    
                    # Post to selected platforms
                    with st.spinner("Processing your post..."):
                        success = post_to_social_media(image_path, caption, platforms)
                        
                        if success:
                            st.success(f"Successfully processed for: {', '.join(success)}")
                        
                    # Cleanup
                    try:
                        os.remove(image_path)
                        os.rmdir(os.path.dirname(image_path))
                    except:
                        pass

# Add footer with custom styling
st.markdown("""
<style>
.footer {
    position: relative;
    margin-top: 70px;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: grey;
    text-align: center;
    padding: 20px;
    font-size: 14px;
}

/* Add padding to main content to prevent footer overlap */
.main .block-container {
    padding-bottom: 70px;
}

/* Ensure footer stays at bottom even with little content */
.stApp {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.main {
    flex: 1;
}
</style>
<div class="footer">Made with ❤️ using OpenAI</div>
""", unsafe_allow_html=True)

def get_image_suggestions(prompt):
    """Get image suggestions based on the prompt"""
    try:
        # Clean the prompt and extract key terms
        search_terms = prompt.lower().split()
        relevant_terms = [term for term in search_terms if len(term) > 3]
        
        if not relevant_terms:
            return []
            
        # Use the most relevant terms for searching
        search_query = " ".join(relevant_terms[:3])
        return scrape_images(search_query, num_images=6)
        
    except Exception as e:
        st.error(f"Error getting image suggestions: {str(e)}")
        return []

def display_image_suggestions(images):
    """Display image suggestions with attribution"""
    if not images:
        return
        
    cols = st.columns(3)
    for idx, img_data in enumerate(images):
        with cols[idx % 3]:
            st.image(img_data['path'], use_column_width=True)
            st.caption(f"📸 By {img_data['photographer']}")
            if st.button(f"Use Image {idx + 1}", key=f"use_img_{idx}"):
                st.session_state.selected_image = img_data['path']
                st.session_state.image_attribution = f"Photo by [{img_data['photographer']}]({img_data['url']}) on Unsplash"

# Main app code
def main():
    pass

def save_uploadable_image(image):
    """Save PIL Image to a temporary file for uploading"""
    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'post_image.jpg')
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Save with high quality
        image.save(temp_path, 'JPEG', quality=95)
        return temp_path
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def post_to_social_media(image_path, caption, platforms):
    """Provide easy options for social media posting"""
    results = []
    
    try:
        if 'facebook' in platforms:
            # Facebook manual posting option
            st.info("📘 Facebook Posting Instructions:")
            st.write("1. Save the image and copy the caption below")
            st.write("2. Go to your Facebook page/profile")
            st.write("3. Create a new post and upload the saved image")
            st.write("4. Paste the caption and post!")
            
            with open(image_path, 'rb') as img_file:
                btn = st.download_button(
                    label="📥 Download Image for Facebook",
                    data=img_file,
                    file_name="facebook_post.jpg",
                    mime="image/jpeg"
                )
            st.text_area("📝 Caption for Facebook", value=caption, height=100)
            results.append("Facebook (Manual)")

        if 'twitter' in platforms:
            # Twitter manual posting option
            st.info("🐦 Twitter Posting Instructions:")
            st.write("1. Save the image and copy the caption")
            st.write("2. Go to Twitter")
            st.write("3. Create a new tweet with the saved image")
            st.write("4. Paste the caption and tweet!")
            
            with open(image_path, 'rb') as img_file:
                btn = st.download_button(
                    label="📥 Download Image for Twitter",
                    data=img_file,
                    file_name="twitter_post.jpg",
                    mime="image/jpeg"
                )
            # Trim caption to Twitter's limit
            twitter_caption = caption[:280] if len(caption) > 280 else caption
            st.text_area("📝 Caption for Twitter", value=twitter_caption, height=100)
            results.append("Twitter (Manual)")

        if 'instagram' in platforms:
            # Instagram manual posting option
            st.info("📸 Instagram Posting Instructions:")
            st.write("1. Save the image and copy the caption")
            st.write("2. Open Instagram app on your phone")
            st.write("3. Create a new post with the saved image")
            st.write("4. Paste the caption and share!")
            
            with open(image_path, 'rb') as img_file:
                btn = st.download_button(
                    label="📥 Download Image for Instagram",
                    data=img_file,
                    file_name="instagram_post.jpg",
                    mime="image/jpeg"
                )
            st.text_area("📝 Caption for Instagram", value=caption, height=100)
            results.append("Instagram (Manual)")

        # Add a QR code for easy mobile transfer
        st.info("📱 Quick Transfer to Phone:")
        st.write("Scan this QR code with your phone to download the image:")
        qr_code = generate_qr_code(image_path)
        st.image(qr_code, width=200)
            
        return results
        
    except Exception as e:
        st.error(f"Error preparing social media content: {str(e)}")
        return results

def generate_qr_code(file_path):
    """Generate QR code for easy file transfer"""
    import qrcode
    from PIL import Image

    # Create a temporary URL or use file path
    data = f"file://{os.path.abspath(file_path)}"
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    
    qr_image = qr.make_image(fill_color="black", back_color="white")
    return qr_image

# Create necessary directories
os.makedirs('./config/', exist_ok=True)
os.makedirs('./scraped_images/', exist_ok=True)
os.makedirs('./.cache/', exist_ok=True)

if __name__ == "__main__":
    main()
