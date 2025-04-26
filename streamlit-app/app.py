import csv
import io
import json
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urljoin

import facebook
import numpy as np
import openai
import pandas as pd
import replicate
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance
from rembg import remove
from social_media import SocialMediaManager
import boto3
import base64
import botocore

import config

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="GenovateAI - Social Media Content Generator",
    page_icon="🎯",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Create necessary directories at startup
os.makedirs('scraped_images', exist_ok=True)
os.makedirs('.cache', exist_ok=True)
os.makedirs('../embedding-generation', exist_ok=True)

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None
    st.warning("⚠️ OpenAI API key not found. Content will be generated using templates only. To use AI-powered generation, please add your OpenAI API key to the .env file.")

# Initialize social media manager
social_media = SocialMediaManager()

# Add config validation at startup
missing_keys = config.validate_config()
if missing_keys:
    st.error(f"Missing required API keys: {', '.join(missing_keys)}")
    st.info("Please set up your API keys in the .env file")
    st.stop()

# Initialize AWS credentials directly from .env
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_REGION', 'us-east-1')

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
os.environ['AWS_DEFAULT_REGION'] = region

# Initialize Bedrock client
try:
    # Use default credential provider chain
    bedrock_client = boto3.client('bedrock-runtime')
    
    # Test the connection with a simple request
    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-image-generator-v2:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": "test",
                    "width": 512,
                    "height": 512,
                    "numberOfImages": 1
                }
            })
        )
        st.sidebar.success("✅ AWS credentials loaded and verified")
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        
        st.sidebar.error(f"❌ AWS Error: {error_code}")
        if error_code == 'UnrecognizedClientException':
            st.sidebar.warning("""
            Authentication failed. Please verify:
            1. Your AWS credentials are correct
            2. Your AWS account has access to Bedrock
            3. Bedrock is available in your region
            4. You've accepted the Bedrock terms of service
            """)
        elif error_code == 'AccessDeniedException':
            st.sidebar.warning("Access denied. Please check your IAM permissions.")
        else:
            st.sidebar.warning(f"Error: {error_msg}")
        
        bedrock_client = None
except Exception as e:
    st.sidebar.error(f"❌ AWS Error: {str(e)}")
    bedrock_client = None

# Load data mapping
@st.cache_data
def load_data_mapping():
    try:
        mapping_file = '../embedding-generation/data_mapping.csv'
        if not os.path.exists(mapping_file):
            # Create a default mapping file
            default_data = {
                'file_name': ['default_post'],
                'text': ['Discover our amazing products! Experience quality and style combined. #quality #style #trending'],
                'local_path': ['./images/default.jpg']
            }
            df = pd.DataFrame(default_data)
            df.to_csv(mapping_file, index=False)
            return df
        return pd.read_csv(mapping_file)
    except Exception as e:
        st.error(f"Error loading data mapping: {str(e)}")
        return pd.DataFrame()

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
        
        # Filter posts by product type and tone
        filtered_posts = []
        for _, row in df.iterrows():
            if (product_type.lower() in row['file_name'].lower() or 
                product_type.lower() in row['text'].lower()):
                filtered_posts.append({
                    'text': row['text'],
                    'file_name': row['file_name'],
                    'local_path': row['local_path']
                })
        
        # Sort by relevance (basic implementation)
        filtered_posts = sorted(filtered_posts, 
                             key=lambda x: x['text'].lower().count(tone.lower()), 
                             reverse=True)
        
        return filtered_posts[:n]
        
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

def generate_template_content(product, brand, tone, description=None):
    """Generate content using templates as fallback"""
    templates = {
        'Professional': [
            "Introducing our premium {product}: {description} Experience the excellence of {brand}.",
            "Elevate your lifestyle with our exceptional {product}. {description} #premium #{brand}",
            "Discover unmatched quality with our {product}. {description} Join the {brand} family."
        ],
        'Casual': [
            "Check out our amazing {product}! {description} Love it? Get yours from {brand} 💫",
            "Hey there! Meet our awesome {product}. {description} #lifestyle #{brand}",
            "Looking for the perfect {product}? We've got you covered! {description} #{brand}"
        ],
        'Enthusiastic': [
            "🌟 INCREDIBLE ALERT! Our {product} is here! {description} Join the {brand} revolution!",
            "✨ Get ready to be AMAZED! Our {product} is a game-changer! {description} #{brand}",
            "🔥 The wait is OVER! Experience our fantastic {product}! {description} #{brand}"
        ],
        'Sophisticated': [
            "Indulge in excellence with our curated {product}. {description} A {brand} masterpiece.",
            "Refined elegance meets innovation: our {product}. {description} #{brand} #luxury",
            "Experience the epitome of sophistication with our {product}. {description} #{brand}"
        ]
    }
    
    # Default to Professional if tone not found
    tone_templates = templates.get(tone, templates['Professional'])
    
    # Select a template randomly
    import random
    template = random.choice(tone_templates)
    
    # Format the template
    content = template.format(
        product=product,
        description=description if description else "",
        brand=brand.replace(" ", "")  # Remove spaces for hashtag
    )
    
    # Add some relevant hashtags
    hashtags = [
        f"#{brand.replace(' ', '')}",
        f"#{product.replace(' ', '')}",
        "#lifestyle",
        "#quality",
        "#premium"
    ]
    
    # Add hashtags to content
    content += "\n\n" + " ".join(hashtags[:5])
    
    return content

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
    """Generate content using OpenAI API with fallback to templates"""
    try:
        if client and openai_api_key:
            prompt = f"""Create a social media post for:
            Product: {product}
            Brand: {brand}
            Tone: {tone}
            Description: {description if description else 'A high-quality ' + product}
            
            The post should:
            1. Be attention-grabbing
            2. Match the {tone} tone
            3. Include 3-5 relevant hashtags
            4. Be optimized for social media
            5. Include emojis where appropriate
            6. Be under 200 characters (not counting hashtags)
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional social media content creator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as api_error:
                st.warning(f"API error: {str(api_error)}. Using template-based generation.")
                return generate_template_content(product, brand, tone, description)
        else:
            return generate_template_content(product, brand, tone, description)
            
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return generate_template_content(product, brand, tone, description)

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

def get_titan_ai_request_body(prompt, negative_prompt="", style="realistic"):
    """Generate request body for Titan image generation"""
    seed = random.randint(0, 2147483647)
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": negative_prompt,
            "width": 1024,
            "height": 576,
            "numberOfImages": 1
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "premium",
            "cfgScale": 8,
            "seed": seed,
            "style": style
        }
    }
    return json.dumps(body)

def generate_background(prompt, style="realistic"):
    """Generate background using Amazon Bedrock's Titan Image Generator"""
    try:
        body = get_titan_ai_request_body(prompt, style=style)
        
        response = bedrock_client.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v2:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Debug the response
        st.sidebar.write("Response body:", response_body)
        
        if "images" in response_body and response_body["images"]:
            return base64_to_image(response_body["images"][0])
        else:
            st.error("No images found in response")
            st.sidebar.write("Full response:", response_body)
            return None
    except Exception as e:
        st.error(f"Error generating background: {str(e)}")
        st.sidebar.write("Error details:", str(e))
        return None

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        if not base64_string:
            st.error("Empty base64 string received")
            return None
            
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        st.error(f"Error converting base64 to image: {str(e)}")
        st.sidebar.write("Base64 string:", base64_string[:100] + "..." if base64_string else "None")
        return None

# Call this when the app starts
ensure_backgrounds_exist()

def add_custom_background(image, background_image):
    """Add a custom background image to the product image with smart positioning"""
    try:
        if background_image is None:
            st.error("No background image was generated")
            return add_background(image, (255, 255, 255))  # Fallback to white background
            
        # Ensure both images are in RGBA mode
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        if background_image.mode != 'RGBA':
            background_image = background_image.convert('RGBA')
        
        # Get dimensions
        product_width, product_height = image.size
        bg_width, bg_height = background_image.size
        
        # Calculate the scale factor to make the product image occupy about 60% of the background height
        target_height = int(bg_height * 0.6)
        scale_factor = target_height / product_height
        new_width = int(product_width * scale_factor)
        new_height = target_height
        
        # Resize product image while maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate position to place the product (centered horizontally, slightly lower vertically)
        x_position = (bg_width - new_width) // 2
        y_position = int((bg_height - new_height) * 0.6)  # Place product in lower third
        
        # Create a new blank image with the background size
        final_image = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
        
        # Paste background first
        final_image.paste(background_image, (0, 0))
        
        # Paste the product image on top with its alpha channel as mask
        final_image.paste(image, (x_position, y_position), image)
        
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
        # Background Options
        st.write("🎯 Background Options")
        background_option = st.radio(
            "Choose background type:",
            ["Solid Color", "Search Background Images", "Generate AI Background"],
            horizontal=True
        )
        
        if background_option == "Solid Color":
            bg_color = st.color_picker("Select background color", "#FFFFFF")
            bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        elif background_option == "Generate AI Background":
            # AI Background Generation UI
            st.subheader("🎨 Generate AI Background")
            bg_prompt = st.text_input(
                "Describe the background you want:",
                placeholder="e.g., luxury store interior with Christmas decorations, soft lighting"
            )
            
            style_options = {
                "Realistic": "realistic",
                "Digital Art": "digital-art",
                "Cinematic": "cinematic",
                "Anime": "anime"
            }
            
            bg_style = st.selectbox("Choose style:", list(style_options.keys()))
            
            if bg_prompt and st.button("🎨 Generate Background", type="primary"):
                with st.spinner("🎨 Generating AI background..."):
                    background_image = generate_background(
                        bg_prompt,
                        style=style_options[bg_style]
                    )
                    if background_image:
                        st.image(background_image, caption="Generated Background", use_column_width=True)
                        st.session_state.selected_background = background_image
                        st.success("✨ Background generated successfully!")
                    else:
                        st.error("Failed to generate background. Please try again.")
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
            
            # Store the processed image in session state
            st.session_state.processed_image = final_image
            st.image(final_image, caption='Final Image', use_column_width=True)
            
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

# Instagram Posting Section
if 'current_image' in st.session_state and 'generated_content' in st.session_state:
    st.divider()
    st.subheader("📱 Prepare for Instagram")
    
    social_col1, social_col2 = st.columns([0.6, 0.4])
    
    with social_col1:
        st.write("Preview Post")
        try:
            # Display the processed image from session state
            if 'processed_image' in st.session_state:
                st.image(st.session_state.processed_image, use_container_width=True)
                
                # Add format selection
                format_type = st.selectbox(
                    "Choose Instagram Format",
                    ["All Formats", "Square (1:1)", "Portrait (4:5)", "Landscape (1.91:1)"]
                )
                
                format_mapping = {
                    "Square (1:1)": "square",
                    "Portrait (4:5)": "portrait",
                    "Landscape (1.91:1)": "landscape",
                    "All Formats": "all"
                }
                
                # Optimize image for selected format
                optimized = optimize_for_instagram(
                    st.session_state.processed_image,
                    format_mapping[format_type]
                )
                
                if optimized:
                    if format_type == "All Formats":
                        cols = st.columns(len(optimized))
                        for idx, (format_name, img) in enumerate(optimized.items()):
                            with cols[idx]:
                                st.image(img, caption=f"{format_name.title()} Format")
                                
                                # Add download button for each format
                                img_bytes = BytesIO()
                                img.save(img_bytes, format='JPEG', quality=95)
                                st.download_button(
                                    f"Download {format_name.title()}",
                                    img_bytes.getvalue(),
                                    file_name=f"instagram_{format_name}.jpg",
                                    mime="image/jpeg"
                                )
                    else:
                        # Show single selected format
                        format_name = format_mapping[format_type]
                        st.image(optimized[format_name], caption=f"{format_name.title()} Format")
                        
                        # Add download button
                        img_bytes = BytesIO()
                        optimized[format_name].save(img_bytes, format='JPEG', quality=95)
                        st.download_button(
                            f"Download {format_name.title()}",
                            img_bytes.getvalue(),
                            file_name=f"instagram_{format_name}.jpg",
                            mime="image/jpeg"
                        )
            
            # Display caption
            caption = st.text_area("Caption", value=st.session_state.generated_content, height=100)
            
            # Hashtag input with better organization
            st.subheader("Hashtag Management")
            hashtag_col1, hashtag_col2 = st.columns(2)
            
            with hashtag_col1:
                suggested_hashtags = st.text_input(
                    "Add Custom Hashtags",
                    placeholder="e.g., fashion, style, trending"
                )
            
            with hashtag_col2:
                popular_hashtags = ["#photography", "#instagood", "#fashion", "#art", "#beautiful", 
                                  "#nature", "#photooftheday", "#love", "#instagram", "#picoftheday"]
                selected_popular = st.multiselect(
                    "Add Popular Hashtags",
                    popular_hashtags
                )
            
            # Combine and format hashtags
            custom_hashtags = [f"#{tag.strip('#')}" for tag in suggested_hashtags.split(',') if tag.strip()] if suggested_hashtags else []
            all_hashtags = custom_hashtags + selected_popular
            
            if all_hashtags:
                formatted_hashtags = " ".join(all_hashtags)
                st.markdown("### Final Caption with Hashtags")
                final_caption = f"{caption}\n\n{formatted_hashtags}"
                st.code(final_caption, language=None)
                
                # Add copy button for caption
                st.button("📋 Copy Caption", on_click=lambda: st.write("Caption copied to clipboard!"))
                
        except Exception as e:
            st.error(f"Error in preview section: {str(e)}")
    
    with social_col2:
        st.write("Instagram Export")
        
        # Show posting tips in an expander
        with st.expander("📝 Posting Tips"):
            st.markdown("""
            ### Best Practices
            - Post during peak engagement hours
            - Use relevant hashtags (max 30)
            - Write engaging captions
            - Tag relevant accounts
            - Add location when applicable
            
            ### Optimal Posting Times
            - Monday to Friday: 11 AM - 1 PM
            - Weekends: 10 AM - 11 AM
            - Best days: Tuesday and Thursday
            """)

# Add footer
st.markdown("""
<div class="footer">Made with ❤️ using OpenAI</div>
""", unsafe_allow_html=True)

def optimize_for_instagram(image, format_type="all"):
    """Optimize image for Instagram formats"""
    try:
        # Convert to RGB if image is in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        formats = {
            'square': (1080, 1080),
            'portrait': (1080, 1350),
            'landscape': (1080, 608)
        }
        
        if format_type != "all":
            if format_type not in formats:
                return None
            formats = {format_type: formats[format_type]}
        
        optimized_images = {}
        
        for format_name, dimensions in formats.items():
            # Create a copy of the image
            img_copy = image.copy()
            
            # Calculate aspect ratio
            aspect_ratio = dimensions[0] / dimensions[1]
            
            # Get current dimensions
            width, height = img_copy.size
            current_ratio = width / height
            
            # Calculate new dimensions maintaining aspect ratio
            if current_ratio > aspect_ratio:
                new_width = int(height * aspect_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / aspect_ratio)
            
            # Crop to center
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            img_copy = img_copy.crop((left, top, right, bottom))
            
            # Resize to target dimensions
            img_copy = img_copy.resize(dimensions, Image.LANCZOS)
            
            optimized_images[format_name] = img_copy
            
        return optimized_images
    except Exception as e:
        st.error(f"Error optimizing image: {str(e)}")
        return None







