import io
import json
import os
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rembg import remove

# Set page config
st.set_page_config(
    page_title="Social Media Content Generator",
    page_icon="icon.png",
    layout="wide"
)

# Function to generate scene description
def generate_scene_description(image, product, theme):
    """Generate a detailed scene description"""
    return f"A fully decorated {theme} setting, featuring a {product} in the foreground. The scene creates an inviting ambiance with professional lighting and careful composition, highlighting the product while maintaining a natural and appealing atmosphere."

# Function to generate recommendations
def generate_recommendations(product, brand, tone):
    """Generate posting recommendations"""
    return f"This post effectively captures the {brand} spirit by combining {product} with seasonal elements. The composition and lighting create an inviting ambiance. To replicate this success, incorporate: 1) High-quality product photography, 2) Seasonal themes that resonate with your audience, 3) Professional lighting to highlight product features."

# Initialize session state for storing generated images and posts
if 'historical_posts' not in st.session_state:
    st.session_state.historical_posts = []
if 'similar_posts' not in st.session_state:
    st.session_state.similar_posts = []

# Content templates for different tones
TEMPLATES = {
    "Professional": [
        "Introducing our exquisite {product} - where luxury meets functionality. Perfect for those who appreciate {brand}'s commitment to excellence. ✨ #LuxuryLifestyle #Premium{product} #{brand}Style",
        "Experience unparalleled quality with our latest {product}. Crafted with precision and designed for distinction. A true {brand} masterpiece. 🌟 #Premium #Excellence #{brand}Quality",
        "Elevate your style with {brand}'s signature {product}. Where sophistication meets innovation. 💫 #Innovation #Style #{brand}Collection"
    ],
    "Casual": [
        "Check out our amazing new {product}! Perfect for every day and every way. {brand}'s got your style covered! 😊 #StyleGoals #MustHave #{brand}Vibes",
        "Love your style? You'll love our new {product}! {brand}'s latest addition to your everyday awesome. ✌️ #StyleInspo #CoolFinds #{brand}Life",
        "Hey style lovers! Meet your new favorite {product} from {brand}. Because why settle for less? 🌟 #StyleUpdate #NewArrivals #{brand}Style"
    ],
    "Enthusiastic": [
        "🎉 INCREDIBLE NEWS! Our game-changing {product} is here! Get ready to experience {brand}'s most exciting release yet! 🚀 #GameChanger #MustHave #{brand}Revolution",
        "✨ AMAZING ALERT! Fall in love with our spectacular new {product}! {brand}'s latest masterpiece will blow your mind! 💫 #Amazing #WowFactor #{brand}Magic",
        "🌟 GET EXCITED! Your favorite {brand} just dropped this incredible {product}! Trust us, you need this in your life! 🔥 #Excitement #NewLaunch #{brand}Wow"
    ],
    "Sophisticated": [
        "Discover the artistry of our meticulously crafted {product}. A testament to {brand}'s legacy of excellence. 🎭 #Artistry #Luxury #{brand}Heritage",
        "Immerse yourself in refined elegance with our distinguished {product}. {brand}'s dedication to sophistication personified. ✨ #Refinement #Elegance #{brand}Luxury",
        "Experience the epitome of luxury with our carefully curated {product}. {brand}'s commitment to unparalleled excellence. 🌟 #LuxuryLifestyle #Premium #{brand}Elite"
    ]
}

# Background images for different themes
BACKGROUND_IMAGES = {
    "christmas": "https://images.pexels.com/photos/1708601/pexels-photo-1708601.jpeg",
    "holiday": "https://images.pexels.com/photos/1708601/pexels-photo-1708601.jpeg",
    "modern": "https://images.pexels.com/photos/1939485/pexels-photo-1939485.jpeg",
    "luxury": "https://images.pexels.com/photos/3932930/pexels-photo-3932930.jpeg"
}

def get_background_image(theme):
    """Get appropriate background image based on theme"""
    # Default to modern background if theme not found
    url = BACKGROUND_IMAGES.get("modern")
    
    # Check theme keywords
    theme_lower = theme.lower()
    for key in BACKGROUND_IMAGES:
        if key in theme_lower:
            url = BACKGROUND_IMAGES[key]
            break
    
    try:
        response = requests.get(url)
        background = Image.open(BytesIO(response.content))
        return background
    except Exception as e:
        st.error(f"Error loading background: {str(e)}")
        return None

def remove_background_with_padding(image):
    """Remove background and add padding"""
    try:
        # Remove background
        image_no_bg = remove(image)
        
        # Convert to RGBA if not already
        if image_no_bg.mode != 'RGBA':
            image_no_bg = image_no_bg.convert('RGBA')
        
        # Add padding (20% of the original size)
        width, height = image_no_bg.size
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        new_size = (width + 2*padding_x, height + 2*padding_y)
        
        padded_image = Image.new('RGBA', new_size, (0, 0, 0, 0))
        padded_image.paste(image_no_bg, (padding_x, padding_y), image_no_bg)
        
        return padded_image
    except Exception as e:
        st.error(f"Error removing background: {str(e)}")
        return image

def enhance_product_image(image):
    """Enhance the product image while preserving quality"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance image quality
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        return image
    except Exception as e:
        st.error(f"Error enhancing product image: {str(e)}")
        return image

def enhance_image_with_background(product_image, background_theme):
    """Enhance image and combine with themed background"""
    try:
        # First enhance the product image
        enhanced_product = enhance_product_image(product_image)
        
        # Remove background and add padding
        product_no_bg = remove_background_with_padding(enhanced_product)
        
        # Get background image
        background = get_background_image(background_theme)
        if background is None:
            return None
            
        # Resize background while maintaining aspect ratio
        target_width = 1920  # Full HD width
        ratio = target_width / background.width
        target_height = int(background.height * ratio)
        background = background.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Calculate product size (25% of background width)
        product_width = int(background.width * 0.25)
        ratio = product_width / product_no_bg.width
        product_height = int(product_no_bg.height * ratio)
        product_no_bg = product_no_bg.resize((product_width, product_height), Image.Resampling.LANCZOS)
        
        # Create final composition
        final_image = background.copy()
        
        # Position product in bottom right area
        x_position = background.width - product_width - int(background.width * 0.1)  # 10% padding from right
        y_position = background.height - product_height - int(background.height * 0.1)  # 10% padding from bottom
        
        # Add subtle shadow effect
        shadow = Image.new('RGBA', product_no_bg.size, (0, 0, 0, 0))
        shadow_strength = 50  # Adjust shadow intensity (0-255)
        shadow.paste((0, 0, 0, shadow_strength), (0, 0, product_width, product_height))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Paste shadow and product
        final_image.paste(shadow, (x_position+5, y_position+5), shadow)  # Offset shadow slightly
        final_image.paste(product_no_bg, (x_position, y_position), product_no_bg)
        
        return final_image
    except Exception as e:
        st.error(f"Error creating final image: {str(e)}")
        return None

# Main app header
st.title('Professional Social Media Content Generator')

# Create input section
st.subheader("Background Settings")
background_theme = st.text_input("Enter background theme (e.g., Christmas tree, holiday decoration, warm lights):")

st.subheader("Product Details")
product = st.text_input("Enter your product object, such as: bag, car, perfume etc.")

# Brand Selection and Tone
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Select Brand", ["Luxury Brand", "Casual Brand", "Sports Brand"])
    
with col2:
    tone = st.selectbox("Select Tone", ["Professional", "Casual", "Enthusiastic", "Sophisticated"])

# Image Upload Section
st.subheader("Upload a product image")
uploaded_file = st.file_uploader("Drag and drop file here", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display original and enhanced images
    col3, col4 = st.columns(2)
    
    with col3:
        initial_image = Image.open(uploaded_file)
        # Remove background from initial image
        initial_image_no_bg = remove_background_with_padding(initial_image)
        st.image(initial_image_no_bg, caption='Initial Image (Background Removed)', use_container_width=True)
    
    with col4:
        if st.button("Enhance Image"):
            with st.spinner("Enhancing image and adding background..."):
                enhanced_image = enhance_image_with_background(initial_image, background_theme)
                if enhanced_image:
                    st.image(enhanced_image, caption='Enhanced Image', use_container_width=True)
                    
                    # Create a container for the final post
                    final_post_container = st.container()
                    
                    with final_post_container:
                        # Display initial post text
                        st.subheader("Initial post text")
                        initial_post = f"Elevate your style with {brand}'s latest masterpiece. Crafted with timeless elegance and superior quality, this exquisite {product} embodies unique craftsmanship. Indulge in the epitome of sophistication and let it be your constant companion for life's grandest moments. ✨ #LuxuryBrand #TimelessElegance #HolidayLuxury #GiftOfTheYear"
                        st.text_area("Initial Post", value=initial_post, height=100, key="initial_post", disabled=True)
                        
                        # Display enhanced image
                        st.image(enhanced_image, caption='Final Image', use_container_width=True)
                        
                        # Display final post text
                        st.subheader("final_post_text")
                        final_post = f"Embrace the magic of the season with {brand}'s latest masterpiece ✨ Adorned with exquisite craftsmanship and timeless elegance, this sophisticated {product} is the epitome of refined style. Let it be your cherished companion as you create unforgettable memories amidst the twinkling lights and festive cheer. Indulge in the unparalleled luxury that elevates every moment. #LuxuryBrand #HolidayElegance #TimelessStyle"
                        st.text_area("Final Post", value=final_post, height=100, key="final_post", disabled=True)
                    
                    # Store the final post
                    if 'final_posts' not in st.session_state:
                        st.session_state.final_posts = []
                    
                    st.session_state.final_posts.append({
                        'initial_post': initial_post,
                        'final_post': final_post,
                        'enhanced_image': enhanced_image,
                        'product': product,
                        'brand': brand,
                        'theme': background_theme
                    })
                    
                    # Add download button for enhanced image
                    buf = BytesIO()
                    enhanced_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Enhanced Image",
                        data=byte_im,
                        file_name="enhanced_product_image.png",
                        mime="image/png"
                    )

                    # After generating enhanced image, add Similar Posts section
                    st.subheader("Similar Posts:")
                    with st.spinner("Retrieving Similar Posts..."):
                        # Store the current post in similar posts
                        similar_post = {
                            'image': enhanced_image,
                            'post_text': f"'Tis the season to be jolly! 🎄 Get cozy by the twinkling tree with a warm blanket and your favorite {product}. The perfect winter wonderland scene to unwind after a long day. #HolidayVibes #CozyLights #WinterEssentials",
                            'scene_description': generate_scene_description(enhanced_image, product, background_theme),
                            'recommendations': generate_recommendations(product, brand, tone)
                        }
                        st.session_state.similar_posts.append(similar_post)
                        
                        # Display similar posts
                        for post in st.session_state.similar_posts[-3:]:  # Show last 3 similar posts
                            st.image(post['image'], use_container_width=True)
                            with st.expander("Post Details"):
                                st.write("**Post Text:**", post['post_text'])
                                st.write("**Scene Description:**", post['scene_description'])
                                st.write("**Recommendations:**", post['recommendations'])

# Generate Content Section
st.subheader("Generate Content")
if st.button("Generate Content"):
    if product and brand:
        with st.spinner("Generating content..."):
            # Select a random template for the chosen tone
            import random
            template = random.choice(TEMPLATES[tone])
            
            # Generate content
            generated_content = template.format(
                product=product,
                brand=brand
            )
            
            # Store in historical posts
            st.session_state.historical_posts.append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'brand': brand,
                'content': generated_content,
                'tone': tone,
                'product': product
            })
            
            # Display generated content
            st.markdown("### Generated Post")
            st.write(generated_content)
            
            # Add download button
            st.download_button(
                label="Download Content",
                data=generated_content,
                file_name="social_media_content.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please enter product details and select a brand.")

# Historical Post Analysis
st.subheader("Historical post analysis")
if st.session_state.historical_posts:
    for idx, post in enumerate(reversed(st.session_state.historical_posts[-3:]), 1):
        with st.expander(f"Post_{idx}"):
            st.write("**Post Text:**", post['content'])
            st.write("**Scene Description:**")
            st.write(generate_scene_description(enhanced_image if 'enhanced_image' in locals() else None, post['product'], background_theme))
            st.write("**Recommendations:**")
            st.write(generate_recommendations(post['product'], post['brand'], post['tone']))
else:
    st.info("Generate some posts to see historical analysis")

# Add setup instructions
with st.expander("Setup Instructions"):
    st.write("""
    1. Install required dependencies:
       ```
       pip install -r requirements.txt
       ```
    2. Run the app:
       ```
       streamlit run app.py
       ```
    """)
