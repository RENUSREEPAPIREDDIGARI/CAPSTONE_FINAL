import csv
import io
import json
import os
from datetime import datetime
from io import BytesIO

import openai
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image

# Initialize SambaNova client
client = OpenAI(
    api_key="635091b0-62d7-4e56-a89f-a8f37080e6a6",
    base_url="https://api.sambanova.ai/v1",
)

# Set page config
st.set_page_config(
    page_title="Social Media Content Generator",
    page_icon="icon.png",
    layout="wide"
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
            
            # Debug information
            #st.write(f"Attempting to load image from: {full_path}")
            
            if os.path.exists(full_path):
                return Image.open(full_path)
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

# Main app header
st.title('Social Media Content Generator')

# Input section
st.subheader("Product Details")
col1, col2 = st.columns([0.6, 0.4])
with col1:
    product = st.text_input("Enter your product (e.g., bag, perfume, watch):")
    description = st.text_area("Product Description", 
                             placeholder="Enter a brief description of your product (e.g., A luxurious leather handbag with gold accents)",
                             help="This description will help generate more relevant content")

# Brand Selection and Tone
with col2:
    brand = st.selectbox("Select Brand", ["Luxury Brand", "Casual Brand", "Sports Brand"])
    tone = st.selectbox("Select Tone", ["Professional", "Casual", "Enthusiastic", "Sophisticated"])

# Image Upload Section
st.subheader("Upload a product image")
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None and product and brand and tone:
    # Display uploaded image
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        if st.button("Generate Content"):
            with st.spinner("Analyzing and generating content..."):
                # Find similar posts
                similar_posts = find_similar_posts(product, tone)
                
                if similar_posts:
                    st.subheader("📝 Similar Posts Analysis")
                    
                    # Create tabs for similar posts
                    tabs = st.tabs([f"Similar Post {i+1}" for i in range(len(similar_posts))])
                    
                    for i, (tab, post) in enumerate(zip(tabs, similar_posts)):
                        with tab:
                            col1, col2 = st.columns([0.6, 0.4])
                            
                            with col1:
                                st.markdown("### Content")
                                st.write(post['text'])
                                
                                # Display file info
                                st.caption(f"Source: {post['file_name']}")
                                if post.get('s3_path'):
                                    st.caption(f"Image: {post['s3_path'].split('/')[-1]}")
                            
                            with col2:
                                # Load and display the similar post image
                                image = load_image_from_path(post)
                                if image:
                                    st.image(image, use_container_width=True)
                                else:
                                    st.warning("Image not found")
                    
                    # Generate new content based on patterns and LLM
                    st.subheader("🎯 Generated Content")
                    new_content = generate_content_with_llm(similar_posts, product, brand, tone, description)
                    if new_content:
                        # Make content selectable and copyable
                        st.text_area("New Post", value=new_content, height=150, key="new_post", disabled=False, 
                                   help="Click to select and copy the generated content")
                        
                        # Add a copy button
                        st.markdown("""
                        <style>
                        .stTextArea textarea {
                            font-size: 1rem;
                            line-height: 1.5;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Show a note about AI generation
                        st.info("✨ This content was generated by AI using SambaNova's LLM, taking inspiration from similar successful posts while maintaining originality.")
                else:
                    st.warning("No similar posts found. Try different product type or tone.")







