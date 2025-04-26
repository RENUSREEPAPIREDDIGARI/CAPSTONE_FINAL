import io
import os

import numpy as np
import rembg
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter


def enhance_image_for_social(input_image, platform="instagram", remove_bg=True, 
                           brightness=1.1, contrast=1.2, saturation=1.1, sharpness=1.2):
    """
    Enhance an image for social media posting directly within Streamlit
    
    Args:
        input_image: PIL Image object
        platform: Social media platform (instagram, facebook, twitter, linkedin)
        remove_bg: Whether to remove the background
        brightness/contrast/saturation/sharpness: Enhancement factors
    
    Returns:
        Enhanced PIL Image
    """
    # Platform-specific dimensions (width, height)
    platform_sizes = {
        "instagram": (1080, 1080),  # Square post
        "instagram_story": (1080, 1920),  # Instagram story
        "facebook": (1200, 630),    # Facebook post
        "twitter": (1200, 675),     # Twitter post
        "linkedin": (1200, 627)     # LinkedIn post
    }
    
    img = input_image.copy()
    
    # Remove background if requested
    if remove_bg:
        try:
            img = rembg.remove(img)
        except Exception as e:
            st.warning(f"Error removing background: {e}. Continuing with original image.")
    
    # Get target size
    if platform in platform_sizes:
        target_size = platform_sizes[platform]
    else:
        target_size = platform_sizes["instagram"]
    
    # Resize image while maintaining aspect ratio
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider than target ratio
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller than target ratio
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    
    # Resize image
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new blank image with target dimensions
    new_img = Image.new("RGBA", target_size, (255, 255, 255, 0))
    
    # Calculate position to center the image
    left = (target_size[0] - new_width) // 2
    top = (target_size[1] - new_height) // 2
    
    # Paste the resized image onto the blank canvas
    new_img.paste(img, (left, top), mask=img if img.mode == 'RGBA' else None)
    
    # Apply enhancements
    if new_img.mode != 'RGBA':
        new_img = new_img.convert('RGBA')
    
    # Convert to RGB for enhancements (removing alpha channel)
    rgb_img = Image.new('RGB', new_img.size, (255, 255, 255))
    rgb_img.paste(new_img, mask=new_img.split()[3] if new_img.mode == 'RGBA' else None)
    
    # Apply enhancements
    rgb_img = ImageEnhance.Brightness(rgb_img).enhance(brightness)
    rgb_img = ImageEnhance.Contrast(rgb_img).enhance(contrast)
    rgb_img = ImageEnhance.Color(rgb_img).enhance(saturation)
    rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(sharpness)
    
    # Apply slight blur to smooth edges
    rgb_img = rgb_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Convert back to RGBA if the original was RGBA
    if new_img.mode == 'RGBA':
        # Get alpha channel from original
        alpha = new_img.split()[3]
        r, g, b = rgb_img.split()
        rgb_img = Image.merge('RGBA', (r, g, b, alpha))
    
    return rgb_img

# Example Streamlit integration in your app
def add_image_enhancement_section():
    st.header("Enhance Generated Image")
    
    # Assuming you have an image in session state or uploaded
    if 'generated_image' in st.session_state:
        image = st.session_state.generated_image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image)
        
        # Image enhancement options
        st.subheader("Enhancement Options")
        platform = st.selectbox("Target Platform", 
                              ["instagram", "instagram_story", "facebook", "twitter", "linkedin"],
                              index=0)
        
        remove_bg = st.checkbox("Remove Background", value=True)
        
        st.subheader("Fine-tune Image")
        brightness = st.slider("Brightness", 0.5, 1.5, 1.1, 0.1)
        contrast = st.slider("Contrast", 0.5, 1.5, 1.2, 0.1)
        saturation = st.slider("Saturation", 0.5, 1.5, 1.1, 0.1)
        sharpness = st.slider("Sharpness", 0.5, 1.5, 1.2, 0.1)
        
        if st.button("Enhance Image"):
            # Convert PIL Image or numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            enhanced_image = enhance_image_for_social(
                image, 
                platform=platform,
                remove_bg=remove_bg,
                brightness=brightness,
                contrast=contrast, 
                saturation=saturation,
                sharpness=sharpness
            )
            
            st.session_state.enhanced_image = enhanced_image
            
            # Save the enhanced image
            output_path = f"streamlit-app/enhanced_images/enhanced_{platform}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            enhanced_image.save(output_path)
            
            with col2:
                st.subheader("Enhanced Image")
                st.image(enhanced_image)
                st.success(f"Enhanced image saved to {output_path}")
                
                # Download button
                buf = io.BytesIO()
                enhanced_image.save(buf, format="PNG")
                btn = st.download_button(
                    label="Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name=f"enhanced_{platform}.png",
                    mime="image/png"
                )