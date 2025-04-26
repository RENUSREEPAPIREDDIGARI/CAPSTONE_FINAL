# Social Media Content Generator

A Streamlit-based web application that helps you generate engaging social media content for various platforms including Instagram, Facebook, Twitter, and LinkedIn.

## Features

- Generate platform-specific content for multiple social media platforms
- Choose from different tones: Professional, Casual, Humorous, Inspirational, and Educational
- Customize content with brand voice and target audience
- Upload and preview images
- Get platform-specific hashtag suggestions
- Download generated content as text file

## Try the App Online

You can try the app directly in your browser:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_CLOUD_URL)

## Local Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/social-media-content-generator.git
cd social-media-content-generator
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter your post topic, select the platform and tone, and provide brand guidelines

4. Upload an image if needed

5. Click "Generate Content" to create your social media post

## Requirements

- Python 3.7+
- Streamlit
- Pillow

## Deployment Options

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Go to https://share.streamlit.io/
3. Sign in with your GitHub account
4. Click "New app"
5. Select your forked repository
6. Select the main branch
7. Select `app.py` as the main file
8. Click "Deploy"

### Heroku
1. Create a Heroku account
2. Install Heroku CLI
3. Run the following commands:
```bash
heroku create your-app-name
git push heroku main
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
