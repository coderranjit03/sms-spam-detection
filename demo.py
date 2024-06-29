import streamlit as st
import os

# Make sure the path is correct
image_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'cover.jpeg')

# Ensure the image path exists
if os.path.exists(image_path):
    # Set the background image using inline CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url('https://github.com/Sibikrish3000/starter-telegram-bot/blob/main/assets/cover.jpeg?raw=true');
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error(f"Image not found at path: {image_path}")

# Your other Streamlit code
st.title("Your Streamlit App with Background Image")
st.write("This is an example Streamlit application with a background image.")
