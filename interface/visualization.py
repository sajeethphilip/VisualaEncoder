import streamlit as st

def display_image(image, caption):
    """Display an image in the Streamlit app."""
    st.image(image, caption=caption, use_column_width=True)
