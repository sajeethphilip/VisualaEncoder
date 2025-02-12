import streamlit as st

def create_sliders(region_dims):
    slider_values = []
    for dim in region_dims:
        slider_values.append(st.slider(f"Latent Dim {dim}", -1.0, 1.0, 0.0))
    return slider_values
