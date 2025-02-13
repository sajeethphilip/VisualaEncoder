import streamlit as st
import torch
from autoencoder.model import Autoencoder
from autoencoder.utils import preprocess_image, postprocess_image
from interface.region_marking import mark_region
from interface.sliders import create_sliders
from interface.visualization import display_image

# Load the trained autoencoder
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit app
st.title("Autoencoder Region Manipulation")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Preprocess the image
    image_tensor = preprocess_image(uploaded_image)
    print(f"Image tensor shape: {image_tensor.shape}")  # Debug: Check tensor shape
    display_image(postprocess_image(image_tensor), "Uploaded Image")

    # Mark region of interest
    st.write("Mark the region of interest:")
    roi = mark_region(postprocess_image(image_tensor).convert("RGB"))

    # Identify latent dimensions for the region (placeholder logic)
    _, latent_vector = model(image_tensor)
    region_dims = list(range(10))  # Example: Use first 10 dimensions

    # Create sliders for the region
    slider_values = create_sliders(region_dims)

    # Modify latent vector based on slider values
    modified_latent_vector = latent_vector.clone()
    for dim, value in zip(region_dims, slider_values):
        modified_latent_vector[0, dim] += value

    # Reconstruct the image
    with torch.no_grad():
        reconstructed_image = model.decoder(modified_latent_vector)
    display_image(postprocess_image(reconstructed_image), "Reconstructed Image")
