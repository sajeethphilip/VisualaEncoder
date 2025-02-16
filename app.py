import streamlit as st
import torch
from autoencoder.model import Autoencoder
from autoencoder.utils import preprocess_image, postprocess_image, save_latent_as_csv, load_latent_from_csv
from interface.region_marking import mark_region
from interface.sliders import create_sliders
from interface.visualization import display_image

# Load the trained autoencoder
model = Autoencoder()
dataset_name = "CIFAR10"  # Replace with the dataset name used during training
checkpoint_path = f"Model/checkpoint/Best_{dataset_name}.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
model.eval()

# Streamlit app
st.title("Autoencoder Region Manipulation")

# Option 1: Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Preprocess the image
    image_tensor = preprocess_image(uploaded_image)
    print(f"Image tensor shape: {image_tensor.shape}")  # Debug: Check tensor shape
    
    # Convert tensor to PIL image and display
    pil_image = postprocess_image(image_tensor)
    display_image(pil_image, "Uploaded Image")

    # Encode the image to get the latent vector
    _, latent_vector = model(image_tensor)

    # Save latent vector as CSV
    if st.button("Save Latent Space as CSV"):
        save_latent_as_csv(latent_vector, dataset_name, "latent.csv")
        st.success(f"Latent space saved to data/{dataset_name}/latent.csv")

    # Mark region of interest
    st.write("Mark the region of interest:")
    roi = mark_region(pil_image.convert("RGB"))

    # Identify latent dimensions for the region (placeholder logic)
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
    reconstructed_pil_image = postprocess_image(reconstructed_image)
    display_image(reconstructed_pil_image, "Reconstructed Image")

# Option 2: Load latent vector from CSV and reconstruct image
st.write("---")
st.header("Reconstruct Image from Latent CSV")
uploaded_csv = st.file_uploader("Upload a latent CSV file", type=["csv"])

if uploaded_csv is not None:
    # Load latent vector from CSV
    latent_vector = load_latent_from_csv(uploaded_csv)
    latent_vector = latent_vector.unsqueeze(0)  # Add batch dimension

    # Reconstruct the image
    with torch.no_grad():
        reconstructed_image = model.decoder(latent_vector)
    reconstructed_pil_image = postprocess_image(reconstructed_image)
    display_image(reconstructed_pil_image, "Reconstructed Image from CSV")
    
