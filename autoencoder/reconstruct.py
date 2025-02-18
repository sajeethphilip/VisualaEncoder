import os
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from PIL import Image
from autoencoder.model import Autoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv
from autoencoder.data_loader import load_dataset_config

def reconstruct_from_latent(model, latent_csv_path, device):
    """Reconstruct image from saved latent representation"""
    # Load latent representation
    latent_1d = load_1d_latent_from_csv(latent_csv_path).to(device)

    # Reconstruct through decoder
    with torch.no_grad():
        decoded_flat = model.latent_mapper.inverse_map(latent_1d)
        decoded_volume = decoded_flat.view(1, 512, 1, 1)
        reconstructed = model.decoder(decoded_volume)
        reconstructed = model.adaptive_upsample(reconstructed)

    return reconstructed


def load_model(checkpoint_path, device):
    """Load the trained autoencoder model."""
    model = Autoencoder(latent_dim=128, embedding_dim=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def preprocess_image(image_path, device):
    """Preprocess the input image."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),        # Convert to tensor and normalize to [0, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor

def enhance_features(latent, embedding, model, enhancement_factor=2.0):
    """
    Enhance decisive features in the latent space based on the embedding.
    - latent: Latent space tensor.
    - embedding: Embedded tensor.
    - model: Trained autoencoder model.
    - enhancement_factor: Factor to enhance decisive features.
    """
    # Identify decisive features (e.g., regions with high embedding values)
    decisive_features = torch.abs(embedding).mean(dim=1, keepdim=True)

    # Enhance decisive features in the latent space
    enhanced_latent = latent + enhancement_factor * decisive_features

    # Reconstruct the image
    with torch.no_grad():
        reconstructed_image = model.decoder(enhanced_latent)

    return reconstructed_image

def save_reconstructed_image(image_tensor, dataset_name, filename="reconstructed.png"):
    """Save the reconstructed image to a folder."""
    # Create reconstructed images directory
    recon_dir = f"data/{dataset_name}/reconstructed_images"
    os.makedirs(recon_dir, exist_ok=True)

    # Convert tensor to PIL image
    image = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())

    # Save the image
    image_path = os.path.join(recon_dir, filename)
    image.save(image_path)
    print(f"Reconstructed image saved to {image_path}")

def reconstruct_folder(input_dir, checkpoint_path, dataset_name, config):
    """Reconstruct all images in a directory structure maintaining the hierarchy."""
    device = get_device()
    model = Autoencoder(config).to(device)

    # Load checkpoint
    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create base reconstruction directory
    recon_base_dir = f"data/{dataset_name}/reconstructed_images"
    os.makedirs(recon_base_dir, exist_ok=True)

    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding subdirectory in reconstruction folder
        relative_path = os.path.relpath(root, input_dir)
        recon_dir = os.path.join(recon_base_dir, relative_path)
        os.makedirs(recon_dir, exist_ok=True)

        print(f"Processing directory: {relative_path}")

        # Process each image in the current directory
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct paths
                input_path = os.path.join(root, file)
                recon_path = os.path.join(recon_dir, file)

                # Process image
                try:
                    # Preprocess and reconstruct
                    image_tensor = preprocess_image(input_path, device)
                    with torch.no_grad():
                        _, latent, embeddings = model(image_tensor)
                        enhanced_image = enhance_features(latent, embeddings, model)

                    # Save reconstructed image
                    image = transforms.ToPILImage()(enhanced_image.squeeze(0).cpu())
                    image.save(recon_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

    print(f"Reconstruction complete. Results saved in {recon_base_dir}")

# Modify the main reconstruction function to handle both single files and folders
def reconstruct_image(path, checkpoint_path, dataset_name, config):
    """Handle both single image and folder reconstruction."""
    if os.path.isdir(path):
        print(f"Processing directory: {path}")
        reconstruct_folder(path, checkpoint_path, dataset_name, config)
    else:
        print(f"Processing single image: {path}")
        # Original single image reconstruction code
        device = get_device()
        model = Autoencoder(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        image_tensor = preprocess_image(path, device)
        with torch.no_grad():
            _, latent, embeddings = model(image_tensor)
            enhanced_image = enhance_features(latent, embeddings, model)

        save_reconstructed_image(enhanced_image, dataset_name,
                               filename=os.path.basename(path))


# Example usage
if __name__ == "__main__":
    image_path = "path/to/input/image.png"  # Replace with the path to your input image
    checkpoint_path = "Model/checkpoint/Best_CIFAR10.pth"  # Replace with the path to your model checkpoint
    dataset_name = "CIFAR10"  # Replace with the dataset name
    reconstruct_image(image_path, checkpoint_path, dataset_name)
