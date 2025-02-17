import os
import torch
from torchvision import transforms
from PIL import Image
from autoencoder.model import Autoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv
from autoencoder.data_loader import load_dataset_config

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

def reconstruct_image(image_path, checkpoint_path, dataset_name, enhancement_factor=2.0):
    """Reconstruct an image with enhanced features."""
    # Detect device
    device = get_device()

    config = load_dataset_config(dataset_name)

    # Initialize model
    model = Autoencoder(
        in_channels=config["in_channels"],
        input_size=config["input_size"],
        latent_dim=128,
        embedding_dim=64
    ).to(get_device())

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Preprocess the input image
    image_tensor = preprocess_image(image_path, device)

    # Generate latent space and embeddings
    with torch.no_grad():
        _, latent, embeddings = model(image_tensor)

    # Enhance decisive features and reconstruct the image
    enhanced_image = enhance_features(latent, embeddings, model, enhancement_factor)

    # Save the reconstructed image
    save_reconstructed_image(enhanced_image, dataset_name)

# Example usage
if __name__ == "__main__":
    image_path = "path/to/input/image.png"  # Replace with the path to your input image
    checkpoint_path = "Model/checkpoint/Best_CIFAR10.pth"  # Replace with the path to your model checkpoint
    dataset_name = "CIFAR10"  # Replace with the dataset name
    reconstruct_image(image_path, checkpoint_path, dataset_name)
