from torchvision import transforms
import csv
import torch
import os
import csv
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import csv
import torch

def save_embeddings_as_csv(embeddings, dataset_name, filename="embeddings.csv"):
    """Save the embedded tensors as a flattened 1D CSV file."""
    # Create data directory
    data_dir = f"data/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Flatten and convert to numpy array
    embeddings = embeddings.flatten().detach().cpu().numpy()
    
    # Save as CSV
    csv_path = os.path.join(data_dir, filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(embeddings)
    
    print(f"Embeddings saved to {csv_path}")
    
def visualize_embeddings(embeddings, labels, title="Embedding Space"):
    """Visualize the embedding space using t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10")
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def get_device():
    """Detect and return the best available device (CPU, GPU, or TPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon GPUs
        return torch.device("mps")
    elif torch.xpu.is_available():  # For Intel GPUs
        return torch.device("xpu")
    else:
        return torch.device("cpu")
        
def save_latent_as_csv(latent_vector, dataset_name, filename="latent.csv"):
    """Save the latent vector as a flattened 1D CSV file."""
    # Create data directory
    data_dir = f"data/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Flatten and convert to numpy array
    latent_vector = latent_vector.flatten().detach().numpy()
    
    # Save as CSV
    csv_path = os.path.join(data_dir, filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(latent_vector)
    
    print(f"Latent space saved to {csv_path}")
    
def postprocess_image(image_tensor):
    """Convert a tensor back to a PIL image."""
    print(f"Input tensor shape: {image_tensor.shape}")  # Debug: Check input tensor shape
    
    if image_tensor.dim() == 4:  # If batch dimension exists
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
    print(f"Tensor shape after removing batch dimension: {image_tensor.shape}")  # Debug
    
    if image_tensor.shape[0] > 4:  # Check for invalid number of channels
        raise ValueError(f"Invalid number of channels: {image_tensor.shape[0]}. Expected 1, 3, or 4.")
    
    # Convert tensor to PIL image
    image = transforms.ToPILImage()(image_tensor)
    return imagem(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def postprocess_image(image_tensor):
    """Convert a tensor back to a PIL image."""
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)  # Remove batch dimension and rearrange
    image = transforms.ToPILImage()(image_tensor)
    return image

def load_latent_from_csv(filename="latent.csv"):
    """Load the latent vector from a CSV file."""
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        latent_vector = next(reader)  # Read the first row
    latent_vector = torch.tensor([float(x) for x in latent_vector], dtype=torch.float32)
    return latent_vector
