import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder,ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv,save_checkpoint,update_progress,display_confusion_matrix,create_confusion_matrix
from autoencoder.utils import  load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv,save_batch_latents,display_header
from datetime import datetime
from tqdm import tqdm

def train_model(config):
    """Train the autoencoder model with improved confusion matrix and positioning."""
    device = get_device()
    print(f"Using device: {device}")

    # Get terminal dimensions
    terminal_height = os.get_terminal_size().lines
    header_height = 10  # Reserve space for header

    # Display header
    display_header()

    # Dataset setup
    dataset_config = config["dataset"]
    dataset = load_local_dataset(dataset_config["name"])
    class_names = dataset.classes
    num_classes = len(class_names)

    # Initialize confusion matrix
    confusion_matrix = create_confusion_matrix(num_classes).to(device)

    # Model setup
    model = ModifiedAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"]
    )
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        
        for batch_idx, (images, labels) in enumerate(dataset):
            # Ensure data is on correct device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            reconstructed, latent_1d = model(images)
            
            # Update confusion matrix
            update_confusion_matrix(images, reconstructed, labels, confusion_matrix)
            
            # Compute loss and optimize
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save latent representations
            batch_metadata = {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'loss': loss.item(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Get image paths
            batch_paths = [dataset.imgs[i][0] for i in range(len(images))]
            
            # Save latents
            save_batch_latents(latent_1d, batch_paths, dataset_config["name"], batch_metadata)
            
            # Update displays with static positioning
            display_confusion_matrix(confusion_matrix, class_names, terminal_height, header_height)
            update_progress_bar(epoch + 1, batch_idx + 1, len(dataset), loss.item(), terminal_height)
            
    return model



def save_final_representations(model, loader, device, dataset_name):
    """Save the final latent space and embeddings."""
    latent_space = []
    embeddings_space = []

    with torch.no_grad():
        for batch in loader:
            images, _ = batch
            images = images.to(device)
            _, latent, embeddings = model(images)
            latent_space.append(latent.cpu())
            embeddings_space.append(embeddings.cpu())

    # Concatenate all batches
    latent_space = torch.cat(latent_space, dim=0)
    embeddings_space = torch.cat(embeddings_space, dim=0)

    # Save representations
    save_latent_space(latent_space, dataset_name, "latent.pkl")
    save_embeddings_as_csv(embeddings_space, dataset_name, "embeddings.csv")


if __name__ == "__main__":
    # Example usage
    dataset_name = "mnist"  # Replace with the dataset name
    train_model(dataset_name)
