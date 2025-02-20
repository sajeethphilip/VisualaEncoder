import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder, ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv, save_checkpoint, update_progress, display_confusion_matrix
from autoencoder.utils import load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv, save_batch_latents, display_header
from datetime import datetime

def train_model(config):
    """Train the autoencoder model."""
    device = get_device()
    print(f"Using device: {device}")

    # Dataset setup
    dataset_config = config["dataset"]
    dataset = load_local_dataset(dataset_config["name"])
    
    # Create DataLoader with proper batch size
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    class_names = dataset.classes
    num_classes = len(class_names)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)

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
        running_loss = 0.0
        confusion_matrix.zero_()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            reconstructed, latent_1d = model(images)
            
            # Update confusion matrix
            with torch.no_grad():
                pred_labels = torch.argmax(latent_1d, dim=1)
                for t, p in zip(labels, pred_labels):
                    confusion_matrix[t.long(), p.long()] += 1

            # Compute loss and optimize
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        # Display confusion matrix after each epoch
        display_confusion_matrix(confusion_matrix, class_names)

    return model




def save_final_representations(model, loader, device, dataset_name):
    """Save the final latent space and embeddings."""
    latent_space = []
    embeddings_space = []

    with torch.no_grad():
        for images, _ in loader:
            # Ensure images are tensors and on the correct device
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            images = images.to(device)
            
            # Forward pass
            reconstructed, latent, embeddings = model(images)
            latent_space.append(latent.cpu())
            embeddings_space.append(embeddings.cpu())

    # Concatenate all batches
    latent_space = torch.cat(latent_space, dim=0)
    embeddings_space = torch.cat(embeddings_space, dim=0)

    # Save representations
    save_latent_space(latent_space, dataset_name, "latent.pkl")
    save_embeddings_as_csv(embeddings_space, dataset_name, "embeddings.csv")

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    train_model(config)
