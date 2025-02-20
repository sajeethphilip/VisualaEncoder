import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder,ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv,save_checkpoint
from autoencoder.utils import  load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv,save_batch_latents
from datetime import datetime
from tqdm import tqdm

def train_model(config):
    """
    Train the autoencoder model with comprehensive class distribution monitoring.

    Args:
        config (dict): Configuration dictionary containing model and training parameters
    """
    ## Setup and Initialization

    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Load dataset configuration
    dataset_config = config["dataset"]
    data_dir = os.path.join("data", dataset_config["name"], "train")

    ## Class Structure Verification

    # Get class folders
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"\nFound {len(class_folders)} classes:")

    # Display initial class distribution
    total_images = 0
    class_counts = {}
    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[class_name] = num_images
        total_images += num_images
        print(f"  • {class_name}: {num_images} images ({(num_images/total_images)*100:.1f}%)")

    ## Dataset Loading

    # Create dataset
    train_dataset = load_local_dataset(dataset_config["name"])
    print(f"\nDataset classes: {train_dataset.classes}")
    print(f"Class to idx mapping: {train_dataset.class_to_idx}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    ## Model Setup

    # Initialize model
    model = ModifiedAutoencoder(config, device=device)
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"],
               config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )

    # Loss function
    criterion_recon = nn.MSELoss()

    ## Checkpoint Management

    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load existing checkpoint if available
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(checkpoint_path):
        print(f"\nLoading existing checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))

    ## Training Loop

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0

    print("\nStarting training...")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_class_counts = {class_name: 0 for class_name in class_folders}

        # Progress bar for batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Update class distribution tracking
            for label in labels:
                class_name = train_dataset.classes[label.item()]
                epoch_class_counts[class_name] += 1

            # Move data to device
            images = images.to(device)

            # Forward pass
            reconstructed, latent_1d = model(images)

            # Compute loss and optimize
            loss = criterion_recon(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update metrics
            current_loss = loss.item()
            epoch_loss += current_loss

            # Update progress bar
            progress_bar.set_postfix({
                'loss': current_loss,
                'avg_loss': epoch_loss / (batch_idx + 1)
            })

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Average Loss: {avg_epoch_loss:.4f}")

        # Display class distribution for this epoch
        print("\nClass distribution this epoch:")
        total_samples = sum(epoch_class_counts.values())
        for class_name, count in epoch_class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  • {class_name}: {count} samples ({percentage:.1f}%)")

        # Checkpoint saving
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            model.cpu()
            save_checkpoint(model, epoch + 1, avg_epoch_loss, config, checkpoint_path)
            model.to(device)
            print(f"\nSaved best model checkpoint (loss: {best_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience:
            print(f"\nNo improvement for {patience} epochs. Training stopped.")
            break

    print("\nTraining complete!")
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
