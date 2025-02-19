import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder
from autoencoder.data_loader import load_local_dataset, load_dataset_config
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv,save_checkpoint,load_checkpoint
from datetime import datetime
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder.model import ModifiedAutoencoder
from autoencoder.data_loader import load_local_dataset
from autoencoder.utils import get_device, save_1d_latent_to_csv

def train_model(config):
    """Train the autoencoder model using the provided configuration."""
    # Load dataset
    dataset_config = config["dataset"]
    train_dataset = load_local_dataset(dataset_config["name"])
    # Get feature dimensions from config
    feature_dims = config["model"]["feature_dims"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    device = get_device()
    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Initialize model
    model = ModifiedAutoencoder(config, device=device).to(device)

    # Global image counter for unique latent space saving
    global_image_counter = 0
    start_epoch = 0
    best_loss = float("inf")

    # Try loading checkpoint, but continue with fresh model if not found
    try:
        if os.path.exists(checkpoint_path):
            print(f"Loading existing checkpoint from {checkpoint_path}")
            model, start_epoch, best_loss = load_checkpoint(checkpoint_path, model, config)
        else:
            print("No existing checkpoint found. Starting fresh training...")
            model.latent_mapper._initialize_frequencies()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training...")
        model.latent_mapper._initialize_frequencies()

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )

    criterion_recon = nn.MSELoss()

    # Training loop
    epochs = config["training"]["epochs"] +start_epoch   # Always let there be epoch more counts to go.
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            images, _ = batch
            images = images.to(device)

            # Forward pass
            reconstructed, latent_1d = model(images)

            # Save latent representation only if needed
            if config.get("save_latent_during_training", False):  # Make this configurable
                # Save latent space for each image in batch
                for idx in range(images.size(0)):
                    # Create a consistent image identifier
                    image_name = f"image_{batch_idx * config['training']['batch_size'] + idx}"
                    metadata = {
                        "batch_index": batch_idx,
                        "global_index": batch_idx * config['training']['batch_size'] + idx
                    }
                    save_1d_latent_to_csv(
                        latent_1d[idx],
                        image_name,
                        config["dataset"]["name"],
                        metadata,
                        mode="train"
                    )

            # Compute loss and backprop
            loss = criterion_recon(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({
                'loss': current_loss,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint if loss improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(
                model,
                epoch + 1,
                epoch_loss,
                config,
                checkpoint_path
            )
            print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Training stopped.")
                break

        # Ask user to continue if epochs completed
        if epoch + 1 >= epochs:
            user_input = input("Model still learning. Continue training? (y/n): ").lower()
            if user_input == 'y':
                epochs += 1
            else:
                break

    print("Training complete.")




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
