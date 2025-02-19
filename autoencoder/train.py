import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder,ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv,save_checkpoint,load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv
from datetime import datetime
from tqdm import tqdm

def train_model(config):
    """Train the autoencoder model using the provided configuration."""
    # Load dataset
    dataset_config = config["dataset"]
    train_dataset = load_local_dataset(dataset_config["name"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # Get device and ensure it's set before model creation
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Initialize model
    model = ModifiedAutoencoder(config, device=device)

    # Initialize model weights on CPU first
    model = model.cpu()

    # Global variables
    start_epoch = 0
    best_loss = float("inf")

    # Try loading checkpoint
    try:
        if os.path.exists(checkpoint_path):
            print(f"Loading existing checkpoint from {checkpoint_path}")
            # Load checkpoint to CPU first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_loss = checkpoint.get("loss", float("inf"))

            print("Checkpoint loaded successfully")

            # Ensure frequencies are loaded
            if hasattr(model.latent_mapper, 'frequencies'):
                print(f"Loaded frequencies shape: {model.latent_mapper.frequencies.shape}")
            else:
                print("Initializing frequencies...")
                model.latent_mapper._initialize_frequencies()
        else:
            print("No existing checkpoint found. Starting fresh training...")
            model.latent_mapper._initialize_frequencies()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training...")
        model.latent_mapper._initialize_frequencies()

    # Move model to device after loading checkpoint
    model = model.to(device)
    print(f"Model moved to {device}")

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
    epochs = config["training"]["epochs"] + start_epoch
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # Dictionary to store latent representations and image paths
        latent_dict = {}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            images, paths = batch
            images = images.to(device)  # Move images to device

            # Forward pass
            reconstructed, latent_1d = model(images)

            # Handle paths
            if isinstance(paths, torch.Tensor):
                paths = [str(path.item()) for path in paths]
            else:
                paths = [str(path) for path in paths]

            # Store latent representations
            for idx in range(images.size(0)):
                image_path = paths[idx]
                latent_dict[image_path] = latent_1d[idx].detach().cpu()

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

        # Save latent representations
        for image_path, latent in latent_dict.items():
            subfolder = os.path.dirname(image_path)
            image_name = os.path.basename(image_path).split('.')[0]

            metadata = {
                "epoch": epoch + 1,
                "timestamp": datetime.now().isoformat()
            }
            save_1d_latent_to_csv(latent, image_name, dataset_config["name"], metadata)

        # Save checkpoint if loss improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0

            # Ensure model is on CPU for saving
            model.cpu()
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(
                model,
                epoch + 1,
                epoch_loss,
                config,
                checkpoint_path
            )
            # Move model back to original device
            model.to(device)
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
