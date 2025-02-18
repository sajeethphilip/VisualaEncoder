import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder
from autoencoder.data_loader import load_local_dataset, load_dataset_config
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    device = get_device()

    # Initialize model
    model = ModifiedAutoencoder(config, device=device).to(device)

    # Define optimizer
    optimizer_config = config["model"]["optimizer"]
    initial_lr = config["model"]["learning_rate"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=optimizer_config["weight_decay"],
        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
        eps=optimizer_config["epsilon"]
    )

    # Define loss function
    criterion_recon = nn.MSELoss()

    # Training loop
    best_loss = float("inf")
    start_epoch = 0

    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['training']['epochs']}",
            leave=False
        )

        for batch in progress_bar:
            images, _ = batch
            images = images.to(device)

            # Forward pass
            reconstructed, latent_1d = model(images)

            # Save latent representation for each image
            for idx, img in enumerate(images):
                image_name = f"image_{idx}"
                save_1d_latent_to_csv(latent_1d[idx], image_name, config["dataset"]["name"])

            # Compute reconstruction loss
            loss = criterion_recon(reconstructed, images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar
            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({
                'loss': current_loss,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)

        print(f"Epoch [{epoch + 1}/{config['training']['epochs']}], "
              f"Loss: {epoch_loss:.4f}, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint if best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_dir = config["training"]["checkpoint_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": config
            }, checkpoint_path)

            print(f"Saved best model checkpoint to {checkpoint_path}")

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
