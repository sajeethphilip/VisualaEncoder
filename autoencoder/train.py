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
from tqdm import tqdm
from colorama import init, Fore, Back, Style
def train_model(config):
    """Train the autoencoder model with persistent header and organized progress display."""

    # Get terminal size and setup display areas
    terminal_size = os.get_terminal_size()
    terminal_height = terminal_size.lines

    # Display header once at the top
    print("\033[2J\033[H")  # Clear screen
    display_header()

    # Calculate progress display position
    header_height = 12  # Adjust based on your header size
    progress_start = header_height + 2

    # Initial setup
    device = get_device()
    dataset_config = config["dataset"]
    train_dataset = load_local_dataset(dataset_config["name"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    # Model setup
    model = ModifiedAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )
    criterion_recon = nn.MSELoss()

    # Training loop setup
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0
    best_loss = float("inf")

    # Create checkpoint directory
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

    def update_progress_display(epoch, batch, total_batches, loss, avg_loss):
        print(f"\033[{progress_start}H\033[K")  # Clear progress area

        # Progress bar
        bar_width = 50
        progress = batch / total_batches
        filled = int(bar_width * progress)
        bar = "█" * filled + "-" * (bar_width - filled)

        # Display training information
        print(f"Epoch: {epoch + 1}/{epochs}")
        print(f"Progress: [{bar}] {progress:.1%}")
        print(f"Batch: {batch}/{total_batches}")
        print(f"Current Loss: {loss:.6f}")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Best Loss: {best_loss:.6f}")
        print(f"Patience: {patience_counter}/{patience}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            reconstructed, latent_1d = model(images)
            loss = criterion_recon(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)

            # Update progress
            update_progress_display(epoch, batch_idx + 1, num_batches, loss.item(), avg_loss)

            # Save batch latents
            batch_paths = [train_dataset.imgs[i][0] for i in range(batch_idx * config["training"]["batch_size"],
                         min((batch_idx + 1) * config["training"]["batch_size"], len(train_dataset)))]
            save_batch_latents(latent_1d, batch_paths, dataset_config["name"])

        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            save_checkpoint(model, epoch + 1, avg_epoch_loss, config,
                          os.path.join(config["training"]["checkpoint_dir"], "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\033[{progress_start + 8}H\033[K")
                print("Early stopping triggered!")
                break

    print(f"\033[{progress_start + 8}H\033[K")
    print("Training complete!")
    return model


def save_final_representations(model, loader, device, dataset_name):
    """Save the final latent space and embeddings."""
    latent_space = []
    embeddings_space = []

    with torch.no_grad():
        for batch in loader:
            images, _ = batch
            images = images.to(device)  # Move images to device
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
