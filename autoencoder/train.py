import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder, ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv, save_checkpoint, update_progress, display_confusion_matrix
from autoencoder.utils import load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv, save_batch_latents, display_header,update_confusion_matrix
from datetime import datetime
from tqdm import tqdm
from colorama import init, Fore, Back, Style

def train_model(config):
    """Train the autoencoder model with boxed progress display and confusion matrix."""

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

    # Initialize confusion matrix
    num_classes = len(train_dataset.classes)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def draw_progress_box(epoch, batch, total_batches, loss, avg_loss):
        """Draw a green box around progress information."""
        # Move cursor to progress area
        print(f"\033[{progress_start}H")

        # Box drawing characters
        top_left = "╔"
        top_right = "╗"
        bottom_left = "╚"
        bottom_right = "╝"
        horizontal = "═"
        vertical = "║"

        # Box dimensions
        width = 60
        height = 8
        progress = batch / total_batches

        # Format progress bar with exact spacing
        bar_width = width - 20  # Account for brackets and percentage
        filled = int(bar_width * progress)
        bar = "█" * filled + "-" * (bar_width - filled)

        print(f"{Fore.GREEN}╔{'═' * width}╗{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} Epoch: {epoch + 1}/{epochs:<{width-10}}{Fore.GREEN}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} [{bar}] {progress:.1%}{' ' * (width-len(bar)-8)}{Fore.GREEN}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} Batch: {batch}/{total_batches:<{width-15}}{Fore.GREEN}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} Current Loss: {loss:.6f:<{width-20}}{Fore.GREEN}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} Average Loss: {avg_loss:.6f:<{width-20}}{Fore.GREEN}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}║{Style.RESET_ALL} Best Loss: {best_loss:.6f:<{width-20}}{Fore.GREEN}║{Style.RESET_ALL}")
        print(f"{Fore.GREEN}╚{'═' * width}╝{Style.RESET_ALL}")

    # Initialize best losses
    best_batch_loss = float("inf")
    best_epoch_loss = float("inf")
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            model=model.to(device)
            reconstructed, _ = model(images)
            loss = criterion_recon(reconstructed, images)
            # Update best batch loss
            if loss.item() < best_batch_loss:
                best_batch_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)

            # Update confusion matrix
            update_confusion_matrix(images, reconstructed, labels, confusion_matrix)

            # Update progress display with box and confusion matrix
            draw_progress_box(epoch, batch_idx + 1, num_batches,
                            loss.item(), avg_loss, best_batch_loss,best_epoch_loss)

        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches

        if avg_epoch_loss < best_epoch_loss:
            best_lepoch_oss = avg_epoch_loss
            patience_counter = 0
            save_checkpoint(model, epoch + 1, avg_epoch_loss, config,
                          os.path.join(config["training"]["checkpoint_dir"], "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\033[{progress_start + 20}H\033[K")
                print("Early stopping triggered!")
                break

    print(f"\033[{progress_start + 20}H\033[K")
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
