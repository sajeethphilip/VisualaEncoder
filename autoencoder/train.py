import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder, ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv, save_checkpoint, update_progress, display_confusion_matrix,ssim_loss,save_latent_space_for_epoch,postprocess_hdr_image
from autoencoder.utils import load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv, save_batch_latents, display_header,update_confusion_matrix,preprocess_hdr_image
from datetime import datetime
from tqdm import tqdm
from colorama import init, Fore, Back, Style

def draw_progress_box(epoch, batch, total_batches, loss, avg_loss, progress_start, epochs):
    """
    Draw a progress box with training metrics at a fixed position on the screen.

    Args:
        epoch: Current epoch number
        batch: Current batch number
        total_batches: Total number of batches
        loss: Current loss value
        avg_loss: Average loss so far
        progress_start: Starting row for the progress box
        epochs: Total number of epochs
    """
    # Box drawing characters
    top_left = "╔"
    top_right = "╗"
    bottom_left = "╚"
    bottom_right = "╝"
    horizontal = "═"
    vertical = "║"

    # Box dimensions
    box_width = 53
    box_height = 8

    # Progress bar
    progress = batch / total_batches
    bar_width = box_width - 15
    filled = int(bar_width * progress)
    bar = "█" * filled + "-" * (bar_width - filled)

    # Move cursor to the fixed starting position for the progress box
    print(f"\033[{progress_start};0H", end="")

    # Draw top border
    print(f"{Fore.GREEN}{top_left}{horizontal * box_width}{top_right}{Style.RESET_ALL}")

    # Epoch and progress information
    print(f"{Fore.GREEN}{vertical}{Style.RESET_ALL} Epoch: {epoch + 1}/{epochs} {' ' * (box_width - 14)}{Style.RESET_ALL}") #{Fore.GREEN}{vertical}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{vertical}{Style.RESET_ALL} [{bar}] {progress:.1%} {' ' * 5}{Style.RESET_ALL}") #{Fore.GREEN}{vertical}{Style.RESET_ALL}")

    # Training metrics
    print(f"{Fore.GREEN}{vertical}{Style.RESET_ALL} Batch: {batch}/{total_batches} {' ' * (box_width - 16)}{Style.RESET_ALL}") #{Fore.GREEN}{vertical}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{vertical}{Style.RESET_ALL} Current Loss: {loss:.6f} {' ' * (box_width - 24)}{Style.RESET_ALL}") #{Fore.GREEN}{vertical}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{vertical}{Style.RESET_ALL} Average Loss: {avg_loss:.6f} {' ' * (box_width - 24)}{Style.RESET_ALL}") #{Fore.GREEN}{vertical}{Style.RESET_ALL}")

    # Draw bottom border
    print(f"{Fore.GREEN}{bottom_left}{horizontal * box_width}{bottom_right}{Style.RESET_ALL}")

def train_model(config):
    """Train the autoencoder model with configurable loss functions."""
    # Check if config is None
    if config is None:
        raise ValueError("Config is None. Please provide a valid configuration dictionary.")

    # Print config for debugging
    print("Config:", config)

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
    device = get_device()  # Automatically detect and set the device (CPU, GPU, or TPU)
    print(f"Using device: {device}")

    # Ensure dataset config exists
    if "dataset" not in config:
        raise ValueError("Config is missing 'dataset' key. Please provide a valid dataset configuration.")

    dataset_config = config["dataset"]
    train_dataset = load_local_dataset(dataset_config["name"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    # Model setup
    model = ModifiedAutoencoder(config).to(device)  # Move model to the correct device

    # Check for an existing model checkpoint
    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}")
        model, _, _ = load_checkpoint(checkpoint_path, model, config, device)
    else:
        print("No existing model found. Initializing a new model.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )

    # Define the MSE loss function
    criterion_mse = nn.MSELoss().to(device)

    # Automatically adjust loss functions based on wavelet decomposition
    if config["multiscale"]["enabled"]:
        # Disable MSE and prioritize SSIM when wavelet decomposition is used
        loss_functions = {
            "mse": {
                "enabled": False,  # Disable MSE
                "weight": 0.0
            },
            "ssim": {
                "enabled": True,   # Enable SSIM
                "weight": 1.0     # Full weight to SSIM
            }
        }
    else:
        # Use default loss functions from config
        loss_functions = config["training"].get("loss_functions", {
            "mse": {
                "enabled": True,
                "weight": 1.0
            },
            "ssim": {
                "enabled": False,
                "weight": 0.0
            }
        })

    # Set up loss functions based on the adjusted configuration
    mse_enabled = loss_functions.get("mse", {}).get("enabled", False)
    ssim_enabled = loss_functions.get("ssim", {}).get("enabled", False)

    if mse_enabled:
        mse_weight = loss_functions["mse"].get("weight", 1.0)

    if ssim_enabled:
        ssim_weight = loss_functions["ssim"].get("weight", 1.0)

    # Training loop setup
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0
    best_loss = float("inf")

    # Initialize confusion matrix (not used in this version)
    num_classes = len(train_dataset.classes)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32).to(device)  # Move to device

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move images and labels to the correct device
            images = images.to(device)
            labels = labels.to(device)

            # Preprocess images with wavelet decomposition
            original_images, reconstructed_images = [], []
            for img in images:
                original, reconstructed = preprocess_hdr_image(img.cpu().numpy(), config)
                original_images.append(original)
                reconstructed_images.append(reconstructed)
            original_images = torch.stack(original_images).to(device)
            reconstructed_images = torch.stack(reconstructed_images).to(device)

            # Forward pass: Get predicted images
            predicted_images, latent = model(original_images)

            # Compute loss: Compare predicted images with reconstructed wavelet-decomposed images
            loss = criterion_mse(predicted_images, reconstructed_images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)

            # Update similarity metrics (per class)
            class_metrics, num_groups = update_confusion_matrix(original_images, predicted_images, labels, confusion_matrix, screen_group=0)

            # Update progress display with box
            draw_progress_box(epoch, batch_idx + 1, num_batches, loss.item(), avg_loss, progress_start=14, epochs=epochs)

        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches

        # Save latent space representations for all images at the end of each epoch
        save_latent_space_for_epoch(model, train_loader, device, dataset_config["name"])

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
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
