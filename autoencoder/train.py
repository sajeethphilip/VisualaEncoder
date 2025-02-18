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

def train_model(config):
    """Train the autoencoder model using the provided configuration."""
    # Load dataset
    dataset_config = config["dataset"]
    train_dataset = load_local_dataset(dataset_config["name"])
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    device = get_device()
    # Initialize model
    model = Autoencoder(config).to(device)

     optimizer_config = config["model"]["optimizer"]
    initial_lr = config["model"]["learning_rate"]
    max_lr = initial_lr * 10  # Allow learning rate to grow up to 10x
    increase_factor = 1.2     # Aggressive growth factor
    min_improvement = 0.005   # Minimum improvement threshold
    patience_window = 5       # Window to measure improvement

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=optimizer_config["weight_decay"],
        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
        eps=optimizer_config["epsilon"]
    )

    # Loss history for adaptive learning rate
    loss_history = []
    current_lr = initial_lr

    def adjust_learning_rate(current_loss):
        nonlocal current_lr
        if len(loss_history) >= patience_window:
            # Calculate improvement over patience window
            past_loss = loss_history[-patience_window]
            improvement = (past_loss - current_loss) / past_loss

            if improvement > min_improvement:
                # Positive improvement - increase learning rate
                new_lr = min(current_lr * increase_factor, max_lr)
                if new_lr > current_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    current_lr = new_lr
                    print(f"Increasing learning rate to {current_lr:.6f}")

                # Modified training loop
                for epoch in range(start_epoch, config["training"]["epochs"]):
                    model.train()
                    epoch_loss = 0.0
                    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}", leave=False)

                    batch_losses = []
                    for batch in progress_bar:
                        images, _ = batch
                        images = images.to(device)

                        # Forward pass
                        reconstructed, latent, embedding = model(images)

                        # Compute losses
                        loss_recon = criterion_recon(reconstructed, images) * reconstruction_weight
                        loss_feature = criterion_feature(embedding, torch.zeros_like(embedding)) * feature_weight
                        loss = loss_recon + loss_feature

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()

                        # Add gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        optimizer.step()

                        # Update progress bar and collect loss
                        current_loss = loss.item()
                        epoch_loss += current_loss
                        batch_losses.append(current_loss)
                        progress_bar.set_postfix(loss=current_loss, lr=current_lr)

                    # Calculate average epoch loss
                    epoch_loss /= len(train_loader)
                    loss_history.append(epoch_loss)

                    # Adjust learning rate based on performance
                    adjust_learning_rate(epoch_loss)

                    print(f"Epoch [{epoch + 1}/{config['training']['epochs']}], "
                          f"Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.6f}")

                    # Save model checkpoint if it's the best so far
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save({
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                            "config": config,
                            "current_lr": current_lr
                        }, checkpoint_path)
                        print(f"Saved best model checkpoint to {checkpoint_path}")

                print("Training complete.")
def train_model1(dataset_name):
    """Train the autoencoder model."""
    # Load dataset and config
    config = load_dataset_config(dataset_name)
    dataset = load_local_dataset(dataset_name)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    batch_size = min(64, len(train_dataset) // 10)  # Adjust batch size based on dataset size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model with correct parameters from config
    device = get_device()
    model = Autoencoder(
        in_channels=config['in_channels'],
        input_size=config['input_size'],
        latent_dim=config['input_size'][0] * 2,  # Dynamic latent dimension
        embedding_dim=config['input_size'][0]     # Dynamic embedding dimension
    ).to(device)

    # Define loss and optimizer
    criterion_recon = nn.MSELoss()  # Reconstruction loss
    criterion_embed = nn.MSELoss()  # Embedding loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training parameters
    epochs = 50
    patience = 5  # Early stopping patience
    best_loss = float("inf")
    epochs_without_improvement = 0

    # Create checkpoint directory
    checkpoint_dir = os.path.join("Model", "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_loss = 0.0

        for batch in progress_bar:
            images, _ = batch
            images = images.to(device)

            # Normalize images using dataset statistics
            if hasattr(config, "mean") and hasattr(config, "std"):
                images = transforms.Normalize(config["mean"], config["std"])(images)

            optimizer.zero_grad()

            # Forward pass
            reconstructed, latent, embeddings = model(images)

            # Compute losses
            loss_recon = criterion_recon(reconstructed, images)
            loss_embed = criterion_embed(embeddings, torch.zeros_like(embeddings))
            loss = loss_recon + 0.1 * loss_embed  # Weighted combination

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Calculate average training loss
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, _ = batch
                images = images.to(device)

                # Normalize validation images
                if hasattr(config, "mean") and hasattr(config, "std"):
                    images = transforms.Normalize(config["mean"], config["std"])(images)

                reconstructed, latent, embeddings = model(images)
                loss = criterion_recon(reconstructed, images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

        # Early stopping and model saving
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0

            # Save the best model with configuration
            checkpoint_path = os.path.join(checkpoint_dir, f"Best_{dataset_name}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    # Save final latent space and embeddings
    model.eval()
    save_final_representations(model, train_loader, device, dataset_name)

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
