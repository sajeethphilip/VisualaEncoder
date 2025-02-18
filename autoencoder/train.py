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
def get_model(config):
    """Initialize the appropriate autoencoder model based on config."""
    model_selection = config["model"]["model_selection"]

    # Check if complex model is enabled
    if model_selection["complex"]["enabled"]:
        model = ComplexAutoencoder(
            config=model_selection["complex"]
        )
    else:
        # Default to simple model
        model_selection["simple"]["enabled"] = True  # Ensure simple is enabled
        model = SimpleAutoencoder(
            latent_dim=model_selection["simple"]["latent_dim"],
            conv_layers=model_selection["simple"]["conv_layers"],
            use_batch_norm=model_selection["simple"]["use_batch_norm"]
        )

    return model


def train_model(config):
    """Train the autoencoder model using the provided configuration."""
    # Load dataset
    dataset_config = config["dataset"]
    train_dataset = load_local_dataset(dataset_config["name"])
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    device = get_device()

    # Initialize model
    #model = Autoencoder(config).to(device)
    model = get_model(config).to(device)

    # Define optimizer with adaptive learning rate parameters
    optimizer_config = config["model"]["optimizer"]
    initial_lr = config["model"]["learning_rate"]
    max_lr = initial_lr * 10
    increase_factor = 1.2
    min_improvement = 0.005
    patience_window = 5

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=optimizer_config["weight_decay"],
        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
        eps=optimizer_config["epsilon"]
    )

    # Define loss functions
    reconstruction_weight = config["model"]["autoencoder_config"]["reconstruction_weight"]
    feature_weight = config["model"]["autoencoder_config"]["feature_weight"]
    criterion_recon = nn.MSELoss()
    criterion_feature = nn.MSELoss()

    # Loss history and learning rate tracking
    loss_history = []
    current_lr = initial_lr
    best_loss = float("inf")
    start_epoch = 0

    def adjust_learning_rate(current_loss):
        nonlocal current_lr
        if len(loss_history) >= patience_window:
            past_loss = loss_history[-patience_window]
            improvement = (past_loss - current_loss) / past_loss

            if improvement > min_improvement:
                new_lr = min(current_lr * increase_factor, max_lr)
                if new_lr > current_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    current_lr = new_lr
                    print(f"Increasing learning rate to {current_lr:.6f}")

    # Training loop
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar and collect loss
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_losses.append(current_loss)
            progress_bar.set_postfix({
                'loss': current_loss,
                'lr': current_lr,
                '[recon]': loss_recon.item(),
                '[feature]': loss_feature.item()
            })

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)

        # Adjust learning rate based on performance
        adjust_learning_rate(epoch_loss)

        print(f"Epoch [{epoch + 1}/{config['training']['epochs']}], "
              f"Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.6f}")

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
                "config": config,
                "current_lr": current_lr
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
