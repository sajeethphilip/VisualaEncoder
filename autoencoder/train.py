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
    # [Previous initialization code remains the same until optimizer definition]

    # Modified optimizer configuration with adaptive learning rate parameters
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

    # [Previous checkpoint loading code remains the same]

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



if __name__ == "__main__":
    # Example usage
    dataset_name = "mnist"  # Replace with the dataset name
    train_model(dataset_name)
