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
    """Train the autoencoder model with enhanced class folder verification."""
    # Load dataset
    dataset_config = config["dataset"]
    data_dir = os.path.join("data", dataset_config["name"], "train")

    # Verify and log class structure
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"\nFound {len(class_folders)} class folders:")
    for idx, class_name in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  Class {idx}: {class_name} - {num_images} images")

    # Create dataset with class verification
    train_dataset = load_local_dataset(dataset_config["name"])
    print(f"\nDataset classes: {train_dataset.classes}")
    print(f"Class to idx mapping: {train_dataset.class_to_idx}")

    # Create a weighted sampler for balanced sampling
    class_counts = torch.tensor([class_counts[cls] for cls in train_dataset.classes])
    weights = 1.0 / class_counts
    sample_weights = weights[train_dataset.targets]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )


    # Modify the DataLoader to use the sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,  # Add sampler
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    # Verify batch contents - Modified to handle both string and tensor paths
    sample_batch_images, sample_batch_labels = next(iter(train_loader))
    class_counts = {}

    # Get paths from dataset.imgs if available
    if hasattr(train_dataset, 'imgs'):
        paths = [train_dataset.imgs[i][0] for i in range(len(sample_batch_images))]
    else:
        # If no imgs attribute, construct paths from classes and indices
        paths = [os.path.join(data_dir, train_dataset.classes[label.item()])
                for label in sample_batch_labels]

    for path in paths:
        class_name = os.path.basename(os.path.dirname(path))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\nSample batch class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")

    # Get device and ensure it's set before model creation
    device = get_device()
    print(f"\nUsing device: {device}")

    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Initialize model
    model = ModifiedAutoencoder(config, device=device)
    model = model.cpu()

    # Global variables
    start_epoch = 0
    best_loss = float("inf")

    # Try loading checkpoint
    try:
        if os.path.exists(checkpoint_path):
            print(f"Loading existing checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_loss = checkpoint.get("loss", float("inf"))
            print("Checkpoint loaded successfully")
        else:
            print("No existing checkpoint found. Starting fresh training...")
        model.latent_mapper._initialize_frequencies()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training...")
        model.latent_mapper._initialize_frequencies()

    model = model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )

    criterion_recon = nn.MSELoss()

    # Enhanced training loop with class verification
    epochs = config["training"]["epochs"] + start_epoch
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_class_counts = {class_name: 0 for class_name in class_folders}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Get full paths based on class labels
            if hasattr(train_dataset, 'imgs'):
                full_paths = [train_dataset.imgs[i][0] for i in range(len(images))]
            else:
                full_paths = [os.path.join(data_dir, train_dataset.classes[label.item()])
                            for label in labels]

            # Update class counts for monitoring
            for path in full_paths:
                class_name = os.path.basename(os.path.dirname(path))
                epoch_class_counts[class_name] += 1

            images = images.to(device)

            # Forward pass
            reconstructed, latent_1d = model(images)

            # Save batch latents with metadata
            batch_metadata = {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'timestamp': datetime.now().isoformat()
            }
            save_batch_latents(latent_1d, full_paths, dataset_config["name"], batch_metadata)

            # Compute loss and backprop
            loss = criterion_recon(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            num_batches += 1

            progress_bar.set_postfix({
                'loss': current_loss,
                'avg_loss': epoch_loss / num_batches,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Print epoch statistics with class distribution
        epoch_loss /= num_batches
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("\nClass distribution this epoch:")
        for class_name, count in epoch_class_counts.items():
            print(f"  {class_name}: {count} images")

        # Save checkpoint if loss improved
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            model.cpu()
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model, epoch + 1, epoch_loss, config, checkpoint_path)
            model.to(device)
            print(f"\nSaved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nNo improvement for {patience} epochs. Training stopped.")
                break

        if epoch + 1 >= epochs:
            user_input = input("\nModel still learning. Continue training? (y/n): ").lower()
            if user_input == 'y':
                epochs += 1
            else:
                break

    print("\nTraining complete.")
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
