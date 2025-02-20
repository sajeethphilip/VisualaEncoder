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
    """Train the autoencoder model with separate latent space generation."""

    # Initial setup
    device = get_device()
    dataset_config = config["dataset"]
    data_dir = os.path.join("data", dataset_config["name"], "train")

    # Create dataset and loader for training
    train_dataset = load_local_dataset(dataset_config["name"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    # Model setup
    model = ModifiedAutoencoder(config, device=device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )
    criterion_recon = nn.MSELoss()

    # Training loop
    print("\nStarting training...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion_recon(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch [{epoch + 1}] Average Loss: {avg_epoch_loss:.4f}")

        # After each epoch, generate latent space representations class by class
        print("\nGenerating latent space representations...")
        model.eval()

        # Create separate loaders for each class
        for class_idx, class_name in enumerate(train_dataset.classes):
            # Filter dataset for current class
            class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_idx]
            class_subset = torch.utils.data.Subset(train_dataset, class_indices)

            # Create non-shuffling loader for this class
            class_loader = DataLoader(
                class_subset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["training"]["num_workers"]
            )

            print(f"\nProcessing class: {class_name} ({len(class_subset)} images)")

            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(tqdm(class_loader, desc=f"Generating latents for {class_name}")):
                    # Get full paths for images in this batch
                    batch_start_idx = batch_idx * config["training"]["batch_size"]
                    batch_indices = class_indices[batch_start_idx:batch_start_idx + len(images)]
                    full_paths = [train_dataset.imgs[idx][0] for idx in batch_indices]

                    # Generate latent representations
                    images = images.to(device)
                    reconstructed, latent_1d = model(images)

                    # Save latent representations
                    batch_metadata = {
                        'epoch': epoch + 1,
                        'class': class_name,
                        'batch': batch_idx,
                        'timestamp': datetime.now().isoformat()
                    }
                    save_batch_latents(latent_1d, full_paths, dataset_config["name"], batch_metadata)

                    # Save reconstructed images if needed
                    # Add code here to save reconstructed images

        print(f"\nCompleted epoch {epoch + 1} with latent space generation")

    print("\nTraining complete!")
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
