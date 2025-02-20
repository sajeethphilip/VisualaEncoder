import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder,ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv,save_checkpoint,update_progress,display_confusion_matrix
from autoencoder.utils import  load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv,save_batch_latents,display_header
from datetime import datetime
from tqdm import tqdm

def train_model(config):
    """Train the autoencoder model with separate latent space generation."""

    # Initial setup
    device = get_device()
    dataset_config = config["dataset"]
    data_dir = os.path.join("data", dataset_config["name"], "train")

    # Define class_folders first
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"\nFound {len(class_folders)} classes:")
    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  • {class_name}: {num_images} images")    display_header()

    # Initialize tracking
    class_correct = {cls: 0 for cls in class_folders}
    class_total = {cls: 0 for cls in class_folders}

    # Training loop
    for epoch in range(epochs):
        model.train()

        # Process each class separately for latent space generation
        for class_name in class_folders:
            message = f"Processing class: {class_name}"
            update_progress(message, 0, class_total[class_name])

            class_indices = [i for i, (_, label) in enumerate(train_dataset)
                           if train_dataset.classes[label] == class_name]
            class_subset = torch.utils.data.Subset(train_dataset, class_indices)
            class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)

            for batch_idx, (images, _) in enumerate(class_loader):
                # Training step
                reconstructed, latent_1d = model(images)

                # Save latent representations
                batch_paths = [train_dataset.imgs[i][0] for i in class_indices[batch_idx*batch_size:
                             (batch_idx+1)*batch_size]]
                save_batch_latents(latent_1d, batch_paths, dataset_name)

                # Update progress and confusion matrix
                current = (batch_idx + 1) * batch_size
                accuracy = class_correct[class_name] / class_total[class_name] * 100
                update_progress(message, current, class_total[class_name], accuracy)
                display_confusion_matrix(class_correct, class_total)




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
