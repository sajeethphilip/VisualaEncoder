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

def train_model(config):
    """Train the autoencoder model."""
    device = get_device()
    print(f"Using device: {device}")

    # Dataset setup with proper batch handling
    dataset_config = config["dataset"]
    dataset = load_local_dataset(dataset_config["name"])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,  # Fixed batch size to avoid BatchNorm issues
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    class_names = dataset.classes
    num_classes = len(class_names)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)

    # Model setup
    model = ModifiedAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"]
    )
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        confusion_matrix.zero_()
        
        # Training with progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            reconstructed, latent_1d = model(images)
            
            # Update confusion matrix
            with torch.no_grad():
                pred_labels = torch.argmax(latent_1d, dim=1)
                for t, p in zip(labels, pred_labels):
                    confusion_matrix[t.long(), p.long()] += 1

            # Compute loss and optimize
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        # After each epoch, generate latent space for all classes
        model.eval()
        with torch.no_grad():
            for class_idx, class_name in enumerate(class_names):
                class_path = os.path.join(dataset_config["train_dir"], str(class_idx))
                latent_dir = os.path.join(f"data/{dataset_config['name']}/latent_space/train", str(class_idx))
                recon_dir = os.path.join(f"data/{dataset_config['name']}/reconstructed_images/train", str(class_idx))
                
                os.makedirs(latent_dir, exist_ok=True)
                os.makedirs(recon_dir, exist_ok=True)

                # Create class-specific dataloader
                class_dataset = datasets.ImageFolder(
                    root=os.path.dirname(class_path),
                    transform=dataset.transform
                )
                class_loader = torch.utils.data.DataLoader(
                    class_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=4
                )

                for images, _ in class_loader:
                    images = images.to(device)
                    reconstructed, latent_1d = model(images)

                    # Save individual latent representations and reconstructions
                    for idx, (latent, recon) in enumerate(zip(latent_1d, reconstructed)):
                        # Get original image path
                        img_path = class_dataset.imgs[idx][0]
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        
                        # Save latent representation
                        save_1d_latent_to_csv(
                            latent, 
                            img_path,
                            dataset_config["name"],
                            {'epoch': epoch + 1, 'class': class_idx}
                        )
                        
                        # Save reconstructed image
                        recon_img = transforms.ToPILImage()(recon.cpu())
                        recon_img.save(os.path.join(recon_dir, f"{base_name}.png"))

        # Display confusion matrix after each epoch
        display_confusion_matrix(confusion_matrix, class_names)

    return model





def save_final_representations(model, loader, device, dataset_name):
    """Save the final latent space and embeddings."""
    latent_space = []
    embeddings_space = []

    with torch.no_grad():
        for images, _ in loader:
            # Ensure images are tensors and on the correct device
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            images = images.to(device)
            
            # Forward pass
            reconstructed, latent, embeddings = model(images)
            latent_space.append(latent.cpu())
            embeddings_space.append(embeddings.cpu())

    # Concatenate all batches
    latent_space = torch.cat(latent_space, dim=0)
    embeddings_space = torch.cat(embeddings_space, dim=0)

    # Save representations
    save_latent_space(latent_space, dataset_name, "latent.pkl")
    save_embeddings_as_csv(embeddings_space, dataset_name, "embeddings.csv")

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    train_model(config)
