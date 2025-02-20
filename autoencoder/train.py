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
        print(f"  • {class_name}: {num_images} images")
    
    # Create dataset and loaders
    train_dataset = load_local_dataset(dataset_config["name"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    # Model initialization
    model = ModifiedAutoencoder(config, device=device)
    model = model.to(device)
    
    # Optimizer setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], 
               config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )
    
    criterion_recon = nn.MSELoss()
    
    # Checkpoint handling
    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint if exists
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))
    
    # Get epochs from config
    epochs = config["training"]["epochs"]
    # Ensure at least one epoch of training
    epochs = max(epochs + start_epoch, start_epoch + 1)
    
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0
    
    print("\nStarting training...")
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for images, _ in progress_bar:
            images = images.to(device)
            reconstructed, _ = model(images)
            
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
                'avg_loss': epoch_loss / num_batches
            })
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint if loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            model.cpu()
            save_checkpoint(model, epoch + 1, avg_epoch_loss, config, checkpoint_path)
            model.to(device)
            print(f"\nSaved best model checkpoint (loss: {best_loss:.4f})")
            
            # Generate latent space after saving best model
            print("\nGenerating latent space representations...")
            model.eval()
            with torch.no_grad():
                for class_name in class_folders:
                    class_dir = os.path.join(data_dir, class_name)
                    class_dataset = load_local_dataset(dataset_config["name"])
                    class_loader = DataLoader(
                        class_dataset,
                        batch_size=config["training"]["batch_size"],
                        shuffle=False,
                        num_workers=config["training"]["num_workers"]
                    )
                    
                    print(f"\nProcessing class: {class_name}")
                    for batch_idx, (images, _) in enumerate(class_loader):
                        images = images.to(device)
                        _, latent_1d = model(images)
                        
                        # Get paths for saving
                        if hasattr(class_dataset, 'imgs'):
                            full_paths = [class_dataset.imgs[i][0] for i in range(len(images))]
                        else:
                            full_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        # Save latent representations
                        save_batch_latents(latent_1d, full_paths, dataset_config["name"])
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nNo improvement for {patience} epochs. Training stopped.")
            break
    
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
