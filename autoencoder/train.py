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
    """Train the autoencoder model with proper device handling and display."""
    
    # Get device first
    device = get_device()
    
    # Display header (stays at top)
    print("\033[2J\033[H")  # Clear screen
    print("\033[96m" + "="*80 + "\033[0m")
    print("\033[93m" + "Visual Autoencoder Tool".center(80) + "\033[0m")
    print("\033[96m" + "="*80 + "\033[0m")
    print("\033[97m" + "Author: ".rjust(40) + "\033[93m" + "Ninan Sajeeth Philip" + "\033[0m")
    print("\033[97m" + "Organisation: ".rjust(40) + "\033[92m" + "AIRIS" + "\033[0m")
    print("\033[92m" + "Thelliyoor".center(80) + "\033[0m")
    print("\033[96m" + "="*80 + "\033[0m\n")
    
    # Dataset setup
    dataset_config = config["dataset"]
    data_dir = os.path.join("data", dataset_config["name"], "train")
    
    # Verify class structure
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(class_folders)} classes:")
    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        num_images = len([f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  • {class_name}: {num_images} images")
    
    # Create dataset and loader
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))
    
    # Training loop
    epochs = max(config["training"]["epochs"] + start_epoch, start_epoch + 1)
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0
    
    # Get terminal size for display
    terminal_size = os.get_terminal_size()
    terminal_height = terminal_size.lines
    header_height = 10  # Reserve space for header
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Training progress
        print(f"\033[{header_height+1};0H")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for images, _ in tqdm(train_loader, leave=False, position=terminal_height-header_height):
            images = images.to(device)
            reconstructed, latent_1d = model(images)
            
            loss = criterion_recon(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        
        # Save checkpoint if improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            
            # Save model checkpoint
            model.cpu()
            save_checkpoint(model, epoch + 1, avg_epoch_loss, config, checkpoint_path)
            model.to(device)
            
            # Generate latent space after saving best model
            print(f"\033[{terminal_height-1};0H")
            print("Generating latent space representations...")
            
            model.eval()
            with torch.no_grad():
                for class_name in class_folders:
                    # Create class-specific loader
                    class_indices = [i for i, (_, label) in enumerate(train_dataset) 
                                   if train_dataset.classes[label] == class_name]
                    class_subset = torch.utils.data.Subset(train_dataset, class_indices)
                    class_loader = DataLoader(
                        class_subset,
                        batch_size=config["training"]["batch_size"],
                        shuffle=False,
                        num_workers=config["training"]["num_workers"]
                    )
                    
                    print(f"\033[{terminal_height-1};0H")
                    print(f"Processing class: {class_name} ({len(class_subset)} images)")
                    
                    for images, _ in tqdm(class_loader, leave=False, position=terminal_height-header_height):
                        images = images.to(device)
                        _, latent_1d = model(images)
                        
                        # Get paths for saving
                        batch_indices = [class_indices[i] for i in range(len(images))]
                        full_paths = [train_dataset.imgs[i][0] for i in batch_indices]
                        
                        # Save latent representations with original filenames
                        save_batch_latents(latent_1d, full_paths, dataset_config["name"])
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\033[{terminal_height-1};0H")
            print(f"No improvement for {patience} epochs. Training stopped.")
            break
    
    print(f"\033[{terminal_height-1};0H")
    print("Training complete!")
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
