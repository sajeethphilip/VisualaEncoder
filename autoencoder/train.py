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
    """Train the autoencoder model with confusion matrix and verified latent space saving."""
    
    # Get device first
    device = get_device()
    
    # Display header with fixed positioning
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
    
    # Initialize confusion matrix tracking
    class_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    confusion_matrix = {cls: {'correct': 0, 'total': 0} for cls in class_folders}
    
    # Model and training setup
    model = ModifiedAutoencoder(config, device=device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
        betas=(config["model"]["optimizer"]["beta1"], config["model"]["optimizer"]["beta2"]),
        eps=config["model"]["optimizer"]["epsilon"]
    )
    criterion_recon = nn.MSELoss()
    
    # Load checkpoint if exists
    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))
    
    # Training loop
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping"]["patience"]
    patience_counter = 0
    
    # Get terminal size for display
    terminal_size = os.get_terminal_size()
    terminal_height = terminal_size.lines
    header_height = 10  # Reserve space for header
    
    def update_confusion_matrix(original, reconstructed, class_name):
        """Update confusion matrix based on reconstruction quality."""
        mse = torch.mean((original - reconstructed)**2).item()
        threshold = 0.1  # Adjust based on your needs
        confusion_matrix[class_name]['total'] += 1
        if mse < threshold:
            confusion_matrix[class_name]['correct'] += 1
    
    def display_confusion_matrix():
        """Display color-coded confusion matrix."""
        print("\033[{};0H".format(header_height + 1))
        print("Reconstruction Accuracy Matrix:")
        for class_name in confusion_matrix:
            correct = confusion_matrix[class_name]['correct']
            total = confusion_matrix[class_name]['total']
            accuracy = (correct / total) * 100 if total > 0 else 0
            color = "\033[92m" if accuracy > 80 else "\033[91m"  # Green if >80%, red otherwise
            print(f"{color}{class_name}: {accuracy:.1f}% ({correct}/{total})\033[0m")
    
    # Training loop with confusion matrix and latent space saving
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\033[{header_height+len(class_folders)+3};0H")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Process each class separately
        for class_name in class_folders:
            class_dir = os.path.join(data_dir, class_name)
            class_dataset = load_local_dataset(dataset_config["name"])
            class_loader = DataLoader(
                class_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                num_workers=config["training"]["num_workers"]
            )
            
            for images, _ in tqdm(class_loader, 
                                desc=f"Processing {class_name}", 
                                position=terminal_height-header_height,
                                leave=False):
                images = images.to(device)
                reconstructed, latent_1d = model(images)
                
                # Update confusion matrix
                update_confusion_matrix(images, reconstructed, class_name)
                
                # Save latent representations
                batch_metadata = {
                    'epoch': epoch + 1,
                    'class': class_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Get full paths for saving
                if hasattr(class_dataset, 'imgs'):
                    full_paths = [class_dataset.imgs[i][0] for i in range(len(images))]
                else:
                    full_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Save latent space
                save_batch_latents(latent_1d, full_paths, dataset_config["name"], batch_metadata)
                
                # Training step
                loss = criterion_recon(reconstructed, images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Display updated confusion matrix
            display_confusion_matrix()
        
        # Handle checkpoints and early stopping
        avg_epoch_loss = epoch_loss / num_batches
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            save_checkpoint(model, epoch + 1, avg_epoch_loss, config, checkpoint_path)
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
