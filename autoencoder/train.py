import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder
from autoencoder.data_loader import load_dataset, load_dataset_config
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv

def train_model(dataset_name):
    """Train the autoencoder model."""
    # Load dataset and config
    dataset, config = load_dataset(dataset_name, train=True)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% training, 20% validation
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = get_device()
    model = Autoencoder(
        in_channels=config["in_channels"],
        input_size=config["input_size"],
        latent_dim=128,
        embedding_dim=64
    ).to(device)
    
    # Define loss and optimizer
    criterion_recon = nn.MSELoss()  # Reconstruction loss
    criterion_embed = nn.MSELoss()  # Embedding loss (can be customized)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training parameters
    epochs = 50
    patience = 5  # Early stopping patience
    best_loss = float("inf")
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_loss = 0.0
        
        for batch in progress_bar:
            images, _ = batch
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latent, embeddings = model(images)
            
            # Compute losses
            loss_recon = criterion_recon(reconstructed, images)
            loss_embed = criterion_embed(embeddings, torch.zeros_like(embeddings))  # Placeholder for embedding loss
            loss = loss_recon + loss_embed  # Combine losses
            
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
                
                # Forward pass
                reconstructed, latent, embeddings = model(images)
                
                # Compute validation loss
                loss = criterion_recon(reconstructed, images)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0  # Reset the counter
            
            # Save the best model
            checkpoint_path = f"Model/checkpoint/Best_{dataset_name}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": config  # Save dataset configuration
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break
    
    # Save latent space and embeddings for the entire dataset
    model.eval()
    latent_space = []
    embeddings_space = []
    
    with torch.no_grad():
        for batch in train_loader:
            images, _ = batch
            images = images.to(device)
            
            # Forward pass
            _, latent, embeddings = model(images)
            
            # Store latent and embeddings
            latent_space.append(latent.cpu())
            embeddings_space.append(embeddings.cpu())
    
    # Concatenate all batches
    latent_space = torch.cat(latent_space, dim=0)
    embeddings_space = torch.cat(embeddings_space, dim=0)
    
    # Save latent space and embeddings
    save_latent_space(latent_space, dataset_name, "latent.pkl")
    save_embeddings_as_csv(embeddings_space, dataset_name, "embeddings.csv")

if __name__ == "__main__":
    # Example usage
    dataset_name = "mnist"  # Replace with the dataset name
    train_model(dataset_name)
