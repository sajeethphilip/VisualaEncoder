import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder
from autoencoder.data_loader import load_torchvision_dataset, download_and_extract, load_local_dataset
from autoencoder.utils import get_device

import torch
import torch.nn as nn

class EmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(EmbeddingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        embeddings: Tensor of shape (batch_size, embedding_dim)
        labels: Tensor of shape (batch_size,) containing class labels
        """
        batch_size = embeddings.size(0)
        loss = 0.0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if labels[i] == labels[j]:  # Same class
                    loss += F.mse_loss(embeddings[i], embeddings[j])  # Pull closer
                else:  # Different classes
                    loss += max(0, self.margin - F.mse_loss(embeddings[i], embeddings[j]))  # Push apart
        
        return loss / (batch_size * (batch_size - 1))  # Normalize by number of pairs

# Hyperparameters
latent_dim = 128
batch_size = 64
epochs = 50  # Increase max epochs since early stopping may stop training earlier
learning_rate = 1e-3
dataset_name = "CIFAR10"  # Can be "CIFAR10", "MNIST", or a path to a local dataset
data_source = "torchvision"  # Can be "torchvision", "url", or "local"
patience = 5  # Number of epochs to wait for improvement before stopping

# Detect device
device = get_device()
print(f"Using device: {device}")

# Load dataset
if data_source == "torchvision":
    dataset = load_torchvision_dataset(dataset_name, train=True)
elif data_source == "url":
    url = "https://example.com/dataset.zip"  # Replace with the actual URL
    download_and_extract(url)
    dataset = load_local_dataset("./data/extracted_dataset")
elif data_source == "local":
    dataset = load_local_dataset("./data/local_dataset")
else:
    raise ValueError(f"Unsupported data source: {data_source}")

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% training, 20% validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = Autoencoder(latent_dim).to(device)  # Move model to the detected device
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create checkpoint directory
os.makedirs("Model/checkpoint", exist_ok=True)
best_loss = float("inf")
epochs_without_improvement = 0  # Counter for early stopping

# Initialize model, loss, and optimizer
model = Autoencoder(latent_dim=128, embedding_dim=64).to(device)
criterion_recon = nn.MSELoss()  # Reconstruction loss
criterion_embed = EmbeddingLoss(margin=1.0)  # Embedding loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    epoch_loss = 0.0
    
    for batch in progress_bar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        reconstructed, embeddings = model(images)
        
        # Compute losses
        loss_recon = criterion_recon(reconstructed, images)
        loss_embed = criterion_embed(embeddings, labels)
        loss = loss_recon + loss_embed  # Combine losses
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average epoch loss
    epoch_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = images.to(device)  # Move data to the same device as the model
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            val_loss += loss.item()
    
    # Calculate average validation loss
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")
    
    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0  # Reset the counter
        checkpoint_path = f"Model/checkpoint/Best_{dataset_name}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
            "device": str(device)  # Save device metadata
        }, checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break
