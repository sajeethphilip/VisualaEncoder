import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder.model import Autoencoder
from autoencoder.data_loader import load_torchvision_dataset, download_and_extract, load_local_dataset
from autoencoder.utils import get_device

# Hyperparameters
latent_dim = 128
batch_size = 64
epochs = 20
learning_rate = 1e-3
dataset_name = "CIFAR10"  # Can be "CIFAR10", "MNIST", or a path to a local dataset
data_source = "torchvision"  # Can be "torchvision", "url", or "local"

# Detect device
device = get_device()
print(f"Using device: {device}")

# Load dataset
if data_source == "torchvision":
    train_dataset = load_torchvision_dataset(dataset_name, train=True)
elif data_source == "url":
    url = "https://example.com/dataset.zip"  # Replace with the actual URL
    download_and_extract(url)
    train_dataset = load_local_dataset("./data/extracted_dataset")
elif data_source == "local":
    train_dataset = load_local_dataset("./data/local_dataset")
else:
    raise ValueError(f"Unsupported data source: {data_source}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = Autoencoder(latent_dim).to(device)  # Move model to the detected device
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create checkpoint directory
os.makedirs("Model/checkpoint", exist_ok=True)
best_loss = float("inf")

# Training loop with tqdm
for epoch in range(epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    epoch_loss = 0.0
    
    for batch in progress_bar:
        images, _ = batch
        images = images.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average epoch loss
    epoch_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    
    # Save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        checkpoint_path = f"Model/checkpoint/Best_{dataset_name}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
            "device": str(device)  # Save device metadata
        }, checkpoint_path)
        print(f"Saved best model to {checkpoint_path}")
