import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder.model import Autoencoder

# Hyperparameters
latent_dim = 128
batch_size = 64
epochs = 20
learning_rate = 1e-3

# Load dataset (e.g., CIFAR-10)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = Autoencoder(latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        images, _ = batch
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "autoencoder.pth")
