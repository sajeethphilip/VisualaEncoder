import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, embedding_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )
        
        # Embedding layer
        self.embedding = nn.Linear(latent_dim, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=0),  # 1x1 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        embedding = self.embedding(latent)  # Map latent space to embedding space
        reconstructed = self.decoder(embedding)
        return reconstructed, latent, embedding
