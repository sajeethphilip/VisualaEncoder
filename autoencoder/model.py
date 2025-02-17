import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_channels, input_size, latent_dim=128, embedding_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (input_size[0] // 8) * (input_size[1] // 8), latent_dim)
        )
        
        # Embedding layer
        self.embedding = nn.Linear(latent_dim, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        embedding = self.embedding(latent)
        reconstructed = self.decoder(embedding)
        return reconstructed, latent, embedding
