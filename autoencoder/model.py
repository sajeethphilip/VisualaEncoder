import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_channels, input_size, latent_dim=128, embedding_dim=64):
        super(Autoencoder, self).__init__()

        self.input_size = input_size  # Store the input size for reference
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: (H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (1, 1)
        )

        # Calculate the flattened size after the encoder
        self.encoder_output_size = 128 * 1 * 1  # 128 channels * 1 * 1 (after AdaptiveAvgPool2d)

        # Latent space and embedding
        self.fc_latent = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_embedding = nn.Linear(latent_dim, embedding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(embedding_dim, self.encoder_output_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (2, 2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # Output: (8, 8)
            nn.Sigmoid()
        )

        # Adaptive upsampling to match the original input size
        self.adaptive_upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        latent = self.fc_latent(x)
        embedding = self.fc_embedding(latent)

        # Decoder
        x = self.decoder_fc(embedding)
        x = x.view(x.size(0), 128, 1, 1)  # Reshape to (batch_size, 128, 1, 1)
        x = self.decoder(x)
        x = self.adaptive_upsample(x)  # Upsample to the original input size
        return x, latent, embedding
