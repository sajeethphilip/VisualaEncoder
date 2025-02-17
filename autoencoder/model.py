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
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # Output: (H/2, W/2)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (H/4, W/4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (H/8, W/8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: (H/16, W/16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Adaptive pooling to ensure fixed-size latent space
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (1, 1)
        )

        # Calculate the flattened size after the encoder
        self.encoder_output_size = 512 * 1 * 1  # 512 channels * 1 * 1 (after AdaptiveAvgPool2d)

        # Latent space and embedding
        self.fc_latent = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_embedding = nn.Linear(latent_dim, embedding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(embedding_dim, self.encoder_output_size)
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: (2, 2)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # Layer 2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Layer 3
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # Layer 4
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),  # Output: (16, 16)
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
        x = x.view(x.size(0), 512, 1, 1)  # Reshape to (batch_size, 512, 1, 1)
        x = self.decoder(x)
        x = self.adaptive_upsample(x)  # Upsample to the original input size
        return x, latent, embedding


class Autoencoder1(nn.Module):
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
