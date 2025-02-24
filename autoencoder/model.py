import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CosineLatentMapper(nn.Module):
    def __init__(self, high_dim=512, device=None):
        super(CosineLatentMapper, self).__init__()
        self.high_dim = high_dim
        self.device = device if device is not None else torch.device("cpu")
        self._initialize_frequencies()

    def _initialize_frequencies(self):
        """Initialize or reinitialize frequencies"""
        if not hasattr(self, 'frequencies'):
            frequencies = torch.tensor([2*math.pi*prime/self.high_dim for prime in self._get_first_n_primes(self.high_dim)])
            frequencies = frequencies / frequencies.max()  # Normalize to [0,1] range
            # Change shape to match input dimensions - keep as 1D tensor
            self.register_buffer('frequencies', frequencies)


    def load_state_dict(self, state_dict, strict=False):
        """Override load_state_dict to handle missing frequencies"""
        if 'frequencies' not in state_dict:
            print("Initializing new frequencies for latent mapper...")
            self._initialize_frequencies()
            # Continue loading the rest of the state dict
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=strict)



    def _get_first_n_primes(self, n):
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % prime != 0 for prime in primes):
                primes.append(num)
            num += 1
        return primes

    def forward_map(self, x):
        # x shape: [batch_size, high_dim]
        # Normalize input to preserve class-specific features
        x = torch.tanh(x)  # Bound input to [-1,1]

        # Reshape frequencies to match batch dimension
        # frequencies shape: [high_dim] -> [1, high_dim]
        freqs = self.frequencies.view(1, -1)

        # Expand frequencies to match batch size
        # [1, high_dim] -> [batch_size, high_dim]
        freqs = freqs.expand(x.size(0), -1)

        # Element-wise multiplication instead of matmul
        angles = x * freqs

        return torch.clamp(torch.cos(angles), -0.99, 0.99)

    def inverse_map(self, y):
        # y shape: [batch_size, high_dim]
        y = torch.clamp(y, -0.99, 0.99)
        angles = torch.arccos(y)

        # Reshape frequencies for inverse mapping
        freqs = self.frequencies.view(1, -1).expand(y.size(0), -1)

        return torch.tanh(angles / (freqs + 1e-8))  # Add small epsilon


class ModifiedAutoencoder(nn.Module):
    def __init__(self, config, device=None):
        super(ModifiedAutoencoder, self).__init__()

        # Load configurations
        dataset_config = config["dataset"]
        model_config = config["model"]
        self.input_size = dataset_config["input_size"]
        self.in_channels = dataset_config["in_channels"]
        self.feature_dims = model_config["feature_dims"]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, self.feature_dims, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.feature_dims),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1))
        )  # Close the encoder Sequential block properly

        # Cosine Latent Mapper
        self.latent_mapper = CosineLatentMapper(high_dim=self.feature_dims, device=device)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dims, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.adaptive_upsample = nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)  # (batch_size, 512)

        # Map to 1D using cosine transformation
        latent_1d = self.latent_mapper.forward_map(x_flat)  # (batch_size, 1)

        # Map back to 512D
        decoded_flat = self.latent_mapper.inverse_map(latent_1d)  # (batch_size, 512)

        # Reshape and decode
        decoded_volume = decoded_flat.view(x.size(0), self.feature_dims, 1, 1)
        reconstructed = self.decoder(decoded_volume)
        reconstructed = self.adaptive_upsample(reconstructed)

        return reconstructed, latent_1d

class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()

        # Load dataset and model configurations
        dataset_config = config["dataset"]
        model_config = config["model"]

        self.input_size = dataset_config["input_size"]
        self.in_channels = dataset_config["in_channels"]
        self.latent_dim = model_config["feature_dims"]
        self.embedding_dim = model_config["autoencoder_config"].get("embedding_dim", 64)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Latent space and embedding
        self.encoder_output_size = 512 * 1 * 1
        self.fc_latent = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_embedding = nn.Linear(self.latent_dim, self.embedding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(self.embedding_dim, self.encoder_output_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Adaptive upsampling
        self.adaptive_upsample = nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_latent(x)
        embedding = self.fc_embedding(latent)

        # Decoder
        x = self.decoder_fc(embedding)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        x = self.adaptive_upsample(x)
        return x, latent, embedding

class Autoencoder3(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()

        # Load dataset and model configurations
        dataset_config = config["dataset"]
        model_config = config["model"]

        self.input_size = dataset_config["input_size"]
        self.in_channels = dataset_config["in_channels"]
        self.latent_dim = model_config["feature_dims"]
        self.embedding_dim = model_config["autoencoder_config"].get("embedding_dim", 64)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Latent space and embedding
        self.encoder_output_size = 512 * 1 * 1
        self.fc_latent = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_embedding = nn.Linear(self.latent_dim, self.embedding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(self.embedding_dim, self.encoder_output_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Adaptive upsampling
        self.adaptive_upsample = nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_latent(x)
        embedding = self.fc_embedding(latent)

        # Decoder
        x = self.decoder_fc(embedding)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        x = self.adaptive_upsample(x)
        return x, latent, embedding

class Autoencoder2(nn.Module):
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
