import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, conv_layers=3, use_batch_norm=False):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        in_channels = 3
        current_channels = 32

        for _ in range(conv_layers):
            encoder_layers.extend([
                nn.Conv2d(in_channels, current_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm2d(current_channels))
            in_channels = current_channels
            current_channels *= 2

        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.Flatten(),
            nn.Linear(current_channels//2 * 4 * 4, latent_dim)
        )

        # Decoder
        decoder_layers = []
        self.decoder_input = nn.Linear(latent_dim, current_channels//2 * 4 * 4)

        for i in range(conv_layers):
            decoder_layers.extend([
                nn.ConvTranspose2d(current_channels//2, current_channels//4,
                                 kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
            if use_batch_norm and i < conv_layers-1:
                decoder_layers.append(nn.BatchNorm2d(current_channels//4))
            current_channels //= 2

        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(self.decoder_input(latent).view(-1, 128, 4, 4))
        return reconstructed, latent, latent  # Return latent twice for compatibility

class ComplexAutoencoder(nn.Module):
    def __init__(self, config):
        super(ComplexAutoencoder, self).__init__()

        self.latent_dim = config["latent_dim"]
        self.embedding_dim = config["embedding_dim"]
        use_batch_norm = config["use_batch_norm"]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((1, 1)) if config["use_adaptive_pooling"] else nn.Identity()
        )

        # Latent space mappings
        self.fc_latent = nn.Linear(512, self.latent_dim)
        self.fc_embedding = nn.Linear(self.latent_dim, self.embedding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(self.embedding_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_latent(x)
        embedding = self.fc_embedding(latent)
        x = self.decoder_fc(embedding)
        x = x.view(-1, 512, 4, 4)
        reconstructed = self.decoder(x)
        return reconstructed, latent, embedding

class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()

        # Load configurations
        dataset_config = config["dataset"]
        model_config = config["model"]

        self.input_size = dataset_config["input_size"]
        self.in_channels = dataset_config["in_channels"]
        self.latent_dim = model_config["feature_dims"]
        self.embedding_dim = model_config["autoencoder_config"].get("embedding_dim", 64)

        # Original encoder
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

        # Feature enhancement module
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.Sigmoid()  # Output attention weights
        )

        # Latent space processing
        self.encoder_output_size = 512 * 1 * 1
        self.fc_latent = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_embedding = nn.Linear(self.latent_dim, self.embedding_dim)

        # Feature attention module
        self.attention_module = nn.Sequential(
            nn.Linear(self.embedding_dim, self.latent_dim),
            nn.Sigmoid()
        )

        # Decoder components
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

        self.adaptive_upsample = nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False)

    def enhance_features(self, latent):
        # Generate feature attention weights
        attention_weights = self.feature_enhancer(latent)
        # Apply attention to latent representation
        enhanced_latent = latent * attention_weights
        return enhanced_latent, attention_weights

    def forward(self, x):
        # Encoder path
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_latent(x)

        # Feature enhancement
        enhanced_latent, attention_weights = self.enhance_features(latent)

        # Generate embedding with enhanced features
        embedding = self.fc_embedding(enhanced_latent)

        # Generate attention mask from embedding
        attention_mask = self.attention_module(embedding)

        # Decoder path with enhanced features
        x = self.decoder_fc(embedding)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        x = self.adaptive_upsample(x)

        return x, enhanced_latent, embedding, attention_mask


class Autoencoder4(nn.Module):
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
