import torch
from autoencoder.model import Autoencoder

def test_autoencoder_forward_pass():
    """Test the forward pass of the autoencoder."""
    model = Autoencoder(latent_dim=128)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch of 1 image, 3 channels, 32x32
    reconstructed, latent = model(input_tensor)
    
    # Check output shapes
    assert reconstructed.shape == input_tensor.shape, "Reconstructed image shape mismatch"
    assert latent.shape == (1, 128), "Latent vector shape mismatch"

def test_autoencoder_training():
    """Test if the autoencoder can overfit to a single batch."""
    model = Autoencoder(latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Create a single batch of data
    input_tensor = torch.randn(1, 3, 32, 32)
    
    # Train for a few steps
    for _ in range(10):
        optimizer.zero_grad()
        reconstructed, _ = model(input_tensor)
        loss = criterion(reconstructed, input_tensor)
        loss.backward()
        optimizer.step()
    
    # Check if the loss decreases
    assert loss.item() < 1.0, "Loss did not decrease during training"
