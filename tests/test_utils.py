import torch
from autoencoder.utils import preprocess_image, postprocess_image
from PIL import Image

def test_preprocess_image():
    """Test the image preprocessing function."""
    # Create a dummy image
    dummy_image = Image.new("RGB", (32, 32), color="red")
    
    # Preprocess the image
    image_tensor = preprocess_image(dummy_image)
    
    # Check the output shape and type
    assert image_tensor.shape == (1, 3, 32, 32), "Preprocessed image shape mismatch"
    assert isinstance(image_tensor, torch.Tensor), "Output should be a torch.Tensor"

def test_postprocess_image():
    """Test the image postprocessing function."""
    # Create a dummy tensor
    dummy_tensor = torch.randn(3, 32, 32)
    
    # Postprocess the tensor
    image = postprocess_image(dummy_tensor)
    
    # Check the output type
    assert isinstance(image, Image.Image), "Output should be a PIL.Image"
