import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path):
    """Preprocess an image for the autoencoder."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def postprocess_image(image_tensor):
    """Convert a tensor back to a PIL image."""
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)  # Remove batch dimension and rearrange
    image = transforms.ToPILImage()(image_tensor)
    return image
