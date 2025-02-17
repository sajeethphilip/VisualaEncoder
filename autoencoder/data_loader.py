import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def load_dataset_config(dataset_name):
    """Load dataset configuration from JSON file."""
    config_path = f"data/{dataset_name}/{dataset_name}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["dataset"]


def load_local_dataset(dataset_name, transform=None):
    """Load a dataset from a local directory."""
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    data_dir=f"data/{dataset_name}/train/"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset
