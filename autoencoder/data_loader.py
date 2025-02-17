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

def load_dataset(dataset_name, train=True):
    """Load a dataset based on the configuration."""
    config = load_dataset_config(dataset_name)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(config["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"])
    ])
    
    if config["type"] == "torchvision":
        # Load torchvision dataset
        if dataset_name == "mnist":
            dataset = datasets.MNIST(root="data", train=train, download=True, transform=transform)
        elif dataset_name == "cifar10":
            dataset = datasets.CIFAR10(root="data", train=train, download=True, transform=transform)
        elif dataset_name == "cifar100":
            dataset = datasets.CIFAR100(root="data", train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")
    else:
        # Load custom dataset from local directory
        data_dir = config["train_dir"] if train else config["test_dir"]
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    return dataset, config

def download_and_extract(url, extract_to="./data"):
    """Download and extract a dataset from a URL."""
    os.makedirs(extract_to, exist_ok=True)
    filename = os.path.join(extract_to, url.split("/")[-1])
    
    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    # Extract the file
    print(f"Extracting {filename}...")
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif filename.endswith(".tar.gz") or filename.endswith(".tar"):
        with tarfile.open(filename, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    print(f"Dataset extracted to {extract_to}")

def load_local_dataset(data_dir, transform=None):
    """Load a dataset from a local directory."""
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset
