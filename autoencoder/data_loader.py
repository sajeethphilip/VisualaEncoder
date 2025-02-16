import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import requests
import zipfile
import tarfile

def load_torchvision_dataset(dataset_name, root="./data", train=True):
    """Load a dataset from torchvision.datasets."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset

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
