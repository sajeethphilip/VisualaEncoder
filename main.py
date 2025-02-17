import os
import torch
import requests
import zipfile
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from autoencoder.model import Autoencoder
from autoencoder.train import train_model
from autoencoder.reconstruct import reconstruct_image
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv

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

def load_dataset(dataset_name, data_dir="./data", train=True):
    """Load a dataset from torchvision.datasets or a local directory."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)
    elif dataset_name == "MNIST":
        dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    else:
        # Load custom dataset from local directory
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, dataset_name), transform=transform)
    
    return dataset

def main():
    """Main function for user interaction."""
    print("Welcome to the Autoencoder Tool!")
    
    # Step 1: Select data source
    print("\nSelect data source:")
    print("1. Torchvision dataset (e.g., CIFAR10, MNIST)")
    print("2. URL to download dataset")
    print("3. Local file")
    data_source = input("Enter your choice (1/2/3): ")
    
    if data_source == "1":
        # Load torchvision dataset
        dataset_name = input("Enter dataset name (e.g., CIFAR10, MNIST, CIFAR100): ")
        dataset = load_dataset(dataset_name)
    elif data_source == "2":
        # Download dataset from URL
        url = input("Enter the URL to download the dataset: ")
        download_and_extract(url)
        dataset_name = input("Enter a name for the dataset: ")
        dataset = load_dataset(dataset_name)
    elif data_source == "3":
        # Load local file
        dataset_name = input("Enter the path to the local dataset folder: ")
        dataset = load_dataset(dataset_name)
    else:
        raise ValueError("Invalid choice. Please select 1, 2, or 3.")
    
    # Step 2: Train or Predict
    print("\nSelect mode:")
    print("1. Train")
    print("2. Predict")
    mode = input("Enter your choice (1/2): ")
    
    if mode == "1":
        # Train the model
        print("\nTraining the model...")
        train_model(dataset, dataset_name)
    elif mode == "2":
        # Predict (reconstruct images)
        print("\nReconstructing images...")
        checkpoint_path = input("Enter the path to the trained model checkpoint: ")
        image_path = input("Enter the path to the input image: ")
        reconstruct_image(image_path, checkpoint_path, dataset_name)
    else:
        raise ValueError("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
