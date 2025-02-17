import os
import torch
import requests
import zipfile
import tarfile
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from autoencoder.model import Autoencoder
from autoencoder.train import train_model
from autoencoder.reconstruct import reconstruct_image
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv
from torchvision import datasets, transforms
from PIL import Image
from autoencoder.train import train_model
from autoencoder.reconstruct import reconstruct_image
from autoencoder.data_loader import load_dataset, load_dataset_config

def create_json_config(dataset_name, data_dir, image_path):
    """Create a JSON configuration file by reading the first image in the dataset."""
    # Read the first image to determine its shape
    image = Image.open(image_path)
    width, height = image.size
    in_channels = 1 if image.mode == "L" else 3  # Grayscale or RGB
    
    # Create JSON configuration
    config = {
        "dataset": {
            "name": dataset_name,
            "type": "custom",
            "in_channels": in_channels,
            "num_classes": 0,  # Update if class labels are available
            "input_size": [height, width],  # Height x Width
            "mean": [0.5] * in_channels,  # Default mean
            "std": [0.5] * in_channels,  # Default std
            "train_dir": os.path.join(data_dir, "train"),
            "test_dir": os.path.join(data_dir, "test"),
            "image_type": "grayscale" if in_channels == 1 else "rgb"
        }
    }
    
    # Save JSON file
    json_path = os.path.join(data_dir, f"{dataset_name}.json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Created JSON configuration file at {json_path}")
    return config

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
        data_dir = os.path.join("data", dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if JSON file exists
        json_path = os.path.join(data_dir, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            print(f"JSON configuration file not found at {json_path}. Creating one...")
            # Load the dataset to get the first image
            dataset = datasets.__dict__[dataset_name](root="data", train=True, download=True)
            image, _ = dataset[0]
            image_path = os.path.join(data_dir, "sample_image.png")
            image.save(image_path)
            create_json_config(dataset_name, data_dir, image_path)
        
        # Load dataset
        dataset, config = load_dataset(dataset_name)
    elif data_source == "2":
        # Download dataset from URL
        url = input("Enter the URL to download the dataset: ")
        dataset_name = input("Enter a name for the dataset: ")
        data_dir = os.path.join("data", dataset_name)
        download_and_extract(url, data_dir)
        
        # Check if JSON file exists
        json_path = os.path.join(data_dir, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            print(f"JSON configuration file not found at {json_path}. Creating one...")
            # Find the first image in the dataset
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        create_json_config(dataset_name, data_dir, image_path)
                        break
                break
        
        # Load dataset
        dataset, config = load_dataset(dataset_name)
    elif data_source == "3":
        # Load local file
        dataset_name = input("Enter the path to the local dataset folder: ")
        data_dir = dataset_name
        dataset_name = os.path.basename(os.path.normpath(dataset_name))
        
        # Check if JSON file exists
        json_path = os.path.join(data_dir, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            print(f"JSON configuration file not found at {json_path}. Creating one...")
            # Find the first image in the dataset
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        create_json_config(dataset_name, data_dir, image_path)
                        break
                break
        
        # Load dataset
        dataset, config = load_dataset(dataset_name)
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
        train_model(dataset_name)  # Pass only dataset_name
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
