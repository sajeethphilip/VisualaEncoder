from torchvision import transforms, datasets
import csv
import torch
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import requests
import zipfile
import tarfile
from tqdm import tqdm
import gzip
import numpy as np
from PIL import Image
import json
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
#import datetime

import os
import requests
import zipfile
import tarfile
import shutil
import json
from PIL import Image
from tqdm import tqdm

import os
import torch
import torchvision
from torchvision import datasets, transforms
import logging
import json
import random
import numpy as np
from PIL import Image
import shutil

def find_first_image(directory):
    """Find the first image file in a directory or its subdirectories."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No image files found in {directory}")

def extract_and_organize(source_path, dataset_name, is_url=False):
    """Unified function to handle compressed files and organize dataset"""
    data_dir = os.path.join("data", dataset_name)
    temp_dir = os.path.join("temp", dataset_name)
    os.makedirs(temp_dir, exist_ok=True)

    if is_url:
        # Download from URL
        print(f"Downloading from {source_path}...")
        response = requests.get(source_path, stream=True)
        filename = os.path.join(temp_dir, source_path.split('/')[-1])
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        source_path = filename

    # Extract based on file type
    if source_path.endswith(('.zip', '.ZIP')):
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    elif source_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(source_path, 'r:gz') as tar_ref:
            tar_ref.extractall(temp_dir)
    elif source_path.endswith('.tar'):
        with tarfile.open(source_path, 'r:') as tar_ref:
            tar_ref.extractall(temp_dir)

    # Organize into train/test
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move files to appropriate directories
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                if 'train' in root.lower():
                    dest_dir = train_dir
                elif 'test' in root.lower():
                    dest_dir = test_dir
                else:
                    dest_dir = train_dir  # Default to train

                # Create class subdirectories if present
                class_name = os.path.basename(os.path.dirname(src_path))
                if class_name != dataset_name:
                    dest_dir = os.path.join(dest_dir, class_name)
                    os.makedirs(dest_dir, exist_ok=True)

                shutil.copy2(src_path, os.path.join(dest_dir, file))

    # Cleanup
    shutil.rmtree(temp_dir)
    return data_dir



def save_1d_latent_to_csv(latent_1d, image_name, dataset_name, metadata=None):
    """Save 1D latent representation to CSV with metadata"""
    data_dir = f"data/{dataset_name}/latent_space"
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, f"{image_name}_latent.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Save metadata
        writer.writerow(["image_name", image_name])
        writer.writerow(["timestamp", datetime.now().isoformat()])
        # Save frequencies if available
        if hasattr(latent_1d, 'frequencies'):
            writer.writerow(["frequencies"])
            writer.writerow(latent_1d.frequencies.detach().cpu().numpy().flatten())
        # Save latent values
        writer.writerow(["latent_values"])
        writer.writerow(latent_1d.detach().cpu().numpy().flatten())
    return csv_path

def load_1d_latent_from_csv(csv_path):
    """Load 1D latent representation from CSV with metadata"""
    metadata = {}
    latent_data = None
    frequencies = None

    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:  # Metadata row
                metadata[row[0]] = row[1]
            elif row[0] == "frequencies":
                frequencies = next(reader)  # Read next row for frequencies
            elif row[0] == "latent_values":
                latent_data = next(reader)  # Read next row for values

    if latent_data is None:
        raise ValueError("No latent values found in CSV file")

    latent_1d = torch.tensor([float(x) for x in latent_data])
    if frequencies is not None:
        metadata['frequencies'] = torch.tensor([float(x) for x in frequencies])

    return latent_1d.unsqueeze(0), metadata



def save_checkpoint(model, epoch, loss, config, checkpoint_path):
    """Save checkpoint with explicit frequency handling."""
    # Ensure model is in eval mode and on CPU for saving
    model.eval()
    model = model.cpu()

    # Verify frequencies exist and initialize if needed
    if not hasattr(model.latent_mapper, 'frequencies') or model.latent_mapper.frequencies is None:
        print("Initializing frequencies before saving...")
        # Assuming latent_mapper has the initialization method
        model.latent_mapper._initialize_frequencies()

    # Create state dict with explicit frequency saving
    state_dict = model.state_dict()

    # Verify frequencies are in state dict
    if 'latent_mapper.frequencies' not in state_dict:
        state_dict['latent_mapper.frequencies'] = model.latent_mapper.frequencies

    # Save complete checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "frequencies": model.latent_mapper.frequencies,  # Save frequencies separately as well
        "loss": loss,
        "config": config
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path} with frequencies shape: {model.latent_mapper.frequencies.shape}")


def load_checkpoint(checkpoint_path, model, config):
    """Load checkpoint with robust frequency handling."""
    device = model.device
    try:
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # First handle frequencies
        if "frequencies" in checkpoint:
            print("Loading frequencies from checkpoint...")
            model.latent_mapper.frequencies = checkpoint["frequencies"].to(device)
        else:
            print("No frequencies found in checkpoint, initializing...")
            model.latent_mapper._initialize_frequencies()

        # Clean state dict
        state_dict = checkpoint["model_state_dict"]
        cleaned_state_dict = {}

        for k, v in state_dict.items():
            # Remove any 'module.' prefix from DataParallel
            name = k.replace('module.', '')
            cleaned_state_dict[name] = v

        # Ensure frequencies are in state dict
        if 'latent_mapper.frequencies' not in cleaned_state_dict:
            cleaned_state_dict['latent_mapper.frequencies'] = model.latent_mapper.frequencies

        # Load state dict
        model.load_state_dict(cleaned_state_dict, strict=True)

        print(f"Successfully loaded checkpoint with frequencies shape: {model.latent_mapper.frequencies.shape}")
        return model, checkpoint.get("epoch", 0), checkpoint.get("loss", float("inf"))

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Initializing fresh model with new frequencies...")
        model.latent_mapper._initialize_frequencies()
        return model, 0, float("inf")

def setup_dataset(dataset_name):
    """Set up a torchvision dataset and return a full configuration."""
    # Convert dataset name to uppercase for torchvision
    dataset_name = dataset_name.upper()
    data_dir = os.path.join("data", dataset_name)
    input_data_dir = os.path.join(data_dir, "input_data")
    os.makedirs(input_data_dir, exist_ok=True)

    # Ask user about combining train and test data
    combine_data = input("Do you want to combine train and test data into a single training folder? (y/n): ").lower().strip() == 'y'

    # Basic transform for initial download
    transform = transforms.ToTensor()

    try:
        # Get dataset class dynamically
        dataset_class = getattr(datasets, dataset_name)

        # Download training data
        train_dataset = dataset_class(
            root=input_data_dir,
            train=True,
            download=True,
            transform=transform
        )

        # Download test data
        test_dataset = dataset_class(
            root=input_data_dir,
            train=False,
            download=True,
            transform=transform
        )

        # Create train and test directories
        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")

        # Clear existing directories if they exist
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        def save_dataset_to_folders(dataset, base_dir, is_train=True, start_idx=0):
            split = "train" if is_train else "test"
            print(f"Organizing {split} data...")

            # Create class subdirectories
            for class_idx in range(len(dataset.classes) if hasattr(dataset, 'classes') else len(set(dataset.targets))):
                os.makedirs(os.path.join(base_dir, str(class_idx)), exist_ok=True)

            # Save each image to its class folder
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                if isinstance(img, torch.Tensor):
                    # Convert tensor to PIL Image
                    if img.shape[0] == 1:  # Grayscale
                        img = transforms.ToPILImage()(img).convert('L')
                    else:  # RGB
                        img = transforms.ToPILImage()(img)

                # Save image with continuous indexing
                img_path = os.path.join(base_dir, str(label), f"{idx + start_idx}.png")
                img.save(img_path)

            return len(dataset)

        # Save images to their respective folders
        if combine_data:
            print("Combining train and test data into training folder...")
            num_train = save_dataset_to_folders(train_dataset, train_dir, is_train=True, start_idx=0)
            save_dataset_to_folders(test_dataset, train_dir, is_train=True, start_idx=num_train)
        else:
            save_dataset_to_folders(train_dataset, train_dir, is_train=True)
            save_dataset_to_folders(test_dataset, test_dir, is_train=False)

        # Calculate dataset statistics dynamically
        def get_dataset_stats(dataset):
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
            first_image, _ = dataset[0]
            num_channels = first_image.shape[0]
            input_size = list(first_image.shape[1:])

            channels_sum = torch.zeros(num_channels)
            channels_squared_sum = torch.zeros(num_channels)
            num_samples = 0

            for data, _ in dataloader:
                channels_sum += torch.mean(data, dim=[0, 2, 3]) * data.size(0)
                channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3]) * data.size(0)
                num_samples += data.size(0)

            mean = channels_sum / num_samples
            std = (channels_squared_sum / num_samples - mean ** 2) ** 0.5

            return mean.tolist(), std.tolist(), num_channels, input_size

        # Calculate statistics from training set
        mean, std, in_channels, input_size = get_dataset_stats(train_dataset)

        # Determine number of classes
        if hasattr(train_dataset, 'classes'):
            num_classes = len(train_dataset.classes)
        else:
            num_classes = len(torch.unique(torch.tensor(train_dataset.targets)))

        # Create dataset info dictionary
        dataset_info = {
            "dataset": {
                "name": dataset_name,
                "type": "torchvision",
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": input_size,
                "mean": mean,
                "std": std,
                "train_dir": train_dir,
                "test_dir": test_dir,
                "image_type": "image",
                "combined_train_test": combine_data
            },
            "model": {
                "encoder_type": "autoenc",
                "feature_dims": 128,
                "learning_rate": 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-08
                },
                "scheduler": {
                    "type": "ReduceLROnPlateau",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-06,
                    "verbose": True
                },
                "autoencoder_config": {
                    "reconstruction_weight": 1.0,
                    "feature_weight": 0.1,
                    "convergence_threshold": 0.001,
                    "min_epochs": 10,
                    "patience": 5,
                    "enhancements": {
                        "enabled": True,
                        "use_kl_divergence": True,
                        "use_class_encoding": True,
                        "kl_divergence_weight": 0.5,
                        "classification_weight": 0.5,
                        "clustering_temperature": 1.0,
                        "min_cluster_confidence": 0.7
                    }
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": 4,
                "checkpoint_dir": os.path.join(data_dir, "checkpoints"),
                "validation_split": 0.2,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            },
            "augmentation": {
                "enabled": False,
                "random_crop": {
                    "enabled": True,
                    "padding": 4
                },
                "random_rotation": {
                    "enabled": True,
                    "degrees": 10
                },
                "horizontal_flip": {
                    "enabled": True,
                    "probability": 0.5
                },
                "vertical_flip": {
                    "enabled": False
                },
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1
                },
                "normalize": {
                    "enabled": True,
                    "mean": mean,
                    "std": std
                }
            },
            "logging": {
                "log_dir": os.path.join(data_dir, "logs"),
                "tensorboard": {
                    "enabled": True,
                    "log_dir": os.path.join(data_dir, "tensorboard")
                },
                "save_frequency": 5,
                "metrics": ["loss", "accuracy", "reconstruction_error"]
            },
            "output": {
                "features_file": os.path.join(data_dir, f"{dataset_name}.csv"),
                "model_dir": os.path.join(data_dir, "models"),
                "visualization_dir": os.path.join(data_dir, "visualizations")
            }
        }

        # Save dataset info to JSON file
        json_path = os.path.join(data_dir, f"{dataset_name}.json")
        with open(json_path, 'w') as f:
            json.dump(dataset_info, f, indent=4)

        return dataset_info

    except AttributeError:
        logging.error(f"Dataset {dataset_name} not found in torchvision.datasets")
        return None
    except Exception as e:
        logging.error(f"Error setting up dataset: {str(e)}")
        return None

def get_augmentation_transform(config):
    """Create a data augmentation transform based on the configuration."""
    augmentation_config = config["augmentation"]
    transform_list = []

    if augmentation_config["enabled"]:
        if augmentation_config["random_crop"]["enabled"]:
            transform_list.append(transforms.RandomCrop(
                size=config["dataset"]["input_size"],
                padding=augmentation_config["random_crop"]["padding"]
            ))
        if augmentation_config["random_rotation"]["enabled"]:
            transform_list.append(transforms.RandomRotation(
                degrees=augmentation_config["random_rotation"]["degrees"]
            ))
        if augmentation_config["horizontal_flip"]["enabled"]:
            transform_list.append(transforms.RandomHorizontalFlip(
                p=augmentation_config["horizontal_flip"]["probability"]
            ))
        if augmentation_config["color_jitter"]["enabled"]:
            transform_list.append(transforms.ColorJitter(
                brightness=augmentation_config["color_jitter"]["brightness"],
                contrast=augmentation_config["color_jitter"]["contrast"],
                saturation=augmentation_config["color_jitter"]["saturation"],
                hue=augmentation_config["color_jitter"]["hue"]
            ))

    # Normalization
    transform_list.append(transforms.Normalize(
        mean=config["dataset"]["mean"],
        std=config["dataset"]["std"]
    ))

    return transforms.Compose(transform_list)

def download_and_extract(url, extract_to="./data"):
    """
    Download a file from a URL, extract it to a temporary folder, and organize the data into train/test folders.

    Args:
        url (str): The URL of the file to download.
        extract_to (str): The base directory to organize the dataset (e.g., "./data/CIFAR100").

    Returns:
        str: The path to the organized dataset directory.
    """
    # Create the base directory
    os.makedirs(extract_to, exist_ok=True)

    # Create a temporary directory for downloading and extracting
    temp_dir = os.path.join("temp", os.path.basename(extract_to))
    os.makedirs(temp_dir, exist_ok=True)

    # Get the filename from the URL
    filename = os.path.join(temp_dir, url.split("/")[-1])

    # Download the file with a progress bar
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Get the total file size
        total_size = int(response.headers.get("content-length", 0))

        # Download the file in chunks
        with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        print(f"Download complete: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

    # Extract the file
    print(f"Extracting {filename}...")
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(filename, "r:gz") as tar_ref:
                tar_ref.extractall(temp_dir)
        elif filename.endswith(".tar"):
            with tarfile.open(filename, "r:") as tar_ref:
                tar_ref.extractall(temp_dir)
        else:
            print(f"Unsupported file format: {filename}")
            return None

        print(f"Extraction complete: {temp_dir}")
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"Failed to extract {filename}: {e}")
        return None

    # Organize the data into train/test folders
    organize_data(temp_dir, extract_to)

    # Create JSON configuration file
    create_json_config(extract_to)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")

    return extract_to

def organize_data(temp_dir, extract_to):
    """
    Organize the dataset into train/test folders.

    Args:
        temp_dir (str): The temporary directory containing the extracted dataset.
        extract_to (str): The base directory to organize the dataset (e.g., "./data/CIFAR100").
    """
    # Create train and test folders
    train_dir = os.path.join(extract_to, "train")
    test_dir = os.path.join(extract_to, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Search for images and organize them
    print("Organizing data into train/test folders...")
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                # Move the image to the train or test folder
                src_path = os.path.join(root, file)
                if "train" in root.lower():
                    dest_dir = train_dir
                elif "test" in root.lower():
                    dest_dir = test_dir
                else:
                    # Default to train folder
                    dest_dir = train_dir

                # Create subdirectories based on labels (if available)
                label = os.path.basename(root)
                dest_dir = os.path.join(dest_dir, label)
                os.makedirs(dest_dir, exist_ok=True)

                # Move the file
                dest_path = os.path.join(dest_dir, file)
                shutil.move(src_path, dest_path)

    print(f"Data organized into {train_dir} and {test_dir}")

def create_json_config(data_dir):
    """
    Create a JSON configuration file for the dataset.

    Args:
        data_dir (str): The directory containing the organized dataset.
    """
    # Determine input size and number of classes
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Get the first image to determine input size
    first_image_path = None
    for root, _, files in os.walk(train_dir):
        if files:
            first_image_path = os.path.join(root, files[0])
            break

    if first_image_path:
        image = Image.open(first_image_path)
        input_size = list(image.size)  # Width x Height
        in_channels = 1 if image.mode == "L" else 3  # Grayscale or RGB
    else:
        input_size = [32, 32]  # Default size
        in_channels = 3  # Default channels

    # Count the number of classes
    num_classes = len(os.listdir(train_dir))

    # Create JSON configuration
    config = {
        "dataset": {
            "name": os.path.basename(data_dir),
            "type": "custom",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "input_size": input_size,
            "mean": [0.5] * in_channels,  # Default mean
            "std": [0.5] * in_channels,  # Default std
            "train_dir": train_dir,
            "test_dir": test_dir,
            "image_type": "grayscale" if in_channels == 1 else "rgb"
        }
    }

    # Save JSON file
    json_path = os.path.join(data_dir, f"{os.path.basename(data_dir)}.json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Created JSON configuration file at {json_path}")

#-------------------------------

def save_latent_space(latent, dataset_name, filename="latent.pkl"):
    """Save the latent space as a pickle file."""
    data_dir = f"data/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)

    # Save as pickle
    pkl_path = os.path.join(data_dir, filename)
    with open(pkl_path, "wb") as f:
        pickle.dump(latent.cpu().numpy(), f)

    print(f"Latent space saved to {pkl_path}")

def save_embeddings_as_csv(embeddings, dataset_name, filename="embeddings.csv"):
    """Save the embedded tensors as a flattened 1D CSV file."""
    data_dir = f"data/{dataset_name}"
    os.makedirs(data_dir, exist_ok=True)

    # Flatten and convert to numpy array
    embeddings = embeddings.flatten().detach().cpu().numpy()

    # Save as CSV
    csv_path = os.path.join(data_dir, filename)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(embeddings)

    print(f"Embeddings saved to {csv_path}")

def visualize_embeddings(embeddings, labels, title="Embedding Space"):
    """Visualize the embedding space using t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10")
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def get_device():
    """Detect and return the best available device (CPU, GPU, or TPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon GPUs
        return torch.device("mps")
    elif torch.xpu.is_available():  # For Intel GPUs
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def postprocess_image(image_tensor):
    """Convert a tensor back to a PIL image."""
    print(f"Input tensor shape: {image_tensor.shape}")  # Debug: Check input tensor shape

    if image_tensor.dim() == 4:  # If batch dimension exists
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
    print(f"Tensor shape after removing batch dimension: {image_tensor.shape}")  # Debug

    if image_tensor.shape[0] > 4:  # Check for invalid number of channels
        raise ValueError(f"Invalid number of channels: {image_tensor.shape[0]}. Expected 1, 3, or 4.")

    # Convert tensor to PIL image
    image = transforms.ToPILImage()(image_tensor)
    return imagem(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

import os
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from PIL import Image
from autoencoder.model import ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv
from autoencoder.data_loader import load_dataset_config


def load_model(checkpoint_path, device):
    """Load the trained autoencoder model."""
    model = Autoencoder(latent_dim=128, embedding_dim=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def preprocess_image(image, device, config):
    """
    Preprocess the input image according to dataset config.
    Args:
        image: Can be a file path (str), PIL Image, or Streamlit UploadedFile.
        device: The device to load the tensor onto.
        config: Configuration dictionary.
    Returns:
        image_tensor: Preprocessed image tensor.
    """
    # Get channel info from config
    in_channels = config["dataset"]["in_channels"]
    input_size = tuple(config["dataset"]["input_size"])

    transform_list = [
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ]

    # Add grayscale conversion if needed
    if in_channels == 1:
        transform_list.insert(0, transforms.Grayscale(num_output_channels=1))

    transform = transforms.Compose(transform_list)

    # Handle different input types
    if isinstance(image, str):  # File path
        image = Image.open(image)
    elif hasattr(image, "read"):  # Streamlit UploadedFile or file-like object
        image = Image.open(image)
    elif isinstance(image, Image.Image):  # PIL Image
        pass
    else:
        raise ValueError("Unsupported image input type. Expected file path, PIL Image, or file-like object.")

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def enhance_features(latent, embedding, model, enhancement_factor=2.0):
    """
    Enhance decisive features in the latent space based on the embedding.
    - latent: Latent space tensor.
    - embedding: Embedded tensor.
    - model: Trained autoencoder model.
    - enhancement_factor: Factor to enhance decisive features.
    """
    # Identify decisive features (e.g., regions with high embedding values)
    decisive_features = torch.abs(embedding).mean(dim=1, keepdim=True)

    # Enhance decisive features in the latent space
    enhanced_latent = latent + enhancement_factor * decisive_features

    # Reconstruct the image
    with torch.no_grad():
        reconstructed_image = model.decoder(enhanced_latent)

    return reconstructed_image

def save_reconstructed_image(original_tensor, reconstructed_tensor, dataset_name, filename="comparison.png"):
    """Save the original and reconstructed images side by side."""
    # Create reconstructed images directory
    recon_dir = f"data/{dataset_name}/reconstructed_images"
    os.makedirs(recon_dir, exist_ok=True)

    # Convert tensors to PIL images
    original_image = transforms.ToPILImage()(original_tensor.squeeze(0).cpu())
    reconstructed_image = transforms.ToPILImage()(reconstructed_tensor.squeeze(0).cpu())

    # Create a new image with both original and reconstructed
    total_width = original_image.width * 2
    max_height = original_image.height

    comparison_image = Image.new('RGB', (total_width, max_height))

    # Paste the images
    comparison_image.paste(original_image, (0, 0))
    comparison_image.paste(reconstructed_image, (original_image.width, 0))

    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison_image)
    draw.text((10, 10), "Original", fill="white")
    draw.text((original_image.width + 10, 10), "Reconstructed", fill="white")

    # Save the comparison image
    image_path = os.path.join(recon_dir, filename)
    comparison_image.save(image_path)
    print(f"Comparison image saved to {image_path}")

def reconstruct_image(path, checkpoint_path, dataset_name, config):
    """Handle both single image and folder reconstruction."""
    if os.path.isdir(path):
        print(f"Processing directory: {path}")
        reconstruct_folder(path, checkpoint_path, dataset_name, config)
    else:
        print(f"Processing single image: {path}")
        device = get_device()

        model = ModifiedAutoencoder(config, device=device).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        image_tensor = preprocess_image(path, device, config)

        with torch.no_grad():
            reconstructed, latent_1d = model(image_tensor)
            save_reconstructed_image(
                image_tensor,
                reconstructed,
                dataset_name,
                filename=os.path.basename(path)
            )

def reconstruct_from_latent(csv_path, checkpoint_path, dataset_name, config):
    """Reconstruct images from latent CSV files with robust checkpoint loading."""
    device = get_device()
    model = ModifiedAutoencoder(config, device=device).to(device)

    print(f"Loading model checkpoint from {checkpoint_path}...")

    # Use the improved load_checkpoint function
    model, _, _ = load_checkpoint(checkpoint_path, model, config)
    model.eval()

    # Verify frequencies are loaded
    if not hasattr(model.latent_mapper, 'frequencies') or model.latent_mapper.frequencies is None:
        raise RuntimeError("Frequencies not properly initialized after loading checkpoint")

    print(f"Model loaded successfully with frequencies shape: {model.latent_mapper.frequencies.shape}")

    # Process files
    if os.path.isfile(csv_path):
        files = [csv_path]
    else:
        files = [f for f in os.listdir(csv_path) if f.endswith('_latent.csv')]
        files = [os.path.join(csv_path, f) for f in files]

    for csv_file in tqdm(files, desc="Reconstructing from latent representations"):
        try:
            latent_data, metadata = load_1d_latent_from_csv(csv_file)
            latent_1d = latent_data.to(device)

            with torch.no_grad():
                decoded_flat = model.latent_mapper.inverse_map(latent_1d)
                decoded_volume = decoded_flat.view(1, 512, 1, 1)
                reconstructed = model.decoder(decoded_volume)
                reconstructed = model.adaptive_upsample(reconstructed)

            output_name = os.path.basename(csv_file).replace('_latent.csv', '_reconstructed.png')
            save_reconstructed_image(None, reconstructed, dataset_name, output_name)

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")


def reconstruct_folder(input_dir, checkpoint_path, dataset_name, config):
    """Reconstruct all images in a directory structure maintaining the hierarchy."""
    device = get_device()

    # Use ModifiedAutoencoder instead of Autoencoder
    model = ModifiedAutoencoder(config, device=device).to(device)

    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create base reconstruction directory
    recon_base_dir = f"data/{dataset_name}/reconstructed_images"
    os.makedirs(recon_base_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        recon_dir = os.path.join(recon_base_dir, relative_path)
        os.makedirs(recon_dir, exist_ok=True)

        print(f"Processing directory: {relative_path}")

        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                recon_path = os.path.join(recon_dir, file)

                try:
                    image_tensor = preprocess_image(input_path, device,config)
                    with torch.no_grad():
                        # Modified forward pass returns only reconstructed and latent_1d
                        reconstructed, latent_1d = model(image_tensor)

                    # Save reconstructed image
                    image = transforms.ToPILImage()(reconstructed.squeeze(0).cpu())
                    image.save(recon_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")



# Example usage
if __name__ == "__main__":
    image_path = "path/to/input/image.png"  # Replace with the path to your input image
    checkpoint_path = "Model/checkpoint/Best_CIFAR10.pth"  # Replace with the path to your model checkpoint
    dataset_name = "CIFAR10"  # Replace with the dataset name
    reconstruct_image(image_path, checkpoint_path, dataset_name)
