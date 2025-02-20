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

import os
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from PIL import Image
from autoencoder.model import ModifiedAutoencoder


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
    create_json_config(data_dir)
    # Cleanup
    shutil.rmtree(temp_dir)
    return data_dir


def save_1d_latent_to_csv(latent_1d, image_path, dataset_name, metadata=None):
    """
    Save 1D latent representation to CSV with metadata, maintaining original folder hierarchy.

    Args:
        latent_1d: PyTorch tensor containing the latent representation
        image_path: Full path to the original image
        dataset_name: Name of the dataset
        metadata: Optional dictionary of additional metadata
    """
    # Create base latent space directory
    base_latent_dir = f"data/{dataset_name}/latent_space"

    # Get the relative path from the dataset directory to maintain hierarchy
    base_data_dir = os.path.abspath(f"data/{dataset_name}")
    abs_image_path = os.path.abspath(image_path)

    # Extract relative path while preserving class structure
    try:
        rel_path = os.path.relpath(abs_image_path, base_data_dir)
    except ValueError:
        # If the paths are on different drives or have other issues
        # Fall back to extracting class name from the path
        rel_path = '/'.join(abs_image_path.split(os.sep)[-3:])  # Take last 3 components

    target_dir = os.path.join(base_latent_dir, os.path.dirname(rel_path))

    # Create all necessary directories
    os.makedirs(target_dir, exist_ok=True)

    # Get original filename without extension
    original_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create the CSV filename with the same basename as the image
    csv_path = os.path.join(target_dir, f"{original_filename}.csv")

    # Convert latent values to numpy and flatten if needed
    if isinstance(latent_1d, torch.Tensor):
        latent_values = latent_1d.detach().cpu().numpy().flatten()
    else:
        latent_values = latent_1d.flatten()

    # Create DataFrame with metadata and latent values
    data = {
        'type': ['metadata'] * (len(metadata) if metadata else 0) + ['latent_values'],
        'key': (list(metadata.keys()) if metadata else []) + ['values'],
        'value': (list(map(str, metadata.values())) if metadata else []) + [','.join(map(str, latent_values))]
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


def save_batch_latents(batch_latents, image_paths, dataset_name, batch_metadata=None):
    """Save latent representations with unique identifiers."""

    # Initialize tracking
    base_data_dir = os.path.abspath(f"data/{dataset_name}")
    base_latent_dir = os.path.join(base_data_dir, "latent_space")

    # Create unique batch identifier
    batch_id = f"epoch_{batch_metadata['epoch']}_batch_{batch_metadata['batch']}"

    # Process each image in batch
    for idx, (latent, path) in enumerate(zip(batch_latents, image_paths)):
        try:
            # Get proper path structure
            abs_image_path = os.path.abspath(path)
            path_parts = abs_image_path.split(os.sep)
            train_idx = path_parts.index("train")
            class_name = path_parts[train_idx + 1]

            # Create unique target directory including batch info
            target_dir = os.path.join(base_latent_dir, "train", class_name)
            os.makedirs(target_dir, exist_ok=True)

            # Create unique filename using batch and image info
            original_filename = os.path.splitext(os.path.basename(path))[0]
            unique_filename = f"{original_filename}_{batch_id}.csv"
            csv_path = os.path.join(target_dir, unique_filename)

            # Save with metadata
            metadata = {
                'batch_id': batch_id,
                'image_idx': idx,
                'class': class_name,
                'original_path': path,
                'timestamp': datetime.now().isoformat()
            }
            if batch_metadata:
                metadata.update(batch_metadata)

            latent_values = latent.detach().cpu().numpy().flatten()
            data = {
                'type': ['metadata'] * len(metadata) + ['latent_values'],
                'key': list(metadata.keys()) + ['values'],
                'value': list(map(str, metadata.values())) + [','.join(map(str, latent_values))]
            }

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

        except Exception as e:
            print(f"Error saving latent for {path}: {str(e)}")
            continue




def reconstruct_from_latent(latent_dir, checkpoint_path, dataset_name, config):
    """
    Reconstruct images from latent CSV files maintaining the original folder hierarchy.

    Args:
        latent_dir: Directory containing latent space CSV files
        checkpoint_path: Path to the model checkpoint
        dataset_name: Name of the dataset
        config: Model configuration dictionary
    """
    device = get_device()
    print(f"Using device: {device}")

    # Initialize model
    model = ModifiedAutoencoder(config, device=device).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create base reconstruction directory
    base_recon_dir = f"data/{dataset_name}/reconstructed_images"
    os.makedirs(base_recon_dir, exist_ok=True)

    def process_directory(current_dir):
        """Process a directory and its subdirectories recursively."""
        # Get relative path from latent directory
        rel_path = os.path.relpath(current_dir, latent_dir)

        # Create corresponding reconstruction directory
        recon_dir = os.path.join(base_recon_dir, rel_path)
        os.makedirs(recon_dir, exist_ok=True)

        # Process all CSV files in current directory
        csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"Processing {rel_path}: {len(csv_files)} files")

            for file in tqdm(csv_files, desc=f"Processing {rel_path}"):
                try:
                    csv_path = os.path.join(current_dir, file)

                    # Load latent data from CSV
                    df = pd.read_csv(csv_path)
                    latent_row = df[df['type'] == 'latent_values'].iloc[0]
                    latent_values = np.array([float(x) for x in latent_row['value'].split(',')]).reshape(1, -1)
                    latent_tensor = torch.tensor(latent_values, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        # Inverse map the latent values
                        decoded_flat = model.latent_mapper.inverse_map(latent_tensor)
                        decoded_volume = decoded_flat.view(1, -1, 1, 1)
                        reconstructed = model.decoder(decoded_volume)

                        if hasattr(model, 'adaptive_upsample'):
                            reconstructed = model.adaptive_upsample(reconstructed)

                    # Save reconstructed image
                    output_name = os.path.splitext(file)[0] + '.png'
                    output_path = os.path.join(recon_dir, output_name)

                    # Convert and save
                    image = transforms.ToPILImage()(reconstructed.squeeze(0).cpu())
                    image.save(output_path)

                except Exception as e:
                    print(f"Error processing {csv_path}: {str(e)}")
                    continue

        # Process subdirectories
        subdirs = [d for d in os.listdir(current_dir)
                  if os.path.isdir(os.path.join(current_dir, d))]
        for subdir in subdirs:
            process_directory(os.path.join(current_dir, subdir))

    # Start processing from root directory
    process_directory(latent_dir)
    print("Reconstruction complete!")

def load_1d_latent_from_csv(csv_path):
    """
    Load 1D latent representation from CSV with metadata.

    Returns:
        tuple: (latent_tensor, metadata_dict)
    """
    try:
        df = pd.read_csv(csv_path)

        # Extract metadata
        metadata = {}
        metadata_rows = df[df['type'] == 'metadata']
        for _, row in metadata_rows.iterrows():
            metadata[row['key']] = row['value']

        # Extract latent values
        latent_row = df[df['type'] == 'latent_values'].iloc[0]
        latent_values = [float(x) for x in latent_row['value'].split(',')]

        # Convert to tensor
        latent_tensor = torch.tensor(latent_values)

        return latent_tensor, metadata

    except Exception as e:
        raise ValueError(f"Error loading latent values from CSV: {str(e)}")



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
    try:
        # Get device from model's parameters
        device = next(model.parameters()).device

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

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
            cleaned_state_dict[name] = v.to(device)  # Ensure weights are on the correct device

        # Ensure frequencies are in state dict
        if 'latent_mapper.frequencies' not in cleaned_state_dict:
            cleaned_state_dict['latent_mapper.frequencies'] = model.latent_mapper.frequencies.to(device)

        # Load state dict
        model.load_state_dict(cleaned_state_dict, strict=True)

        # Ensure the model is on the correct device
        model = model.to(device)

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
    print(f"Found {num_classes} classes of images")
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
    """
    Save reconstructed images, with optional original image comparison.

    Args:
        original_tensor: Original image tensor or None if unavailable
        reconstructed_tensor: Reconstructed image tensor
        dataset_name: Name of the dataset
        filename: Output filename
    """
    # Create reconstructed images directory
    recon_dir = f"data/{dataset_name}/reconstructed_images"
    os.makedirs(recon_dir, exist_ok=True)

    # Convert reconstructed tensor to PIL image
    reconstructed_image = transforms.ToPILImage()(reconstructed_tensor.squeeze(0).cpu())

    if original_tensor is not None:
        # If we have an original image, create a comparison
        original_image = transforms.ToPILImage()(original_tensor.squeeze(0).cpu())

        # Create a new image with both original and reconstructed
        total_width = original_image.width * 2
        max_height = original_image.height
        comparison_image = Image.new('RGB', (total_width, max_height))

        # Paste the images
        comparison_image.paste(original_image, (0, 0))
        comparison_image.paste(reconstructed_image, (original_image.width, 0))

        # Add labels
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison_image)
        draw.text((10, 10), "Original", fill="white")
        draw.text((original_image.width + 10, 10), "Reconstructed", fill="white")

        output_image = comparison_image
    else:
        # If no original image, just save the reconstruction
        output_image = reconstructed_image

    # Save the image
    image_path = os.path.join(recon_dir, filename)
    output_image.save(image_path)
    #print(f"Image saved to {image_path}")

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


def load_local_dataset(dataset_name, transform=None):
    """Load a dataset from a local directory."""
    # Load config first
    config = load_dataset_config(dataset_name)

    if transform is None:
        # Use dataset-specific normalization from config
        transform_list = []

        # Only apply grayscale transformation if in_channels is 1
        if config['in_channels'] == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Add ToTensor and Normalize transformations
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std'])
        ])

        transform = transforms.Compose(transform_list)

    data_dir = f"data/{dataset_name}/train/"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Print class distribution
    class_counts = {cls: 0 for cls in dataset.classes}
    for _, label in dataset:
        class_counts[dataset.classes[label]] += 1

    print("Class distribution in dataset:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")

    return dataset

def load_dataset_config(dataset_name):
    """Load dataset configuration from JSON file."""
    config_path = f"data/{dataset_name}/{dataset_name}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["dataset"]


# Example usage
if __name__ == "__main__":
    image_path = "path/to/input/image.png"  # Replace with the path to your input image
    checkpoint_path = "Model/checkpoint/Best_CIFAR10.pth"  # Replace with the path to your model checkpoint
    dataset_name = "CIFAR10"  # Replace with the dataset name
    reconstruct_image(image_path, checkpoint_path, dataset_name)
