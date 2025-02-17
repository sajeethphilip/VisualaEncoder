from torchvision import transforms
import csv
import torch
import os
import csv
import torch
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

def download_and_extract(url, dataset_name, extract_to="./data"):
    """
    Download a file from a URL, extract it, and intelligently process it into a standardized format.
    
    Args:
        url (str): The URL of the file to download.
        dataset_name (str): Name of the dataset (e.g., 'MNIST', 'CIFAR10').
        extract_to (str): Base directory for all datasets.
    
    Returns:
        str: The path to the processed dataset directory.
    """
    # Create the dataset-specific directories
    dataset_dir = os.path.join(extract_to, dataset_name)
    raw_dir = os.path.join(dataset_dir, "raw")
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    for directory in [raw_dir, train_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Download and extract files
    filename = download_file(url, raw_dir)
    if filename:
        extract_file(filename, raw_dir)
    
    # Process the extracted data
    process_dataset(dataset_dir)
    
    # Generate dataset info
    generate_dataset_info(dataset_dir)
    
    return dataset_dir

def download_file(url, raw_dir):
    """Download a file with progress bar."""
    try:
        filename = os.path.join(raw_dir, url.split("/")[-1])
        print(f"Downloading {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
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
        return filename
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def extract_file(filename, extract_to):
    """Extract compressed files."""
    try:
        if filename.endswith(".gz") and not (filename.endswith(".tar.gz") or filename.endswith(".tgz")):
            with gzip.open(filename, 'rb') as f_in:
                with open(filename[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(filename, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        elif filename.endswith(".tar"):
            with tarfile.open(filename, "r:") as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"Extraction complete: {extract_to}")
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")

def process_dataset(dataset_dir):
    """
    Intelligently process dataset based on content analysis.
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    # Analyze content of raw directory
    files = os.listdir(raw_dir)
    
    # Check for IDX format files (MNIST-like)
    idx_files = [f for f in files if f.endswith('-idx3-ubyte') or f.endswith('-idx1-ubyte')]
    if idx_files:
        process_idx_files(dataset_dir)
        return
    
    # Check for image directories
    image_files = []
    for root, _, filenames in os.walk(raw_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, filename))
    
    if image_files:
        process_image_files(dataset_dir)

def process_idx_files(dataset_dir):
    """Process IDX format files."""
    raw_dir = os.path.join(dataset_dir, "raw")
    files = os.listdir(raw_dir)
    
    # Find training and test files
    train_images_file = next((f for f in files if 'train' in f and 'images' in f), None)
    train_labels_file = next((f for f in files if 'train' in f and 'labels' in f), None)
    test_images_file = next((f for f in files if ('t10k' in f or 'test' in f) and 'images' in f), None)
    test_labels_file = next((f for f in files if ('t10k' in f or 'test' in f) and 'labels' in f), None)
    
    if train_images_file and train_labels_file:
        process_split(dataset_dir, "train", train_images_file, train_labels_file)
    
    if test_images_file and test_labels_file:
        process_split(dataset_dir, "test", test_images_file, test_labels_file)

def process_split(dataset_dir, split, images_file, labels_file):
    """Process a single split of IDX format data."""
    images = read_idx3_ubyte(os.path.join(dataset_dir, "raw", images_file))
    labels = read_idx1_ubyte(os.path.join(dataset_dir, "raw", labels_file))
    
    split_dir = os.path.join(dataset_dir, split)
    for i, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(split_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        image_path = os.path.join(label_dir, f"{i}.png")
        Image.fromarray(image).save(image_path)

def process_image_files(dataset_dir):
    """Process regular image files."""
    raw_dir = os.path.join(dataset_dir, "raw")
    
    # Try to detect train/test split
    train_dir = next((d for d in os.listdir(raw_dir) if 'train' in d.lower()), None)
    test_dir = next((d for d in os.listdir(raw_dir) if 'test' in d.lower()), None)
    
    if train_dir:
        process_image_split(os.path.join(raw_dir, train_dir), os.path.join(dataset_dir, "train"))
    else:
        # If no train/test split found, process all as training data
        process_image_split(raw_dir, os.path.join(dataset_dir, "train"))
    
    if test_dir:
        process_image_split(os.path.join(raw_dir, test_dir), os.path.join(dataset_dir, "test"))

def process_image_split(src_dir, dst_dir):
    """Process a directory of images."""
    # Check if images are already in class directories
    has_class_dirs = any(os.path.isdir(os.path.join(src_dir, d)) for d in os.listdir(src_dir))
    
    if has_class_dirs:
        # Process each class directory
        for class_name in os.listdir(src_dir):
            class_path = os.path.join(src_dir, class_name)
            if os.path.isdir(class_path):
                dst_class_dir = os.path.join(dst_dir, class_name)
                os.makedirs(dst_class_dir, exist_ok=True)
                
                for i, img_file in enumerate(os.listdir(class_path)):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_img = os.path.join(class_path, img_file)
                        dst_img = os.path.join(dst_class_dir, f"{i}.png")
                        process_and_save_image(src_img, dst_img)
    else:
        # All images are in one directory, create a single class
        os.makedirs(dst_dir, exist_ok=True)
        for i, img_file in enumerate(os.listdir(src_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_img = os.path.join(src_dir, img_file)
                dst_img = os.path.join(dst_dir, "0", f"{i}.png")
                os.makedirs(os.path.join(dst_dir, "0"), exist_ok=True)
                process_and_save_image(src_img, dst_img)

def generate_dataset_info(dataset_dir):
    """Generate dataset info by analyzing the processed data."""
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    # Get sample image to determine properties
    sample_image = None
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                sample_image = Image.open(os.path.join(root, file))
                break
        if sample_image:
            break
    
    if sample_image:
        # Determine image properties
        if sample_image.mode == 'L':
            in_channels = 1
            image_type = 'grayscale'
        elif sample_image.mode == 'RGB':
            in_channels = 3
            image_type = 'rgb'
        else:
            in_channels = len(sample_image.getbands())
            image_type = sample_image.mode.lower()
        
        input_size = sample_image.size
        
        # Count number of classes
        num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        
        info = {
            "dataset": {
                "name": os.path.basename(dataset_dir),
                "type": "custom",
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "mean": [0.5] * in_channels,  # Default values
                "std": [0.5] * in_channels,   # Default values
                "train_dir": train_dir,
                "test_dir": test_dir,
                "image_type": image_type
            }
        }
        
        # Save dataset info
        with open(os.path.join(dataset_dir, "dataset_info.json"), 'w') as f:
            json.dump(info, f, indent=4)

def read_idx3_ubyte(file_path):
    """Read IDX3-UBYTE format image data."""
    with open(file_path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        num_cols = int.from_bytes(f.read(4), byteorder="big")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

def read_idx1_ubyte(file_path):
    """Read IDX1-UBYTE format label data."""
    with open(file_path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def process_and_save_image(src_path, dst_path):
    """Process and save a single image."""
    try:
        with Image.open(src_path) as img:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img.save(dst_path, format='PNG')
    except Exception as e:
        print(f"Error processing image {src_path}: {e}")
        

def save_images_and_labels(images, labels, output_dir):
    """Save images and labels to the standard directory structure."""
    for i, (image, label) in enumerate(zip(images, labels)):
        # Create class directory if it doesn't exist
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        # Save image with standardized naming
        image_path = os.path.join(label_dir, f"{i}.png")
        Image.fromarray(image).save(image_path)
    
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
