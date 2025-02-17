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

def download_and_extract(url, extract_to="./data"):
    """
    Download a file from a URL or load from torchvision and organize into train/test folders.
    
    Args:
        url (str): The URL of the file to download or dataset name from torchvision.
        extract_to (str): The directory to extract the file to.
    
    Returns:
        str: The path to the processed directory.
    """
    dataset_name = url.split("/")[-1].split(".")[0] if "/" in url else url
    base_dir = os.path.join(extract_to, dataset_name)
    raw_dir = os.path.join(base_dir, "raw")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # Create necessary directories
    for dir_path in [raw_dir, train_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Check if it's a torchvision dataset
    if "/" not in url and hasattr(datasets, url):
        return process_torchvision_dataset(url, base_dir)
    
    # Download and extract if it's a URL
    filename = download_file(url, raw_dir)
    if not filename:
        return None
    
    # Extract the downloaded file
    extracted_path = extract_file(filename, raw_dir)
    if not extracted_path:
        return None
    
    # Process the extracted data
    process_extracted_data(raw_dir, train_dir, test_dir)
    
    return base_dir

def download_file(url, raw_dir):
    """Download file from URL with progress bar."""
    filename = os.path.join(raw_dir, url.split("/")[-1])
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

def extract_file(filename, extract_to):
    """Extract compressed file to specified directory."""
    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(filename, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        elif filename.endswith(".tar"):
            with tarfile.open(filename, "r:") as tar_ref:
                tar_ref.extractall(extract_to)
        elif filename.endswith(".gz"):
            output_file = filename[:-3]
            with gzip.open(filename, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return extract_to
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")
        return None

def process_torchvision_dataset(dataset_name, base_dir):
    """Process built-in torchvision datasets."""
    dataset_class = getattr(datasets, dataset_name)
    
    # Download and process training data
    train_dataset = dataset_class(base_dir, train=True, download=True)
    test_dataset = dataset_class(base_dir, train=False, download=True)
    
    # Create train/test directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process and save training data
    process_dataset(train_dataset, train_dir)
    process_dataset(test_dataset, test_dir)
    
    return base_dir

def process_dataset(dataset, output_dir):
    """Process and save dataset images to appropriate directories."""
    for idx, (data, label) in enumerate(dataset):
        # Create label directory
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        # Convert data to image and save
        if isinstance(data, torch.Tensor):
            data = transforms.ToPILImage()(data)
        elif isinstance(data, np.ndarray):
            data = Image.fromarray(data)
        
        data.save(os.path.join(label_dir, f"{idx}.png"))

def process_extracted_data(raw_dir, train_dir, test_dir):
    """Process extracted data and organize into train/test folders."""
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Determine if file belongs to train or test set
            is_train = "train" in root.lower() or "train" in file.lower()
            output_dir = train_dir if is_train else test_dir
            
            if file.endswith((".png", ".jpg", ".jpeg")):
                process_image_file(file_path, output_dir)
            elif file.endswith((".idx3-ubyte", ".idx1-ubyte")):
                process_idx_file(file_path, output_dir)
            elif file.endswith((".csv", ".tsv")):
                process_tabular_file(file_path, output_dir)

def process_image_file(file_path, output_dir):
    """Process individual image file."""
    try:
        image = Image.open(file_path)
        # Extract label from parent directory name
        label = os.path.basename(os.path.dirname(file_path))
        if not label.isdigit():
            label = "0"  # Default label if none found
        
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Save image with original filename
        new_path = os.path.join(label_dir, os.path.basename(file_path))
        image.save(new_path)
    except Exception as e:
        print(f"Failed to process image {file_path}: {e}")

def process_idx_file(file_path, output_dir):
    """Process IDX format files (MNIST-style datasets)."""
    try:
        if "images" in file_path:
            images = read_idx3_ubyte(file_path)
            labels = read_idx1_ubyte(file_path.replace("images", "labels"))
            save_images_and_labels(images, labels, output_dir)
    except Exception as e:
        print(f"Failed to process IDX file {file_path}: {e}")

def process_tabular_file(file_path, output_dir):
    """Process tabular data files."""
    try:
        df = pd.read_csv(file_path, sep="," if file_path.endswith(".csv") else "\t")
        # Save processed DataFrame
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Failed to process tabular file {file_path}: {e}")

def read_idx3_ubyte(file_path):
    """Read IDX3-UBYTE format image file."""
    with open(file_path, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        size = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        cols = int.from_bytes(f.read(4), byteorder="big")
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(size, rows, cols)

def read_idx1_ubyte(file_path):
    """Read IDX1-UBYTE format label file."""
    with open(file_path, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder="big")
        size = int.from_bytes(f.read(4), byteorder="big")
        return np.frombuffer(f.read(), dtype=np.uint8)

def save_images_and_labels(images, labels, output_dir):
    """Save images with their corresponding labels."""
    for idx, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        image_path = os.path.join(label_dir, f"{idx}.png")
        Image.fromarray(image).save(image_path)
        
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
