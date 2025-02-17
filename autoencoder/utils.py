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
import pandas as pd


def download_and_extract(url, extract_to="./data"):
    """
    Download a file from a URL and extract it to the specified directory.
    
    Args:
        url (str): The URL of the file to download.
        extract_to (str): The directory to extract the file to.
    
    Returns:
        str: The path to the extracted directory.
    """
    # Create the extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Get the filename from the URL
    filename = os.path.join(extract_to, url.split("/")[-1])
    
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
                zip_ref.extractall(extract_to)
        elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(filename, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        elif filename.endswith(".tar"):
            with tarfile.open(filename, "r:") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported file format: {filename}")
            return None
        
        print(f"Extraction complete: {extract_to}")
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"Failed to extract {filename}: {e}")
        return None
    
    # Process the extracted data
    process_data(extract_to)
    
    return extract_to

def process_data(data_dir):
    """
    Process data in the specified directory.
    
    Args:
        data_dir (str): The directory containing the dataset files.
    """
    # Detect and process the data format
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".csv") or file.endswith(".tsv") or file.endswith(".xlsx"):
                process_tabular_data(file_path, data_dir)
            elif file.endswith(".idx3-ubyte") or file.endswith(".idx1-ubyte"):
                process_mnist(data_dir)
            elif file.endswith((".png", ".jpg", ".jpeg")):
                process_image_data(file_path, data_dir)
            elif file.endswith(".json"):
                process_json_data(file_path, data_dir)
            elif file.endswith(".h5"):
                process_hdf5_data(file_path, data_dir)
            elif file.endswith(".npy"):
                process_numpy_data(file_path, data_dir)
            else:
                print(f"Unsupported file format: {file_path}")

def process_tabular_data(file_path, data_dir):
    """
    Process tabular data (e.g., CSV, TSV, Excel).
    
    Args:
        file_path (str): Path to the tabular data file.
        data_dir (str): Directory to save processed data.
    """
    print(f"Processing tabular data: {file_path}")
    if file_path.endswith(".csv") or file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep="\t" if file_path.endswith(".tsv") else ",")
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    
    # Save processed data
    output_path = os.path.join(data_dir, "processed", "tabular.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Tabular data saved to {output_path}")

def process_image_data(file_path, data_dir):
    """
    Process image data (e.g., PNG, JPG).
    
    Args:
        file_path (str): Path to the image file.
        data_dir (str): Directory to save processed data.
    """
    print(f"Processing image data: {file_path}")
    image = Image.open(file_path)
    output_path = os.path.join(data_dir, "processed", "images", os.path.basename(file_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")

def process_mnist(data_dir):
    """
    Process MNIST dataset files (e.g., .idx3-ubyte and .idx1-ubyte).
    
    Args:
        data_dir (str): The directory containing the MNIST dataset files.
    """
    print("Processing MNIST dataset...")
    train_images_path = os.path.join(data_dir, "raw", "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "raw", "train-labels-idx1-ubyte")
    test_images_path = os.path.join(data_dir, "raw", "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "raw", "t10k-labels-idx1-ubyte")
    
    # Create directories for processed data
    train_dir = os.path.join(data_dir, "processed", "train")
    test_dir = os.path.join(data_dir, "processed", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process training data
    train_images = read_idx3_ubyte(train_images_path)
    train_labels = read_idx1_ubyte(train_labels_path)
    save_images_and_labels(train_images, train_labels, train_dir)
    
    # Process test data
    test_images = read_idx3_ubyte(test_images_path)
    test_labels = read_idx1_ubyte(test_labels_path)
    save_images_and_labels(test_images, test_labels, test_dir)
    
    print(f"MNIST dataset processed and saved to {data_dir}")

def read_idx3_ubyte(file_path):
    """
    Read images from an IDX3-UBYTE file.
    
    Args:
        file_path (str): Path to the IDX3-UBYTE file.
    
    Returns:
        np.ndarray: Array of images.
    """
    with open(file_path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        num_cols = int.from_bytes(f.read(4), byteorder="big")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

def read_idx1_ubyte(file_path):
    """
    Read labels from an IDX1-UBYTE file.
    
    Args:
        file_path (str): Path to the IDX1-UBYTE file.
    
    Returns:
        np.ndarray: Array of labels.
    """
    with open(file_path, "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images_and_labels(images, labels, output_dir):
    """
    Save images and labels to a directory.
    
    Args:
        images (np.ndarray): Array of images.
        labels (np.ndarray): Array of labels.
        output_dir (str): Directory to save the images and labels.
    """
    for i, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, f"{i}.png")
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
