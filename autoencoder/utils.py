from torchvision import transforms
import csv
import torch
import os
import csv
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pickle

import requests
import zipfile
import tarfile
from tqdm import tqdm

import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm

def download_and_extract(url, dataset_name, extract_to="./data"):
    """
    Download a file from a URL, extract it, and process according to dataset configuration.
    Maintains a consistent directory structure for all datasets:
    data/
    └── DATASET_NAME/
        ├── raw/
        │   └── (original files)
        ├── train/
        │   └── (class directories with images)
        └── test/
            └── (class directories with images)
    
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
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get the filename from the URL
    filename = os.path.join(raw_dir, url.split("/")[-1])
    
    # Download the file with a progress bar
    print(f"Downloading {url}...")
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
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        print(f"Download complete: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None
    
    # Extract the file if it's compressed
    if filename.endswith((".gz", ".zip", ".tar", ".tar.gz", ".tgz")):
        print(f"Extracting {filename}...")
        try:
            if filename.endswith(".gz") and not (filename.endswith(".tar.gz") or filename.endswith(".tgz")):
                # Single gzipped file
                with gzip.open(filename, 'rb') as f_in:
                    with open(filename[:-3], 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(filename)  # Remove the compressed file
            elif filename.endswith(".zip"):
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall(raw_dir)
            elif filename.endswith((".tar.gz", ".tgz")):
                with tarfile.open(filename, "r:gz") as tar_ref:
                    tar_ref.extractall(raw_dir)
            elif filename.endswith(".tar"):
                with tarfile.open(filename, "r:") as tar_ref:
                    tar_ref.extractall(raw_dir)
            
            print(f"Extraction complete: {raw_dir}")
        except Exception as e:
            print(f"Failed to extract {filename}: {e}")
            return None
    
    return dataset_dir

def process_idx_dataset(dataset_dir, train_images_file, train_labels_file, 
                       test_images_file, test_labels_file):
    """
    Process IDX format datasets (like MNIST) into the standard directory structure.
    
    Args:
        dataset_dir (str): Base directory for the dataset.
        train_images_file (str): Name of the training images file.
        train_labels_file (str): Name of the training labels file.
        test_images_file (str): Name of the test images file.
        test_labels_file (str): Name of the test labels file.
    """
    # Process training data
    print("Processing training data...")
    train_images = read_idx3_ubyte(os.path.join(dataset_dir, "raw", train_images_file))
    train_labels = read_idx1_ubyte(os.path.join(dataset_dir, "raw", train_labels_file))
    save_images_and_labels(train_images, train_labels, os.path.join(dataset_dir, "train"))
    
    # Process test data
    print("Processing test data...")
    test_images = read_idx3_ubyte(os.path.join(dataset_dir, "raw", test_images_file))
    test_labels = read_idx1_ubyte(os.path.join(dataset_dir, "raw", test_labels_file))
    save_images_and_labels(test_images, test_labels, os.path.join(dataset_dir, "test"))

def process_image_dataset(dataset_dir, train_dir="train", test_dir="test"):
    """
    Process image-based datasets into the standard directory structure.
    Assumes images are organized in subdirectories by class.
    
    Args:
        dataset_dir (str): Base directory for the dataset.
        train_dir (str): Name of the training directory in the raw data.
        test_dir (str): Name of the test directory in the raw data.
    """
    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        src_dir = os.path.join(dataset_dir, "raw", split_dir)
        dst_dir = os.path.join(dataset_dir, split)
        
        # Process each class directory
        for class_dir in os.listdir(src_dir):
            class_path = os.path.join(src_dir, class_dir)
            if os.path.isdir(class_path):
                dst_class_dir = os.path.join(dst_dir, class_dir)
                os.makedirs(dst_class_dir, exist_ok=True)
                
                # Process each image in the class directory
                for i, img_file in enumerate(os.listdir(class_path)):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_img_path = os.path.join(class_path, img_file)
                        dst_img_path = os.path.join(dst_class_dir, f"{i}.png")
                        process_and_save_image(src_img_path, dst_img_path)

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
