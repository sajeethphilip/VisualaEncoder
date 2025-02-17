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

import os
import requests
import zipfile
import tarfile
import gzip
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path

def download_and_extract(source, dataset_name, extract_to="./data"):
    """
    Generic dataset processor that handles various input formats and creates a standardized output.
    
    Args:
        source (str): URL or local path to the dataset
        dataset_name (str): Name of the dataset (will be used as directory name)
        extract_to (str): Base directory for processed datasets
    
    Returns:
        str: Path to the processed dataset directory
    """
    # Setup directory structure
    dataset_dir = os.path.join(extract_to, dataset_name)
    raw_dir = os.path.join(dataset_dir, "raw")
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    for d in [raw_dir, train_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Download if source is URL
    if source.startswith(('http://', 'https://')):
        files = download_dataset(source, raw_dir)
    else:
        files = [source] if os.path.isfile(source) else get_files_from_directory(source)
    
    # Process the raw files
    for file in files:
        process_file(file, raw_dir, train_dir, test_dir)
    
    # Generate dataset info
    generate_dataset_info(dataset_dir)
    
    return dataset_dir

def download_dataset(url, raw_dir):
    """Download dataset from URL."""
    filename = os.path.join(raw_dir, url.split('/')[-1])
    
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Handle compressed files
        return extract_if_compressed(filename, raw_dir)
    except Exception as e:
        print(f"Download failed: {e}")
        return []

def extract_if_compressed(file_path, extract_to):
    """Extract compressed files and return list of extracted files."""
    try:
        if file_path.endswith('.gz') and not file_path.endswith(('.tar.gz', '.tgz')):
            with gzip.open(file_path, 'rb') as f_in:
                extracted_path = file_path[:-3]
                with open(extracted_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            os.remove(file_path)
            return [extracted_path]
            
        elif file_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(extract_to)
            os.remove(file_path)
            return get_files_from_directory(extract_to)
            
        elif file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            os.remove(file_path)
            return get_files_from_directory(extract_to)
            
        return [file_path]
    except Exception as e:
        print(f"Extraction failed: {e}")
        return []

def get_files_from_directory(directory):
    """Get all files from directory recursively."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def process_file(file_path, raw_dir, train_dir, test_dir):
    """Process a single file based on its type."""
    if is_idx_file(file_path):
        process_idx_file(file_path, train_dir, test_dir)
    elif is_image_file(file_path):
        process_image_file(file_path, train_dir, test_dir)
    elif is_binary_file(file_path):
        process_binary_file(file_path, train_dir, test_dir)

def is_idx_file(file_path):
    """Check if file is in IDX format."""
    try:
        with open(file_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), byteorder='big')
            return magic & 0xFFF0000 == 0x0000000
    except:
        return False

def is_image_file(file_path):
    """Check if file is an image."""
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def is_binary_file(file_path):
    """Check if file is a binary data file."""
    # Implement based on your specific binary format requirements
    return False

def process_idx_file(file_path, train_dir, test_dir):
    """Process IDX format file."""
    if 'train' in os.path.basename(file_path).lower():
        if 'image' in os.path.basename(file_path).lower():
            images = read_idx3_ubyte(file_path)
            labels = read_idx1_ubyte(file_path.replace('images', 'labels'))
            save_images_and_labels(images, labels, train_dir)
        elif 'label' not in os.path.basename(file_path).lower():
            # If not clearly marked, assume training data
            images = read_idx3_ubyte(file_path)
            labels = np.zeros(len(images))  # Default labels if not provided
            save_images_and_labels(images, labels, train_dir)
    elif 'test' in os.path.basename(file_path).lower() or 't10k' in os.path.basename(file_path).lower():
        if 'image' in os.path.basename(file_path).lower():
            images = read_idx3_ubyte(file_path)
            labels = read_idx1_ubyte(file_path.replace('images', 'labels'))
            save_images_and_labels(images, labels, test_dir)

def process_image_file(file_path, train_dir, test_dir):
    """Process image file."""
    # Determine if file belongs to train or test set based on path
    is_test = 'test' in file_path.lower()
    output_dir = test_dir if is_test else train_dir
    
    # Try to extract class from directory structure
    class_name = extract_class_from_path(file_path)
    if class_name is None:
        class_name = '0'  # Default class if none found
    
    # Save processed image
    save_single_image(file_path, output_dir, class_name)

def extract_class_from_path(file_path):
    """Extract class name from file path."""
    parts = Path(file_path).parts
    for part in reversed(parts[:-1]):  # Exclude filename
        if part.lower() not in ['train', 'test', 'raw', 'data']:
            return part
    return None

def save_single_image(src_path, output_dir, class_name):
    """Save a single image to the appropriate directory."""
    try:
        class_dir = os.path.join(output_dir, str(class_name))
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate unique filename
        dest_filename = f"{len(os.listdir(class_dir))}.png"
        dest_path = os.path.join(class_dir, dest_filename)
        
        # Process and save image
        with Image.open(src_path) as img:
            # Convert to RGB if needed
            if img.mode not in ['L', 'RGB']:
                img = img.convert('RGB')
            img.save(dest_path)
    except Exception as e:
        print(f"Error processing image {src_path}: {e}")

def read_idx3_ubyte(file_path):
    """Read IDX3-UBYTE format image data."""
    with open(file_path, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        size = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(size, rows, cols)

def read_idx1_ubyte(file_path):
    """Read IDX1-UBYTE format label data."""
    with open(file_path, "rb") as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        size = int.from_bytes(f.read(4), byteorder='big')
        return np.frombuffer(f.read(), dtype=np.uint8)

def save_images_and_labels(images, labels, output_dir):
    """Save images with their corresponding labels."""
    unique_labels = np.unique(labels)
    
    # Create directories for each class
    for label in unique_labels:
        os.makedirs(os.path.join(output_dir, str(label)), exist_ok=True)
    
    # Save images
    for idx, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(output_dir, str(label), f"{idx}.png")
        Image.fromarray(image).save(image_path)

def generate_dataset_info(dataset_dir):
    """Generate dataset info by analyzing the processed data."""
    train_dir = os.path.join(dataset_dir, "train")
    
    # Find a sample image
    sample_image = None
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                sample_image = Image.open(os.path.join(root, file))
                break
        if sample_image:
            break
    
    if sample_image:
        info = {
            "dataset": {
                "name": os.path.basename(dataset_dir),
                "type": "custom",
                "in_channels": 1 if sample_image.mode == 'L' else 3,
                "num_classes": len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]),
                "input_size": list(sample_image.size),
                "mean": [0.5] if sample_image.mode == 'L' else [0.5, 0.5, 0.5],
                "std": [0.5] if sample_image.mode == 'L' else [0.5, 0.5, 0.5],
                "train_dir": train_dir,
                "test_dir": os.path.join(dataset_dir, "test"),
                "image_type": sample_image.mode.lower()
            }
        }
        
        with open(os.path.join(dataset_dir, "dataset_info.json"), 'w') as f:
            json.dump(info, f, indent=4)
        
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
