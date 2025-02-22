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
import numpy as np
import torch
from typing import List, Union
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
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image
from colorama import init, Fore, Back, Style
import os
from tqdm.auto import tqdm
import torch
from torchvision import transforms
from PIL import Image
from autoencoder.model import ModifiedAutoencoder
from colorama import init, Fore, Back, Style
from skimage.metrics import structural_similarity as ssim
import numpy as np


def ssim_loss(original, reconstructed):
    """
    Compute the SSIM loss between original and reconstructed images.
    """
    # Detach tensors from the computation graph and convert to numpy arrays
    original_np = original.detach().cpu().numpy()
    reconstructed_np = reconstructed.detach().cpu().numpy()

    # Compute SSIM for each image in the batch
    ssim_values = []
    for i in range(original_np.shape[0]):
        ssim_value = ssim(
            original_np[i], reconstructed_np[i],
            win_size=7,  # Adjust window size as needed
            channel_axis=0 if original_np.shape[1] == 1 else 1,  # Handle grayscale and RGB
            data_range=1.0  # Assuming images are normalized to [0, 1]
        )
        ssim_values.append(ssim_value)

    # Convert SSIM values to a tensor
    ssim_tensor = torch.tensor(ssim_values, device=original.device)

    # SSIM loss: 1 - SSIM (since SSIM ranges from -1 to 1, with 1 being perfect similarity)
    return 1 - ssim_tensor.mean()

def verify_latent_saving(dataset_name, class_folders):
    """Verify that latent space CSV files are properly saved."""
    base_dir = f"data/{dataset_name}/latent_space/train"
    saved_files = {cls: [] for cls in class_folders}

    for class_name in class_folders:
        class_path = os.path.join(base_dir, class_name)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
            saved_files[class_name] = files

    print("\nLatent Space Saving Summary:")
    for class_name, files in saved_files.items():
        print(f"Class {class_name}: {len(files)} CSV files saved")

    return saved_files

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image
from colorama import init, Fore, Back, Style

def update_confusion_matrix(original, reconstructed, labels, confusion_matrix, screen_group=0):
    """
    Update the confusion matrix based on the similarity between original and reconstructed images.
    Compute similarity metrics (MSE, PSNR, SSIM) for each class and display them in a grid starting at row 15 and column (screen_width // 2 - 20).

    Args:
        original: Original images tensor (B, C, H, W)
        reconstructed: Reconstructed images tensor (B, C, H, W)
        labels: True class labels tensor (B)
        confusion_matrix: Running confusion matrix tensor (num_classes, num_classes) - Not used in this version
        screen_group: The current group of classes to display (0 for the first group, 1 for the next, etc.)

    Returns:
        Dictionary containing similarity metrics for each class
    """
    with torch.no_grad():
        # Ensure all tensors are on the same device
        device = original.device
        original = original.to(device)
        reconstructed = reconstructed.to(device)
        labels = labels.to(device)

        # Initialize dictionaries to store metrics per class
        class_metrics = {
            "MSE": {},
            "PSNR": {},
            "SSIM": {}
        }

        # Get unique classes in the batch
        unique_classes = torch.unique(labels)

        # Compute metrics for each class
        for class_id in unique_classes:
            # Mask for the current class
            class_mask = (labels == class_id)
            class_original = original[class_mask]
            class_reconstructed = reconstructed[class_mask]

            # Skip if no images belong to this class in the batch
            if class_original.shape[0] == 0:
                continue

            # Compute Mean Squared Error (MSE) for the class
            mse = torch.mean((class_original - class_reconstructed) ** 2, dim=(1, 2, 3)).mean()

            # Compute Peak Signal-to-Noise Ratio (PSNR) for the class
            max_pixel = 1.0  # Assuming images are normalized to [0, 1]
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()

            # Compute Structural Similarity Index (SSIM) for the class
            from skimage.metrics import structural_similarity as ssim
            original_np = class_original.cpu().numpy()
            reconstructed_np = class_reconstructed.cpu().numpy()

            # Determine the window size for SSIM
            min_side = min(original_np.shape[2], original_np.shape[3])  # Minimum of height and width
            win_size = min(7, min_side)  # Ensure win_size is <= smallest side of the image
            if win_size % 2 == 0:  # Ensure win_size is odd
                win_size -= 1

            # Skip SSIM computation if the images are too small or have no variation
            if win_size < 3 or torch.allclose(class_original, class_reconstructed, atol=1e-5):
                ssim_value = float('nan')  # Use NaN if SSIM cannot be computed
            else:
                # Compute SSIM for the class
                try:
                    ssim_value = ssim(
                        original_np, reconstructed_np,
                        win_size=win_size,  # Use adjusted window size
                        channel_axis=1,  # Set channel_axis for multichannel images (C is at axis 1 in PyTorch tensors)
                        data_range=max_pixel
                    )
                except ValueError:
                    # Fallback to NaN if SSIM computation fails
                    ssim_value = float('nan')

            # Store metrics for the class
            class_metrics["MSE"][class_id.item()] = mse.item()
            class_metrics["PSNR"][class_id.item()] = psnr
            class_metrics["SSIM"][class_id.item()] = ssim_value

        # Determine screen dimensions
        screen_height = os.get_terminal_size().lines
        screen_width = os.get_terminal_size().columns

        # Calculate the starting column (middle of the screen minus 20 columns)
        start_col = max(0, (screen_width // 2) - 20)  # Ensure start_col is not negative
        end_col = screen_width  # Use the full screen width for display

        # Determine how many classes can fit per row
        class_block_width = 20  # Width of each class block (MSE, PSNR, SSIM)
        num_columns = (end_col - start_col) // class_block_width  # Number of columns per row
        num_columns = max(1, num_columns)  # Ensure at least 1 column

        # Determine how many rows can fit on the screen
        rows_per_screen = screen_height - 15  # Start from row 15
        classes_per_screen = rows_per_screen * num_columns  # Total classes per screen

        # Split the classes into groups
        unique_classes = sorted(unique_classes.tolist())
        class_groups = [unique_classes[i:i + classes_per_screen] for i in range(0, len(unique_classes), classes_per_screen)]

        # Display the current group of classes
        current_group = screen_group % len(class_groups)
        print(f"\033[15;{start_col}H\033[K")  # Move to row 15, start_col and clear the line
        print(f"\033[15;{start_col}HPer-Class Similarity Metrics (Group {current_group + 1}/{len(class_groups)}):")
        row = 16  # Start printing metrics from row 16
        col = start_col

        for class_id in class_groups[current_group]:
            # Print class ID and metrics
            print(f"\033[{row};{col}HClass {class_id}:")
            print(f"\033[{row + 1};{col}H  MSE: {class_metrics['MSE'].get(class_id, 'N/A'):.6f}")
            print(f"\033[{row + 2};{col}H  PSNR: {class_metrics['PSNR'].get(class_id, 'N/A'):.2f} dB")
            print(f"\033[{row + 3};{col}H  SSIM: {class_metrics['SSIM'].get(class_id, 'N/A'):.4f}")

            # Move to the next column
            col += class_block_width
            if col + class_block_width > end_col:  # Move to the next row if the current row is full
                col = start_col
                row += 4

        return class_metrics, len(class_groups)





def display_confusion_matrix(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    true_labels: torch.Tensor,
    confusion_matrix: torch.Tensor,
    threshold: float = 0.1
) -> None:
    """
    Display a scaled confusion matrix with color-coded cells.

    Args:
        original: Original images tensor (B, C, H, W)
        reconstructed: Reconstructed images tensor (B, C, H, W)
        true_labels: True class labels tensor (B)
        confusion_matrix: Running confusion matrix tensor (num_classes, num_classes)
        threshold: MSE threshold for considering reconstruction successful
    """
    with torch.no_grad():
        # Ensure all tensors are on the same device
        device = original.device
        true_labels = true_labels.to(device)
        confusion_matrix = confusion_matrix.to(device)

        # Compute MSE for batch
        mse = torch.mean((original - reconstructed)**2, dim=(1,2,3))

        # Update confusion matrix based on reconstruction quality
        pred_labels = torch.where(mse < threshold, true_labels,
                                torch.tensor(-1, device=device))

        for t, p in zip(true_labels, pred_labels):
            if p != -1:
                confusion_matrix[t][p] += 1
            else:
                confusion_matrix[t][t] -= 1

        # Calculate metrics
        total = confusion_matrix.sum(axis=1)
        correct = confusion_matrix.diag()
        accuracy = torch.where(total > 0, correct.float() / total.float(),
                             torch.zeros_like(total, dtype=torch.float32))
        overall_acc = correct.sum().float() / total.sum().float() if total.sum() > 0 else 0

        # Get terminal size for scaling
        term_width = os.get_terminal_size().columns
        num_classes = confusion_matrix.shape[0]

        # Scale cell size based on number of classes
        cell_width = max(3, min(8, term_width // (num_classes + 5)))

        # Create compact display string
        display_str = f"\nOverall Accuracy: {overall_acc:.1%}\n"

        # Add matrix with color-coded cells
        max_val = confusion_matrix.max()

        # Header
        display_str += "   "
        for j in range(num_classes):
            display_str += f"{j:^{cell_width}}"
        display_str += "\n" + "-" * (3 + cell_width * num_classes) + "\n"

        # Matrix content with color coding
        for i in range(num_classes):
            display_str += f"{i:2} "
            for j in range(num_classes):
                val = confusion_matrix[i, j].item()
                if val == 0:
                    color = Style.RESET_ALL
                else:
                    # Color intensity based on value
                    intensity = min(1.0, val / max_val.item())
                    if i == j:  # Diagonal - use green
                        color = f"\033[38;2;{int(intensity*150)};{int(intensity*255)};{int(intensity*150)}m"
                    else:  # Off-diagonal - use red
                        color = f"\033[38;2;{int(intensity*255)};{int(intensity*150)};{int(intensity*150)}m"

                # Use dots for very narrow cells
                if cell_width <= 3:
                    display_str += f"{color}●{Style.RESET_ALL}"
                else:
                    display_str += f"{color}{val:^{cell_width}.0f}{Style.RESET_ALL}"
            display_str += "\n"

        # Class-wise accuracies in compact form
        display_str += "\nClass Acc: "
        acc_str = " ".join([f"{i}:{accuracy[i]:.0%}" for i in range(num_classes)])

        # Print with proper positioning
        print(f"\033[K{display_str}")



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
def display_header():
    """Display header with branding and class distribution."""
    print("\033[2J\033[H")  # Clear screen

    # Header and branding
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.YELLOW}{'Visual Autoencoder Tool':^80}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")

    # Author and License information
    print(f"{Style.BRIGHT}{Fore.WHITE}{'Author: ':>30}{Fore.YELLOW}Ninan Sajeeth Philip{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'Organisation: ':>30}{Fore.LIGHTGREEN_EX}Artificial Intelligence Research and Intelligent Systems{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{'':>30}Thelliyoor -689544 India{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'License: ':>30}{Fore.BLUE}Creative Commons License{Style.RESET_ALL}\n")
    print(f"{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}")

    # Add class distribution if available
    try:
        dataset_name = os.path.basename(os.getcwd().split('data/')[-1])
        train_dir = f"data/{dataset_name}/train"
        if os.path.exists(train_dir):
            print(f"\n{Style.BRIGHT}{Fore.CYAN}Class Distribution:{Style.RESET_ALL}")
            class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            for class_name in class_dirs:
                class_path = os.path.join(train_dir, class_name)
                num_samples = len([f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{Style.BRIGHT}{Fore.WHITE}{class_name:>30}: {Fore.YELLOW}{num_samples} samples{Style.RESET_ALL}")
    except Exception:
        pass  # Skip class distribution if not available

    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")



def display_confusion_matrix(
    confusion_matrix: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
    terminal_height: int,
    header_height: int
) -> None:
    """
    Display the confusion matrix with proper handling of both numpy arrays and torch tensors.

    Args:
        confusion_matrix: Either a numpy array or torch tensor containing the confusion matrix
        class_names: List of class names
        terminal_height: Height of the terminal
        header_height: Height reserved for header
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()

    # Calculate metrics
    correct = np.diagonal(confusion_matrix)
    total = confusion_matrix.sum(axis=1)
    accuracy = correct / total

    # Create the display string
    display_str = "\033[H"  # Move to top of screen
    display_str += "\n" * header_height  # Move past header

    # Add class-wise accuracy
    display_str += "Class-wise Accuracy:\n"
    for i, (class_name, acc) in enumerate(zip(class_names, accuracy)):
        display_str += f"{class_name}: {acc:.2%}\n"

    # Add overall accuracy
    total_correct = correct.sum()
    total_samples = total.sum()
    overall_accuracy = total_correct / total_samples
    display_str += f"\nOverall Accuracy: {overall_accuracy:.2%}\n"

    # Add confusion matrix
    display_str += "\nConfusion Matrix:\n"
    display_str += "True\\Pred |"
    for name in class_names:
        display_str += f" {name:>8} |"
    display_str += "\n" + "-" * (10 + 11 * len(class_names)) + "\n"

    for i, true_class in enumerate(class_names):
        display_str += f"{true_class:9} |"
        for j in range(len(class_names)):
            display_str += f" {confusion_matrix[i, j]:8.0f} |"
        display_str += "\n"

    print(display_str)

def update_progress(
    epoch: int,
    batch: int,
    total_batches: int,
    loss: float,
    terminal_height: int
) -> None:
    """
    Update the progress display with current training metrics.
    """
    progress = (batch / total_batches) * 100
    display_str = f"\033[{terminal_height-3}H"  # Move to bottom of screen
    display_str += f"Epoch: {epoch} | Batch: {batch}/{total_batches} | "
    display_str += f"Progress: {progress:.1f}% | Loss: {loss:.4f}"
    print(display_str)

def update_progress_bar(epoch, batch, total_batches, loss, terminal_height):
    """
    Update progress bar with static positioning.

    Args:
        epoch: Current epoch number
        batch: Current batch number
        total_batches: Total number of batches
        loss: Current loss value
        terminal_height: Terminal height for positioning
    """
    bar_width = 50
    progress = batch / total_batches
    filled = int(bar_width * progress)
    bar = "█" * filled + "-" * (bar_width - filled)

    # Position cursor and clear line
    print(f"\033[{terminal_height-2};0H\033[K", end="")
    print(f"Epoch {epoch}: [{bar}] {batch}/{total_batches} Loss: {loss:.4f}")

def create_confusion_matrix(num_classes):
    """Initialize a proper confusion matrix."""
    return torch.zeros((num_classes, num_classes), dtype=torch.long)



def save_batch_latents(batch_latents, image_paths, dataset_name, metadata=None):
    """
    Save latent representations with original filenames and metadata.

    Args:
        batch_latents: Tensor of latent representations
        image_paths: List of paths to original images
        dataset_name: Name of the dataset
        metadata: Optional dictionary containing additional metadata
    """
    base_latent_dir = os.path.join(f"data/{dataset_name}", "latent_space")

    for latent, path in zip(batch_latents, image_paths):
        # Maintain original filename and structure
        rel_path = os.path.relpath(path, f"data/{dataset_name}/train")
        target_dir = os.path.join(base_latent_dir, os.path.dirname(rel_path))
        os.makedirs(target_dir, exist_ok=True)

        # Use original filename for CSV
        filename = os.path.splitext(os.path.basename(path))[0]
        csv_path = os.path.join(target_dir, f"{filename}.csv")

        # Prepare data for CSV
        data = {
            'type': ['metadata'] * (len(metadata) if metadata else 0) + ['latent_values'],
            'key': (list(metadata.keys()) if metadata else []) + ['values'],
            'value': (list(map(str, metadata.values())) if metadata else []) +
                    [','.join(map(str, latent.detach().cpu().numpy().flatten()))]
        }

        # Save as DataFrame
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)





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
    """Save model checkpoint properly."""
    model.eval()
    model = model.cpu()

    # Create state dict with all necessary components
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "config": config,
        "frequencies": model.latent_mapper.frequencies
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, config, device):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: The model to load the checkpoint into.
        config: Configuration dictionary.
        device: The device (CPU, GPU, or TPU) to load the checkpoint onto.

    Returns:
        model: The model with loaded weights.
        epoch: The epoch number from the checkpoint.
        best_loss: The best loss value from the checkpoint.
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)  # Load checkpoint onto the correct device
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)  # Ensure model is on the correct device
        epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        return model, epoch, best_loss
    else:
        print("No checkpoint found. Starting from scratch.")
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
                "loss_functions": {
                    "mse": {
                        "enabled": True,
                        "weight": 1.0
                    },
                    "ssim": {
                        "enabled": True,
                        "weight": 0.5
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

def save_latent_space_for_epoch(model, data_loader, device, dataset_name):
    """
    Save latent space representations for all images in the dataset at the end of an epoch.

    Args:
        model: The trained autoencoder model.
        data_loader: DataLoader for the dataset.
        device: The device (CPU/GPU) where computations are performed.
        dataset_name: Name of the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    latent_space = []
    image_paths = []

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)

            # Forward pass: Get latent space representations
            _, latent = model(images)

            # Store latent space representations and image paths
            latent_space.append(latent.cpu())

            # Extract image paths correctly
            if hasattr(data_loader.dataset, 'samples'):
                # If samples is a list of tuples (image_path, label), extract only the paths
                if isinstance(data_loader.dataset.samples[0], tuple):
                    image_paths.extend([sample[0] for sample in data_loader.dataset.samples])
                else:
                    # If samples is a list of strings (image paths), use it directly
                    image_paths.extend(data_loader.dataset.samples)
            else:
                raise ValueError("DataLoader dataset does not have 'samples' attribute.")

    # Concatenate all batches
    latent_space = torch.cat(latent_space, dim=0)

    # Save latent space representations
    save_latent_space(latent_space, image_paths, dataset_name)

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

def save_latent_space(latent_space, image_paths, dataset_name):
    """
    Save latent space representations as CSV files, retaining the exact basename of the input files.

    Args:
        latent_space: Tensor containing latent space representations.
        image_paths: List of paths to the original images.
        dataset_name: Name of the dataset.
    """
    base_latent_dir = os.path.join("data", dataset_name, "latent_space")
    os.makedirs(base_latent_dir, exist_ok=True)

    for latent, image_path in zip(latent_space, image_paths):
        # Get the relative path from the dataset directory to maintain hierarchy
        rel_path = os.path.relpath(os.path.dirname(image_path), f"data/{dataset_name}/train")
        target_dir = os.path.join(base_latent_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save latent space representation as CSV
        csv_path = os.path.join(target_dir, f"{filename}.csv")
        latent_values = latent.numpy().flatten()
        np.savetxt(csv_path, latent_values, delimiter=",")

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
