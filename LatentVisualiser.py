import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog
from typing import Dict, List, Tuple, Optional, Union, Any

class LatentSpaceVisualizer:
    def __init__(self, latent_space_dir: str, config_file: str):
        """
        Initialize the latent space visualizer.

        Args:
            latent_space_dir: Directory containing latent space CSV files.
            config_file: Path to the dataset's configuration file.
        """
        self.latent_space_dir = latent_space_dir
        self.config_file = config_file
        self.dataset_name = os.path.splitext(os.path.basename(config_file))[0]

        # Load configurations
        self.dataset_config = self._load_dataset_config()
        self.classes = self._get_classes_from_latent_space()  # Extract classes from subfolders
        self.latent_dim = self.dataset_config.get('latent_dim', 128)  # Latent space dimensionality

        # Create visualization directory
        self.base_viz_path = os.path.join('visualizations', self.dataset_name, 'latent_space')
        os.makedirs(self.base_viz_path, exist_ok=True)

    def _load_dataset_config(self) -> dict:
        """Load dataset-specific configuration."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading dataset config: {str(e)}")
            return {}

    def _get_classes_from_latent_space(self) -> List[str]:
        """
        Extract class names from the subfolders in the latent space directory.

        Returns:
            List[str]: List of class names.
        """
        if not os.path.exists(self.latent_space_dir):
            raise ValueError(f"Latent space directory not found: {self.latent_space_dir}")

        # Get all subfolders (each subfolder represents a class)
        classes = [d for d in os.listdir(self.latent_space_dir) if os.path.isdir(os.path.join(self.latent_space_dir, d))]
        return classes

    def _load_latent_space_data(self) -> Dict[str, np.ndarray]:
        """
        Load latent space data for all classes.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are class names and values are latent space arrays.
        """
        latent_data = {}
        for class_name in self.classes:
            class_dir = os.path.join(self.latent_space_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: No latent space data found for class {class_name}")
                continue

            # Load all CSV files in the class directory
            latent_arrays = []
            for file in os.listdir(class_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(class_dir, file)
                    latent_values = np.loadtxt(file_path, delimiter=',')
                    latent_arrays.append(latent_values)

            if latent_arrays:
                latent_data[class_name] = np.vstack(latent_arrays)

        return latent_data

    def visualize_latent_space(self, method: str = 'tsne', n_components: int = 3):
        """
        Visualize the latent space using t-SNE or PCA.

        Args:
            method: Dimensionality reduction method ('tsne' or 'pca').
            n_components: Number of components for dimensionality reduction (2 or 3).
        """
        # Load latent space data
        latent_data = self._load_latent_space_data()
        if not latent_data:
            print("No latent space data found.")
            return

        # Combine all latent data into a single array
        all_latent = np.vstack(list(latent_data.values()))
        class_labels = np.concatenate([[class_name] * len(data) for class_name, data in latent_data.items()])

        # Perform dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError("Invalid method. Use 'tsne' or 'pca'.")

        reduced_data = reducer.fit_transform(all_latent)

        # Create a DataFrame for visualization
        df = pd.DataFrame(reduced_data, columns=[f'Dim{i+1}' for i in range(n_components)])
        df['Class'] = class_labels

        # Create 3D or 2D scatter plot
        if n_components == 3:
            fig = px.scatter_3d(df, x='Dim1', y='Dim2', z='Dim3', color='Class',
                                title=f'Latent Space Visualization ({method.upper()})',
                                opacity=0.7, hover_name='Class')
        else:
            fig = px.scatter(df, x='Dim1', y='Dim2', color='Class',
                             title=f'Latent Space Visualization ({method.upper()})',
                             opacity=0.7, hover_name='Class')

        # Show the plot
        fig.show()

def select_latent_space_dir(default_dir: str) -> str:
    """
    Open a file dialog to select the latent space directory.
    If no directory is selected, use the default directory.

    Args:
        default_dir: Default directory to use if no selection is made.

    Returns:
        str: Selected or default directory.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    latent_space_dir = filedialog.askdirectory(title="Select Latent Space Directory", initialdir=default_dir)
    return latent_space_dir if latent_space_dir else default_dir

def main():
    # Ask the user for the name of the latent space data folder
    latentfolder = input("Please enter the name of the latent space data folder (e.g., galaxies): ")
    data_folder = f"data/{latentfolder}"

    # Automatically detect the JSON configuration file in the data folder
    config_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    if not config_files:
        print(f"No JSON configuration file found in the data folder: {data_folder}")
        return

    # Use the first JSON file found
    config_file = os.path.join(data_folder, config_files[0])
    print(f"Using configuration file: {config_file}")

    # Default latent space directory
    default_latent_space_dir = os.path.join(data_folder, "latent_space", "train")

    # Select latent space directory
    latent_space_dir = select_latent_space_dir(default_latent_space_dir)
    print(f"Using latent space directory: {latent_space_dir}")

    # Initialize the visualizer
    visualizer = LatentSpaceVisualizer(latent_space_dir, config_file)

    # Interactive menu for visualization
    while True:
        print("\n--- Latent Space Visualization ---")
        print("1. Visualize with t-SNE (2D)")
        print("2. Visualize with t-SNE (3D)")
        print("3. Visualize with PCA (2D)")
        print("4. Visualize with PCA (3D)")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            visualizer.visualize_latent_space(method='tsne', n_components=2)
        elif choice == '2':
            visualizer.visualize_latent_space(method='tsne', n_components=3)
        elif choice == '3':
            visualizer.visualize_latent_space(method='pca', n_components=2)
        elif choice == '4':
            visualizer.visualize_latent_space(method='pca', n_components=3)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
