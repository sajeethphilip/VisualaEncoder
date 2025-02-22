import os
import json
import torch
from torchvision import datasets, transforms
from PIL import Image
from autoencoder.train import train_model
from autoencoder.utils import download_and_extract, setup_dataset,extract_and_organize,find_first_image,reconstruct_image,reconstruct_from_latent,display_header

def create_default_json_config(dataset_name, data_dir, image_path, latent_dim=128, embedding_dim=64, learning_rate=0.001, batch_size=32, epochs=20):
    """
    Create a default JSON configuration file interactively.

    Args:
        dataset_name: Name of the dataset.
        data_dir: Directory containing the dataset.
        image_path: Path to the first image in the dataset.
        latent_dim: Latent space dimension (default: 128).
        embedding_dim: Embedding dimension (default: 64).
        learning_rate: Learning rate (default: 0.001).
        batch_size: Batch size (default: 32).
        epochs: Number of epochs (default: 20).

    Returns:
        JSON configuration dictionary.
    """
    # Read the first image to determine its shape
    image = Image.open(image_path)
    width, height = image.size
    in_channels = 1 if image.mode == "L" else 3  # Grayscale or RGB

    # Count the number of classes
    classes = len(os.listdir(f"{data_dir}/train"))

    # Create JSON configuration
    config = {
        "dataset": {
            "name": dataset_name,
            "type": "custom",
            "in_channels": in_channels,
            "num_classes": classes,
            "input_size": [height, width],
            "mean": [0.5] * in_channels,
            "std": [0.5] * in_channels,
            "train_dir": os.path.join(data_dir, "train"),
            "test_dir": os.path.join(data_dir, "test"),
            "image_type": "grayscale" if in_channels == 1 else "rgb"
        },
            "multiscale": {
            "enabled": False,
            "method": "wavelet",
            "levels": 3,
            "normalize_per_scale": True,
            "resize_to_input": True
            },

        "model": {
            "encoder_type": "autoenc",
            "feature_dims": latent_dim,
            "learning_rate": learning_rate,
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
            "batch_size": batch_size,
            "epochs": epochs,
            "num_workers": 4,
            "checkpoint_dir": os.path.join(data_dir, "checkpoints"),
            "validation_split": 0.2,
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.001
            },
            "invert_DBNN": True,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.1,
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
                "mean": [0.5] * in_channels,
                "std": [0.5] * in_channels
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

    # Save JSON file
    json_path = os.path.join(data_dir, f"{dataset_name}.json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Created default JSON configuration file at {json_path}")
    return config

def validate_config(config):
    """Validate the configuration dictionary."""
    required_keys = ["dataset", "model", "training", "augmentation", "logging", "output", "multiscale"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    # Validate model-specific keys
    model_keys = ["encoder_type", "feature_dims", "learning_rate", "optimizer", "scheduler", "autoencoder_config"]
    for key in model_keys:
        if key not in config["model"]:
            raise ValueError(f"Missing required key in config['model']: {key}")

    # Validate training-specific keys
    training_keys = ["batch_size", "epochs", "num_workers", "checkpoint_dir", "validation_split", "early_stopping"]
    for key in training_keys:
        if key not in config["training"]:
            raise ValueError(f"Missing required key in config['training']: {key}")

    # Validate multiscale-specific keys
    multiscale_keys = ["enabled", "method", "levels", "normalize_per_scale", "resize_to_input"]
    for key in multiscale_keys:
        if key not in config["multiscale"]:
            raise ValueError(f"Missing required key in config['multiscale']: {key}")

    print("Configuration is valid.")

def check_and_fix_json(json_path, dataset_name, data_dir, image_path):
    """
    Check if JSON file exists and is valid; only add missing features or fix incorrect values.

    Args:
        json_path: Path to the JSON configuration file.
        dataset_name: Name of the dataset.
        data_dir: Directory containing the dataset.
        image_path: Path to the first image in the dataset.

    Returns:
        Updated JSON configuration.
    """
    # Define default multiscale configuration
    default_multiscale_config = {
        "enabled": False,
        "method": "wavelet",
        "levels": 3,
        "normalize_per_scale": True,
        "resize_to_input": True
    }

    # Load default configuration for comparison
    default_config = create_default_json_config(dataset_name, data_dir, image_path)

    try:
        # Try to load existing config
        with open(json_path, "r") as f:
            existing_config = json.load(f)

        # Function to recursively update missing or incorrect keys
        def update_config(existing, default):
            changes_made = False
            for key, default_value in default.items():
                if key not in existing:
                    # Add missing key
                    existing[key] = default_value
                    print(f"Added missing key '{key}' to configuration.")
                    changes_made = True
                elif isinstance(default_value, dict) and isinstance(existing[key], dict):
                    # Recursively update nested dictionaries
                    if update_config(existing[key], default_value):
                        changes_made = True
                elif not isinstance(existing[key], type(default_value)):
                    # Fix incorrect value type only if the existing value is of the wrong type
                    print(f"Fixing incorrect value type for key '{key}' in configuration.")
                    existing[key] = default_value
                    changes_made = True
            return changes_made

        # Update the existing configuration
        changes_made = update_config(existing_config, default_config)

        # Save the updated configuration only if changes were made
        if changes_made:
            print(f"Updating configuration file at {json_path}")
            with open(json_path, "w") as f:
                json.dump(existing_config, f, indent=4)
        else:
            print("Configuration is valid. No changes were made.")

        return existing_config

    except (FileNotFoundError, json.JSONDecodeError) as e:
        # If the file is missing or corrupted, create a new one
        print(f"Config file is missing or corrupted: {str(e)}")
        print(f"Creating new configuration file at {json_path}")
        return create_default_json_config(dataset_name, data_dir, image_path)


def main():
    """Main function for user interaction."""
    # Clear screen and initialize colorama
    os.system('cls' if os.name == 'nt' else 'clear')
    from colorama import init, Fore, Back, Style
    init()

    # Header and branding
    display_header()

    # Mode selection first
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'Select mode:':^80}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'1. ':>35}{Fore.YELLOW}Train{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'2. ':>35}{Fore.YELLOW}Predict{Style.RESET_ALL}")

    mode = input(f"{Style.BRIGHT}{Fore.WHITE}Enter your choice (1/2): {Style.RESET_ALL}")

    try:
        if mode == "2":  # Predict mode
            dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter dataset name: {Style.RESET_ALL}")
            checkpoint_path = os.path.join(f"data/{dataset_name}/checkpoints", "best_model.pth")

            if not os.path.exists(checkpoint_path):
                print(f"{Style.BRIGHT}{Fore.RED}No trained model found. Please train the model first.{Style.RESET_ALL}")
                return

            config_path = os.path.join(f"data/{dataset_name}", f"{dataset_name}.json")
            config = check_and_fix_json(config_path, dataset_name, f"data/{dataset_name}", find_first_image(f"data/{dataset_name}/train"))

            print(f"\n{Style.BRIGHT}{Fore.CYAN}Reconstructing image(s)...{Style.RESET_ALL}")

            if config["training"].get("invert_DBNN", True):
                default_csv_path = os.path.join(f"data/{dataset_name}/latent_space")
                print(f"\nFound latent representations in: {default_csv_path}")
                csv_path = input(f"Enter the path to the latent CSV file/folder (default: {default_csv_path}): ") or default_csv_path
                reconstruct_from_latent(csv_path, checkpoint_path, dataset_name, config)
            else:
                input_path = input(f"{Style.BRIGHT}{Fore.WHITE}Enter path to image or folder to reconstruct: {Style.RESET_ALL}")
                reconstruct_image(input_path, checkpoint_path, dataset_name, config)

        elif mode == "1":  # Train mode
            # Data source selection menu
            print(f"\n{Style.BRIGHT}{Fore.CYAN}{'Select data source:':^80}{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.WHITE}{'1. ':>35}{Fore.YELLOW}Torchvision dataset (e.g., CIFAR10, MNIST){Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.WHITE}{'2. ':>35}{Fore.YELLOW}URL to download dataset{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.WHITE}{'3. ':>35}{Fore.YELLOW}Local file/folder{Style.RESET_ALL}\n")

            data_source = input(f"{Style.BRIGHT}{Fore.WHITE}Enter your choice (1/2/3): {Style.RESET_ALL}")

            if data_source == "1":
                dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter dataset name (e.g., CIFAR10, MNIST): {Style.RESET_ALL}").upper()
                config = setup_dataset(dataset_name)
            elif data_source == "2":
                url = input(f"{Style.BRIGHT}{Fore.WHITE}Enter the URL to download the dataset: {Style.RESET_ALL}")
                dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter a name for the dataset: {Style.RESET_ALL}")
                data_dir = extract_and_organize(url, dataset_name, is_url=True)
                first_image = find_first_image(data_dir)

                # Collect user inputs for training parameters
                latent_dim = int(input("Enter the latent space dimension (default: 128): ") or 128)
                embedding_dim = int(input("Enter the embedding dimension (default: 64): ") or 64)
                learning_rate = float(input("Enter the learning rate (default: 0.001): ") or 0.001)
                batch_size = int(input("Enter the batch size (default: 32): ") or 32)
                epochs = int(input("Enter the number of epochs (default: 20): ") or 20)

                # Create or update JSON configuration
                json_path = os.path.join(data_dir, f"{dataset_name}.json")
                if os.path.exists(json_path):
                    # If JSON file exists, validate and update it
                    config = check_and_fix_json(json_path, dataset_name, data_dir, first_image)
                else:
                    # If JSON file does not exist, create a new one with user-provided values
                    config = create_default_json_config(dataset_name, data_dir, first_image, latent_dim, embedding_dim, learning_rate, batch_size, epochs)
            elif data_source == "3":
                source_path = input(f"{Style.BRIGHT}{Fore.WHITE}Enter the path to the local file/folder: {Style.RESET_ALL}")
                dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter a name for the dataset: {Style.RESET_ALL}")
                if os.path.isfile(source_path) and source_path.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
                    data_dir = extract_and_organize(source_path, dataset_name)
                else:
                    data_dir = source_path
                first_image = find_first_image(data_dir)

                # Collect user inputs for training parameters
                latent_dim = int(input("Enter the latent space dimension (default: 128): ") or 128)
                embedding_dim = int(input("Enter the embedding dimension (default: 64): ") or 64)
                learning_rate = float(input("Enter the learning rate (default: 0.001): ") or 0.001)
                batch_size = int(input("Enter the batch size (default: 32): ") or 32)
                epochs = int(input("Enter the number of epochs (default: 20): ") or 20)

                # Create or update JSON configuration
                json_path = os.path.join(data_dir, f"{dataset_name}.json")
                if os.path.exists(json_path):
                    # If JSON file exists, validate and update it
                    config = check_and_fix_json(json_path, dataset_name, data_dir, first_image)
                else:
                    # If JSON file does not exist, create a new one with user-provided values
                    config = create_default_json_config(dataset_name, data_dir, first_image, latent_dim, embedding_dim, learning_rate, batch_size, epochs)
            else:
                print(f"{Style.BRIGHT}{Fore.RED}Invalid choice. Exiting...{Style.RESET_ALL}")
                return

            print(f"\n{Style.BRIGHT}{Fore.CYAN}Training the model...{Style.RESET_ALL}")
            train_model(config)

        else:
            print(f"{Style.BRIGHT}{Fore.RED}Invalid mode selected. Exiting...{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
