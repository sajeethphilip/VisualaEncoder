import os
import json
import torch
from torchvision import datasets, transforms
from PIL import Image
from autoencoder.train import train_model
from autoencoder.reconstruct import reconstruct_image
from autoencoder.data_loader import load_local_dataset, load_dataset_config
from autoencoder.utils import download_and_extract, setup_dataset,extract_and_organize,find_first_image

def create_default_json_config(dataset_name, data_dir, image_path):
    """Create a default JSON configuration file interactively."""
    # Read the first image to determine its shape
    image = Image.open(image_path)
    width, height = image.size
    in_channels = 1 if image.mode == "L" else 3  # Grayscale or RGB

    # Interactive configuration
    print("\nConfiguring the autoencoder...")
    latent_dim = int(input("Enter the latent space dimension (default: 128): ") or 128)
    embedding_dim = int(input("Enter the embedding dimension (default: 64): ") or 64)
    learning_rate = float(input("Enter the learning rate (default: 0.001): ") or 0.001)
    batch_size = int(input("Enter the batch size (default: 32): ") or 32)
    epochs = int(input("Enter the number of epochs (default: 20): ") or 20)

    # Create JSON configuration
    config = {
        "dataset": {
            "name": dataset_name,
            "type": "custom",
            "in_channels": in_channels,
            "num_classes": 0,  # Update if class labels are available
            "input_size": [height, width],  # Height x Width
            "mean": [0.5] * in_channels,  # Default mean
            "std": [0.5] * in_channels,  # Default std
            "train_dir": os.path.join(data_dir, "train"),
            "test_dir": os.path.join(data_dir, "test"),
            "image_type": "grayscale" if in_channels == 1 else "rgb"
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
            },
            "loss_functions": {
                "astronomical_structure": {
                    "enabled": True,
                    "weight": 1.0,
                    "components": {
                        "edge_preservation": True,
                        "peak_preservation": True,
                        "detail_preservation": True
                    }
                },
                "medical_structure": {
                    "enabled": True,
                    "weight": 1.0,
                    "components": {
                        "boundary_preservation": True,
                        "tissue_contrast": True,
                        "local_structure": True
                    }
                },
                "agricultural_pattern": {
                    "enabled": True,
                    "weight": 1.0,
                    "components": {
                        "texture_preservation": True,
                        "damage_pattern": True,
                        "color_consistency": True
                    }
                }
            },
            "enhancement_modules": {
                "astronomical": {
                    "enabled": True,
                    "components": {
                        "structure_preservation": True,
                        "detail_preservation": True,
                        "star_detection": True,
                        "galaxy_features": True,
                        "kl_divergence": True
                    },
                    "weights": {
                        "detail_weight": 1.0,
                        "structure_weight": 0.8,
                        "edge_weight": 0.7
                    }
                },
                "medical": {
                    "enabled": True,
                    "components": {
                        "tissue_boundary": True,
                        "lesion_detection": True,
                        "contrast_enhancement": True,
                        "subtle_feature_preservation": True
                    },
                    "weights": {
                        "boundary_weight": 1.0,
                        "lesion_weight": 0.8,
                        "contrast_weight": 0.6
                    }
                },
                "agricultural": {
                    "enabled": True,
                    "components": {
                        "texture_analysis": True,
                        "damage_detection": True,
                        "color_anomaly": True,
                        "pattern_enhancement": True,
                        "morphological_features": True
                    },
                    "weights": {
                        "texture_weight": 1.0,
                        "damage_weight": 0.8,
                        "pattern_weight": 0.7
                    }
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
    required_keys = ["dataset", "model", "training", "augmentation", "logging", "output"]
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

    print("Configuration is valid.")

def check_and_fix_json(json_path, dataset_name, data_dir, image_path):
    """Check if JSON file exists and is valid; only add missing features or create if damaged/missing."""
    try:
        # Try to load existing config
        with open(json_path, "r") as f:
            existing_config = json.load(f)

        # Get default config for comparison
        default_config = create_default_json_config(dataset_name, data_dir, image_path)

        # Recursively update missing keys while preserving existing values
        def update_missing(existing, default):
            for key, value in default.items():
                if key not in existing:
                    existing[key] = value
                elif isinstance(value, dict) and isinstance(existing[key], dict):
                    update_missing(existing[key], value)
            return existing

        # Update only missing features
        updated_config = update_missing(existing_config, default_config)

        # Save only if changes were made
        if updated_config != existing_config:
            print(f"Adding missing configuration parameters to {json_path}")
            with open(json_path, "w") as f:
                json.dump(updated_config, f, indent=4)

        return updated_config

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Config file is missing or corrupted: {str(e)}")
        print(f"Creating new configuration file at {json_path}")
        return create_default_json_config(dataset_name, data_dir, image_path)


def main():
    """Main function for user interaction."""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Initialize colorama
    from colorama import init, Fore, Back, Style
    init()

    # Header and branding
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.YELLOW}{'Visual Autoencoder Tool':^80}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")

    # Author and License information
    print(f"{Style.BRIGHT}{Fore.WHITE}{'Author: ':>30}{Fore.YELLOW}Ninan Sajeeth Philip{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'Organisation: ':>30}{Fore.LIGHTGREEN_EX}Artificial Intelligence Research and Intelligent Systems{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}{'':>30}Thelliyoor -689544 India{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'License: ':>30}{Fore.BLUE}Creative Commons License{Style.RESET_ALL}\n")
    print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
    # Data source selection menu
    print(f"{Style.BRIGHT}{Fore.CYAN}{'Select data source:':^80}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'1. ':>35}{Fore.YELLOW}Torchvision dataset (e.g., CIFAR10, MNIST){Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'2. ':>35}{Fore.YELLOW}URL to download dataset{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.WHITE}{'3. ':>35}{Fore.YELLOW}Local file/folder{Style.RESET_ALL}\n")

    data_source = input(f"{Style.BRIGHT}{Fore.WHITE}Enter your choice (1/2/3): {Style.RESET_ALL}")

    try:
        if data_source == "1":
            dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter dataset name (e.g., CIFAR10, MNIST): {Style.RESET_ALL}").upper()
            config = setup_dataset(dataset_name)

        elif data_source == "2":
            url = input(f"{Style.BRIGHT}{Fore.WHITE}Enter the URL to download the dataset: {Style.RESET_ALL}")
            dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter a name for the dataset: {Style.RESET_ALL}")
            data_dir = extract_and_organize(url, dataset_name, is_url=True)
            first_image = find_first_image(data_dir)
            config = create_default_json_config(dataset_name, data_dir, first_image)

        elif data_source == "3":
            source_path = input(f"{Style.BRIGHT}{Fore.WHITE}Enter the path to the local file/folder: {Style.RESET_ALL}")
            dataset_name = input(f"{Style.BRIGHT}{Fore.WHITE}Enter a name for the dataset: {Style.RESET_ALL}")

            if os.path.isfile(source_path) and source_path.endswith(('.zip', '.tar.gz', '.tgz', '.tar')):
                data_dir = extract_and_organize(source_path, dataset_name)
            else:
                data_dir = source_path

            first_image = find_first_image(data_dir)
            config = create_default_json_config(dataset_name, data_dir, first_image)

        else:
            print(f"{Style.BRIGHT}{Fore.RED}Invalid choice. Exiting...{Style.RESET_ALL}")
            return

        # Mode selection
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{'Select mode:':^80}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.WHITE}{'1. ':>35}{Fore.YELLOW}Train{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.WHITE}{'2. ':>35}{Fore.YELLOW}Predict{Style.RESET_ALL}")

        mode = input(f"{Style.BRIGHT}{Fore.WHITE}Enter your choice (1/2): {Style.RESET_ALL}")

        if mode == "1":
            print(f"\n{Style.BRIGHT}{Fore.CYAN}Training the model...{Style.RESET_ALL}")
            train_model(config)
        elif mode == "2":
            input_path = input(f"{Style.BRIGHT}{Fore.WHITE}Enter path to image or folder to reconstruct: {Style.RESET_ALL}")
            checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pth")

            if not os.path.exists(checkpoint_path):
                print(f"{Style.BRIGHT}{Fore.RED}No trained model found. Please train the model first.{Style.RESET_ALL}")
                return

            print(f"\n{Style.BRIGHT}{Fore.CYAN}Reconstructing image(s)...{Style.RESET_ALL}")
            reconstruct_image(input_path, checkpoint_path, dataset_name, config)
        else:
            print(f"{Style.BRIGHT}{Fore.RED}Invalid mode selected. Exiting...{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()

