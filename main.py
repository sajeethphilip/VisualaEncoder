import os
import json
import torch
from torchvision import datasets, transforms
from PIL import Image
from autoencoder.train import train_model
from autoencoder.reconstruct import reconstruct_image
from autoencoder.data_loader import load_local_dataset, load_dataset_config
from autoencoder.utils import download_and_extract, setup_dataset

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
    """Check if the JSON file is valid; if not, replace it with a default."""
    try:
        with open(json_path, "r") as f:
            config = json.load(f)
        # Validate the JSON structure
        if "dataset" not in config:
            raise ValueError("Invalid JSON structure.")
        return config
    except (json.JSONDecodeError, ValueError, FileNotFoundError):
        print(f"Invalid or corrupted JSON file at {json_path}. Replacing with default...")
        return create_default_json_config(dataset_name, data_dir, image_path)

def main():
    """Main function for user interaction."""
    print("Welcome to the Autoencoder Tool!")

    # Step 1: Select data source
    print("\nSelect data source:")
    print("1. Torchvision dataset (e.g., CIFAR10, MNIST)")
    print("2. URL to download dataset")
    print("3. Local file")
    data_source = input("Enter your choice (1/2/3): ")

    config = None  # Initialize config variable
    dataset_name = None  # Initialize dataset_name variable

    if data_source == "1":
        # Load torchvision dataset
        dataset_name = input("Enter dataset name (e.g., CIFAR10, MNIST, CIFAR100): ").upper()
        data_dir = os.path.join("data", dataset_name)
        os.makedirs(data_dir, exist_ok=True)
        config = setup_dataset(dataset_name)

    elif data_source == "2":
        # Download dataset from URL
        url = input("Enter the URL to download the dataset: ")
        dataset_name = input("Enter a name for the dataset: ")
        data_dir = os.path.join("data", dataset_name)
        download_and_extract(url, data_dir)

        # Check if JSON file exists and is valid
        json_path = os.path.join(data_dir, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            print(f"JSON configuration file not found at {json_path}. Creating one...")
            # Find the first image in the dataset
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        config = create_default_json_config(dataset_name, data_dir, image_path)
                        break
                break
        else:
            config = check_and_fix_json(json_path, dataset_name, data_dir, os.path.join(data_dir, "sample_image.png"))

        # Check if dataset has train/test folders
        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            combine = input("Dataset has train and test folders. Combine them? (y/n): ").lower()
            if combine == "y":
                # Combine train and test folders
                dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
            else:
                # Use only the train folder
                dataset = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())
        else:
            # Use the entire dataset
            dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())

    elif data_source == "3":
        # Load local file
        dataset_name = input("Enter the path to the local dataset folder: ")
        data_dir = dataset_name
        dataset_name = os.path.basename(os.path.normpath(dataset_name))

        # Check if JSON file exists and is valid
        json_path = os.path.join(data_dir, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            print(f"JSON configuration file not found at {json_path}. Creating one...")
            # Find the first image in the dataset
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        config = create_default_json_config(dataset_name, data_dir, image_path)
                        break
                break
        else:
            config = check_and_fix_json(json_path, dataset_name, data_dir, os.path.join(data_dir, "sample_image.png"))

        # Load dataset
        dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())

    else:
        raise ValueError("Invalid choice. Please select 1, 2, or 3.")

    # Ensure config is loaded
    if config is None:
        raise ValueError("Configuration not loaded. Please check the dataset path and JSON file.")

    # Debug: Print the config to verify its contents
    print("\nConfiguration loaded:")
    print(json.dumps(config, indent=4))

    # Validate the configuration
    validate_config(config)

    # Step 2: Train or Predict
    print("\nSelect mode:")
    print("1. Train")
    print("2. Predict")
    mode = input("Enter your choice (1/2): ")

    if mode == "1":
        # Train the model
        print("\nTraining the model...")
        train_model(config)  # Pass the configuration
    elif mode == "2":
        # Predict (reconstruct images)
        print("\nReconstructing images...")

        # Default values for reconstruction
        default_checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pth")
        default_image_path = os.path.join(config["dataset"]["train_dir"], "sample_image.png")

        # Prompt user for checkpoint path (with default)
        checkpoint_path = input(
            f"Enter the path to the trained model checkpoint (default: {default_checkpoint_path}): "
        ) or default_checkpoint_path

        # Prompt user for image path (with default)
        image_path = input(
            f"Enter the path to the input image (default: {default_image_path}): "
        ) or default_image_path

        # Reconstruct the image
        reconstruct_image(image_path, checkpoint_path, dataset_name)  # Pass dataset_name instead of config
    else:
        raise ValueError("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()

