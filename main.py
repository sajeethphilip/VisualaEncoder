import os
import json
import torch
from torchvision import datasets, transforms
from PIL import Image
from autoencoder.train import train_model
from autoencoder.utils import download_and_extract, setup_dataset,extract_and_organize,find_first_image,reconstruct_image,reconstruct_from_latent

def create_default_json_config(dataset_name, data_dir, image_path):
    """Create a default JSON configuration file by analyzing the dataset."""
    
    # Analyze dataset structure
    train_dir = os.path.join(data_dir, "train")
    
    # Get class information
    class_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    num_classes = len(class_folders)
    
    # Count total images and analyze first image
    total_images = 0
    class_distribution = {}
    
    # Read the first image to determine properties
    first_image = None
    for class_folder in class_folders:
        class_path = os.path.join(train_dir, class_folder)
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        class_distribution[class_folder] = num_images
        total_images += num_images
        
        # Get first image properties if not already done
        if first_image is None and images:
            first_image_path = os.path.join(class_path, images[0])
            first_image = Image.open(first_image_path)
            width, height = first_image.size
            in_channels = 1 if first_image.mode == "L" else 3  # Grayscale or RGB

    # Calculate mean and std from a sample of images
    print("Calculating dataset statistics...")
    means = []
    stds = []
    sample_size = min(1000, total_images)  # Limit sample size for efficiency
    sampled_images = []
    
    for class_folder in class_folders:
        class_path = os.path.join(train_dir, class_folder)
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        sample_per_class = sample_size // num_classes
        
        for img_file in images[:sample_per_class]:
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            if img.mode != ('L' if in_channels == 1 else 'RGB'):
                img = img.convert('L' if in_channels == 1 else 'RGB')
            img_tensor = transforms.ToTensor()(img)
            sampled_images.append(img_tensor)
    
    if sampled_images:
        sample_tensor = torch.stack(sampled_images)
        means = torch.mean(sample_tensor, dim=[0, 2, 3]).tolist()
        stds = torch.std(sample_tensor, dim=[0, 2, 3]).tolist()
    else:
        means = [0.5] * in_channels
        stds = [0.5] * in_channels

    # Print dataset analysis
    print(f"\nDataset Analysis:")
    print(f"Number of classes: {num_classes}")
    print(f"Total images: {total_images}")
    print("Class distribution:")
    for class_name, count in class_distribution.items():
        print(f"  {class_name}: {count} images")
    print(f"Image properties:")
    print(f"  Size: {width}x{height}")
    print(f"  Channels: {in_channels}")
    print(f"  Mean: {means}")
    print(f"  Std: {stds}")

    # Create configuration with analyzed parameters
    config = {
        "dataset": {
            "name": dataset_name,
            "type": "custom",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "input_size": [height, width],
            "mean": means,
            "std": stds,
            "train_dir": train_dir,
            "test_dir": os.path.join(data_dir, "test"),
            "image_type": "grayscale" if in_channels == 1 else "rgb",
            "class_distribution": class_distribution,
            "total_images": total_images
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
    print(f"\nCreated JSON configuration file at {json_path}")
    
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
            checkpoint_path = os.path.join(config["training"]["checkpoint_dir"], "best_model.pth")

            if not os.path.exists(checkpoint_path):
                print(f"{Style.BRIGHT}{Fore.RED}No trained model found. Please train the model first.{Style.RESET_ALL}")
                return

            print(f"\n{Style.BRIGHT}{Fore.CYAN}Reconstructing image(s)...{Style.RESET_ALL}")
            if config["training"].get("invert_DBNN", True):
                default_csv_path = os.path.join(f"data/{dataset_name}/latent_space")
                print(f"\nFound latent representations in: {default_csv_path}")
                csv_path = input(f"Enter the path to the latent CSV file/folder (default: {default_csv_path}): ") or default_csv_path
                reconstruct_from_latent(csv_path, checkpoint_path, dataset_name, config)
            else:
                input_path = input(f"{Style.BRIGHT}{Fore.WHITE}Enter path to image or folder to reconstruct: {Style.RESET_ALL}")
                # Original image-based reconstruction
                default_image_path = os.path.join(config["dataset"]["train_dir"])
                image_path = input(
                    f"Enter the path to the input image (default: {default_image_path}): "
                ) or default_image_path
                reconstruct_image(image_path, checkpoint_path, dataset_name, config)

        else:
            print(f"{Style.BRIGHT}{Fore.RED}Invalid mode selected. Exiting...{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()

