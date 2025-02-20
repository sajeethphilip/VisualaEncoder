import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from autoencoder.model import Autoencoder, ModifiedAutoencoder
from autoencoder.utils import get_device, save_latent_space, save_embeddings_as_csv, save_checkpoint, update_progress, display_confusion_matrix
from autoencoder.utils import load_checkpoint, load_local_dataset, load_dataset_config, save_1d_latent_to_csv, save_batch_latents, display_header
from datetime import datetime

def train_model(config):
    """Train the autoencoder model with complete latent space generation after each epoch."""
    device = get_device()
    print(f"Using device: {device}")

    # Get terminal dimensions
    terminal_height = os.get_terminal_size().lines
    header_height = 10  # Reserve space for header
    display_header()

    # Dataset setup
    dataset_config = config["dataset"]
    dataset = load_local_dataset(dataset_config["name"])
    class_names = dataset.classes
    num_classes = len(class_names)

    # Initialize confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=device)

    # Model setup
    model = ModifiedAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["optimizer"]["weight_decay"]
    )
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        confusion_matrix.zero_()

        # Regular training loop
        for batch_idx, (images, labels) in enumerate(dataset):
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)

            images = images.to(device)
            labels = labels.to(device)

            reconstructed, latent_1d = model(images)

            # Update confusion matrix
            with torch.no_grad():
                pred_labels = torch.argmax(latent_1d, dim=1)
                for t, p in zip(labels, pred_labels):
                    confusion_matrix[t.long(), p.long()] += 1

            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % config.get("display_interval", 10) == 0:
                avg_loss = running_loss / (batch_idx + 1)
                display_confusion_matrix(confusion_matrix, class_names, terminal_height, header_height)
                print(f"Epoch: {epoch+1}/{config['training']['epochs']} | Batch: {batch_idx+1}/348 | Loss: {avg_loss:.4f}")

        # After each epoch, generate and save latent representations for all classes
        print(f"\nGenerating latent representations for epoch {epoch+1}...")
        model.eval()
        
        # Process each class separately
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(dataset_config["train_dir"], class_name)
            latent_dir = os.path.join(f"data/{dataset_config['name']}/latent_space/train", class_name)
            recon_dir = os.path.join(f"data/{dataset_config['name']}/reconstructed_images/train", class_name)
            
            os.makedirs(latent_dir, exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)

            # Get all images in the class
            class_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in class_files:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path)
                img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    reconstructed, latent_1d = model(img_tensor)

                    # Save latent representation
                    base_name = os.path.splitext(img_file)[0]
                    latent_path = os.path.join(latent_dir, f"{base_name}.csv")
                    metadata = {
                        'epoch': epoch + 1,
                        'class': class_name,
                        'filename': img_file,
                        'timestamp': datetime.now().isoformat()
                    }
                    save_1d_latent_to_csv(latent_1d, img_path, dataset_config["name"], metadata)

                    # Save reconstructed image
                    recon_path = os.path.join(recon_dir, img_file)
                    recon_img = transforms.ToPILImage()(reconstructed.squeeze(0).cpu())
                    recon_img.save(recon_path)

        print(f"Completed epoch {epoch+1} with latent space generation and reconstruction")

    return model


def save_final_representations(model, loader, device, dataset_name):
    """Save the final latent space and embeddings."""
    latent_space = []
    embeddings_space = []

    with torch.no_grad():
        for images, _ in loader:
            # Ensure images are tensors and on the correct device
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images, dtype=torch.float32)
            images = images.to(device)
            
            # Forward pass
            reconstructed, latent, embeddings = model(images)
            latent_space.append(latent.cpu())
            embeddings_space.append(embeddings.cpu())

    # Concatenate all batches
    latent_space = torch.cat(latent_space, dim=0)
    embeddings_space = torch.cat(embeddings_space, dim=0)

    # Save representations
    save_latent_space(latent_space, dataset_name, "latent.pkl")
    save_embeddings_as_csv(embeddings_space, dataset_name, "embeddings.csv")

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    train_model(config)
