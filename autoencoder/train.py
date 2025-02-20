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
    """Train the autoencoder model with proper batch handling."""
    device = get_device()
    print(f"Using device: {device}")

    # Dataset setup with DataLoader
    dataset_config = config["dataset"]
    dataset = load_local_dataset(dataset_config["name"])
    
    # Create DataLoader with proper batch size
    batch_size = config["training"]["batch_size"]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        drop_last=True  # Ensure consistent batch sizes
    )
    
    class_names = dataset.classes
    num_classes = len(class_names)
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
        
        # Process batches
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            reconstructed, latent_1d = model(images)
            
            # Update confusion matrix
            with torch.no_grad():
                pred_labels = torch.argmax(latent_1d, dim=1)
                for t, p in zip(labels, pred_labels):
                    confusion_matrix[t.long(), p.long()] += 1

            # Compute loss and optimize
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Save batch latents
            batch_paths = [dataset.imgs[i][0] for i in range(
                batch_idx * batch_size, 
                min((batch_idx + 1) * batch_size, len(dataset))
            )]
            
            batch_metadata = {
                'epoch': epoch + 1,
                'batch': batch_idx,
                'loss': loss.item(),
                'timestamp': datetime.now().isoformat()
            }
            save_batch_latents(latent_1d.detach().cpu(), batch_paths, dataset_config["name"], batch_metadata)

            if batch_idx % config.get("display_interval", 10) == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Epoch: {epoch+1}/{config['training']['epochs']} | "
                      f"Batch: {batch_idx+1}/{len(dataloader)} | Loss: {avg_loss:.4f}")

        # After each epoch, generate complete latent space
        print("\nGenerating complete latent space...")
        model.eval()
        
        # Process each class
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(dataset_config["train_dir"], class_name)
            latent_dir = os.path.join(f"data/{dataset_config['name']}/latent_space/train", class_name)
            recon_dir = os.path.join(f"data/{dataset_config['name']}/reconstructed_images/train", class_name)
            
            os.makedirs(latent_dir, exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)
            
            # Get all images in the class
            class_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Process class images in batches
            for i in range(0, len(class_files), batch_size):
                batch_files = class_files[i:i + batch_size]
                batch_images = []
                
                for img_file in batch_files:
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path)
                    img_tensor = transforms.ToTensor()(img)
                    batch_images.append(img_tensor)
                
                # Stack batch tensors
                batch_tensor = torch.stack(batch_images).to(device)
                
                with torch.no_grad():
                    reconstructed, latent_1d = model(batch_tensor)
                    
                    # Save individual results
                    for j, img_file in enumerate(batch_files):
                        base_name = os.path.splitext(img_file)[0]
                        
                        # Save latent representation
                        save_1d_latent_to_csv(
                            latent_1d[j], 
                            os.path.join(class_path, img_file),
                            dataset_config["name"],
                            {'epoch': epoch + 1, 'class': class_name}
                        )
                        
                        # Save reconstructed image
                        recon_img = transforms.ToPILImage()(reconstructed[j].cpu())
                        recon_img.save(os.path.join(recon_dir, img_file))

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
