import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from dataset import CustomDataset, transform
from model import ClassificationNet
import pandas as pd
import os
import argparse
from tqdm import tqdm, trange
import numpy as np

class CachedEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, data_df):
        """
        Dataset for loading cached embeddings.
        Args:
            embedding_dir: Directory containing all embeddings
            data_df: DataFrame containing coco_index and label information
        """
        self.embedding_dir = embedding_dir
        self.data_df = data_df
        
        # Verify data format by loading first file
        first_file = torch.load(os.path.join(embedding_dir, f"{data_df['coco_index'].iloc[0]}.pt"))
        if not all(k in first_file for k in ['latent1', 'latent2', 'label']):
            raise RuntimeError("Embedding files missing required data fields")
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        coco_idx = self.data_df['coco_index'].iloc[idx]
        file_path = os.path.join(self.embedding_dir, f"{coco_idx}.pt")
        data = torch.load(file_path)
        return data['latent1'], data['latent2'], data['label']

def prepare_embeddings(args, data_df, device, embedding_dir):
    """
    Prepare and cache VAE embeddings for missing files.
    Args:
        args: Command line arguments
        data_df: DataFrame containing image information
        device: torch device
        embedding_dir: Directory to save embeddings
    """
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Find missing embeddings using coco_index
    missing_indices = []
    for coco_idx in data_df['coco_index']:
        file_path = os.path.join(embedding_dir, f"{coco_idx}.pt")
        if not os.path.exists(file_path):
            missing_indices.append(coco_idx)
    
    if not missing_indices:
        print(f"All embeddings exist for this dataset, skipping encoding...")
        return
    
    print(f"Found {len(missing_indices)} missing embeddings. Processing...")
    
    # Load VAE
    model_id = "/data2/Tianze/weights/stable-diffusion-2-1"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    vae.eval()
    
    # Create dataset only for missing indices
    base_path = "../task_data"
    missing_df = data_df[data_df['coco_index'].isin(missing_indices)].reset_index(drop=True)
    
    dataset = CustomDataset(
        [os.path.join(base_path, "images/training_val_images", f"{idx}.png") 
         for idx in missing_df['coco_index']],
        [os.path.join(base_path, "masks/bbox_masks_training_val", f"{idx}.png") 
         for idx in missing_df['coco_index']],
        missing_df['label'].tolist(),
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Create progress bar for missing files
    pbar = tqdm(total=len(missing_indices), desc=f"Processing missing embeddings")
    
    # Generate and save missing embeddings
    with torch.no_grad():
        for batch_idx, (images1, images2, labels) in enumerate(dataloader):
            images1, images2 = images1.to(device), images2.to(device)
            
            # Encode images
            latent1 = vae.encode(images1).latent_dist.mean
            latent2 = vae.encode(images2).latent_dist.mean
            
            # Save individual embeddings
            for i in range(len(labels)):
                current_idx = batch_idx * dataloader.batch_size + i
                if current_idx >= len(missing_df):
                    break
                    
                coco_idx = missing_df['coco_index'].iloc[current_idx]
                save_path = os.path.join(embedding_dir, f"{coco_idx}.pt")
                
                torch.save({
                    'latent1': latent1[i].cpu(),
                    'latent2': latent2[i].cpu(),
                    'label': labels[i].item(),
                }, save_path)
                pbar.update(1)
    
    pbar.close()
    del vae
    torch.cuda.empty_cache()

def evaluate(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for latent1, latent2, labels in dataloader:
            latent1, latent2, labels = latent1.to(device), latent2.to(device), labels.to(device)
            
            latent1_flat = latent1.view(latent1.size(0), -1)
            latent2_flat = latent2.view(latent2.size(0), -1)
            
            outputs = model(latent1_flat, latent2_flat)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train(args, embedding_dir, train_df, val_df):
    """
    Train the model using cached embeddings.
    Args:
        args: Command line arguments
        embedding_dir: Directory containing all embeddings
        train_df: DataFrame containing training data information
        val_df: DataFrame containing validation data information
    """
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets using DataFrames
    train_dataset = CachedEmbeddingDataset(embedding_dir, train_df)
    val_dataset = CachedEmbeddingDataset(embedding_dir, val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    # Initialize the classification network
    embedding_dim = 4 * 32 * 32
    num_classes = 80
    classification_net = ClassificationNet(embedding_dim, num_classes).to(device)

    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    # Training loop
    num_epochs = 120
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/training_log.txt"
    
    best_acc = 0.0
    checkpoint_dir = f"../checkpoints/objects_from_context_prediction/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    epoch_pbar = trange(num_epochs, desc="Training Progress")
    for epoch in epoch_pbar:
        # Training phase
        classification_net.train()
        running_loss = 0.0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs-1}", 
                         leave=False, unit="batch")
        
        for batch_idx, (latent1, latent2, labels) in enumerate(batch_pbar):
            latent1, latent2, labels = latent1.to(device), latent2.to(device), labels.to(device)
            
            latent1_flat = latent1.view(latent1.size(0), -1)
            latent2_flat = latent2.view(latent2.size(0), -1)
            
            outputs = classification_net(latent1_flat, latent2_flat)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Validation phase
        val_acc = evaluate(classification_net, val_loader, device)
        average_loss = running_loss / len(train_loader)
        
        # Log results
        epoch_pbar.set_postfix({
            'avg_loss': f'{average_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%'
        })
        
        with open(log_file, "a") as f:
            f.write(f"Epoch [{epoch}/{num_epochs-1}], "
                   f"Average Loss: {average_loss:.4f}, "
                   f"Validation Accuracy: {val_acc:.2f}%\n")
        
        # Save latest checkpoint (overwrite previous)
        latest_ckpt_path = os.path.join(checkpoint_dir, "classification_net_latest.pth")
        torch.save(classification_net.state_dict(), latest_ckpt_path)
        
        # Save best model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt_path = os.path.join(checkpoint_dir, "classification_net_best.pth")
            torch.save(classification_net.state_dict(), best_ckpt_path)
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument('--device', type=int, default=0,
                       help="GPU device ID to use (e.g., 0, 1, 2, etc.)")
    
    args = parser.parse_args()
    
    # Common embedding directory for both train and validation
    embedding_dir = "../task_data/cache/cached_embeddings_train_val_op"
    
    # Read train and validation data
    train_df = pd.read_csv('../task_data/objects_from_context_prediction/training_data.csv')
    val_df = pd.read_csv('../task_data/objects_from_context_prediction/validation_data.csv')
    
    # First prepare embeddings for training data
    print("Checking and preparing training embeddings...")
    prepare_embeddings(args, train_df, torch.device(f"cuda:{args.device}"), embedding_dir)
    
    # Then prepare embeddings for validation data
    print("Checking and preparing validation embeddings...")
    prepare_embeddings(args, val_df, torch.device(f"cuda:{args.device}"), embedding_dir)
    
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    print(f"\nStart to train model ClassificationNet on cuda:{args.device}")
    train(args, embedding_dir, train_df, val_df)