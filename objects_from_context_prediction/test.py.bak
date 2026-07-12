import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from diffusers import AutoencoderKL
from dataset import CustomDataset, transform
from model import ClassificationNet, load_classification_model
from tqdm import tqdm
import random
import csv
import argparse
import os
import json

class CachedTestEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, test_df):
        """
        Dataset for loading cached embeddings.
        Args:
            embedding_dir: Directory containing all embeddings
            test_df: DataFrame containing coco_index and label information
        """
        self.embedding_dir = embedding_dir
        self.test_df = test_df
        
        # Verify data format by loading first file
        first_file = torch.load(os.path.join(embedding_dir, f"{test_df['coco_index'].iloc[0]}.pt"))
        if not all(k in first_file for k in ['latent1', 'latent2', 'label']):
            raise RuntimeError("Embedding files missing required data fields")
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self, idx):
        coco_idx = self.test_df['coco_index'].iloc[idx]
        file_path = os.path.join(self.embedding_dir, f"{coco_idx}.pt")
        data = torch.load(file_path)
        return data['latent1'], data['latent2'], data['label']

def prepare_test_embeddings(test_df, device):
    """Prepare and cache VAE embeddings for test data if they don't exist"""
    embedding_dir = f"../task_data/cache/cached_embeddings_testing_op"
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Find missing embeddings using coco_index
    missing_indices = []
    for coco_idx in test_df['coco_index']:
        file_path = os.path.join(embedding_dir, f"{coco_idx}.pt")
        if not os.path.exists(file_path):
            missing_indices.append(coco_idx)
    
    if not missing_indices:
        print("All embeddings exist for test dataset, skipping encoding...")
        return embedding_dir
    
    print(f"Found {len(missing_indices)} missing embeddings. Processing...")
    
    # Load VAE, choose your own path
    vae_model_id = "/data2/Tianze/weights/stable-diffusion-2-1"
    vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae").to(device)
    vae.eval()
    
    # Create dataset only for missing indices
    base_path = "../task_data"
    missing_df = test_df[test_df['coco_index'].isin(missing_indices)].reset_index(drop=True)
    
    dataset = CustomDataset(
        [os.path.join(base_path, "images/testing_images", f"{idx}.png") 
         for idx in missing_df['coco_index']],
        [os.path.join(base_path, "masks/bbox_masks_testing", f"{idx}.png") 
         for idx in missing_df['coco_index']],
        missing_df['label'].tolist(),
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Create progress bar for missing files
    pbar = tqdm(total=len(missing_indices), desc=f"Processing missing embeddings")
    
    # Generate and save embeddings
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
    
    return embedding_dir

def load_category_mapping():
    """Load the mapping from category to supercategory"""
    mapping_file = "../task_data/inpainting_info/supercategory_label_mappings.json"
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Create a mapping from category index to supercategory index
    category_to_supercategory = {}
    for category_idx, data in mapping_data.items():
        category_to_supercategory[int(category_idx)] = data["supercategory_index"]
    
    return category_to_supercategory

def test_normal(model_path, embedding_dir, test_df, batch_size, embedding_dim, num_classes, device):
    """Test the model on standard category classification"""
    # Load the trained classification model
    model = load_classification_model(model_path, embedding_dim, num_classes, device)
    
    # Set model to evaluation mode
    model.eval()

    # Create dataset with cached embeddings
    dataset = CachedTestEmbeddingDataset(embedding_dir, test_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize counters for accuracy calculation
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0

    # Testing loop
    with torch.no_grad():
        for latent1, latent2, labels in tqdm(dataloader, desc="Testing normal categories", unit="batch"):
            latent1, latent2, labels = latent1.to(device), latent2.to(device), labels.to(device)
            # Flatten latent vectors just before passing to the network
            latent1_flat = latent1.view(latent1.size(0), -1)
            latent2_flat = latent2.view(latent2.size(0), -1)
            
            # Forward pass through the classification network
            outputs = model(latent1_flat, latent2_flat)
            
            # Top-k predictions
            _, topk_pred = torch.topk(outputs, k=10, dim=1)
            # Update counters
            total += labels.size(0)
            top1_correct += (topk_pred[:, :1] == labels.view(-1, 1)).sum().item()
            top3_correct += (topk_pred[:, :3] == labels.view(-1, 1)).any(dim=1).sum().item()
            top5_correct += (topk_pred[:, :5] == labels.view(-1, 1)).any(dim=1).sum().item()
            top10_correct += (topk_pred[:, :10] == labels.view(-1, 1)).any(dim=1).sum().item()

    # Calculate accuracies
    top1_accuracy = 100 * top1_correct / total
    top3_accuracy = 100 * top3_correct / total
    top5_accuracy = 100 * top5_correct / total
    top10_accuracy = 100 * top10_correct / total
    
    print(f'Category Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Category Top-3 Accuracy: {top3_accuracy:.2f}%')
    print(f'Category Top-5 Accuracy: {top5_accuracy:.2f}%')
    print(f'Category Top-10 Accuracy: {top10_accuracy:.2f}%')

    return top1_accuracy, top3_accuracy, top5_accuracy, top10_accuracy

def test_supercategory(model_path, embedding_dir, test_df, supercategory_df, batch_size, embedding_dim, num_classes, device):
    """Test the model on supercategory classification"""
    # Load the trained classification model

    model = load_classification_model(model_path, embedding_dim, num_classes, device)
    
    # Set model to evaluation mode
    model.eval()

    # Load category to supercategory mapping
    category_to_supercategory = load_category_mapping()
    
    # Create dataset with cached embeddings using original test_df for compatibility with embeddings
    dataset = CachedTestEmbeddingDataset(embedding_dir, test_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create a dictionary to map coco_index to supercategory label
    coco_to_supercategory = dict(zip(supercategory_df['coco_index'], supercategory_df['label']))

    # Initialize counters for accuracy calculation
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0

    # Testing loop
    with torch.no_grad():
        for latent1, latent2, _ in tqdm(dataloader, desc="Testing supercategories", unit="batch"):
            latent1, latent2 = latent1.to(device), latent2.to(device)
            
            # Get indices for this batch
            indices = test_df['coco_index'].iloc[total:total+len(latent1)].tolist()
            
            # Get supercategory labels for these indices
            batch_super_labels = torch.tensor([coco_to_supercategory[idx] for idx in indices], device=device)
            
            # Flatten latent vectors just before passing to the network
            latent1_flat = latent1.view(latent1.size(0), -1)
            latent2_flat = latent2.view(latent2.size(0), -1)
            
            # Forward pass through the classification network - outputs are at category level
            outputs = model(latent1_flat, latent2_flat)
            
            # Get the top predictions at category level
            _, topk_pred_categories = torch.topk(outputs, k=num_classes, dim=1)  # Get all predictions

            # Convert category predictions to supercategory predictions
            batch_size = topk_pred_categories.size(0)
            supercategory_preds = []
            
            for i in range(batch_size):
                # Convert each category prediction to its corresponding supercategory
                super_preds = [category_to_supercategory[cat.item()] for cat in topk_pred_categories[i]]
                # Remove duplicates while preserving order
                unique_super_preds = []
                for pred in super_preds:
                    if pred not in unique_super_preds:
                        unique_super_preds.append(pred)
                supercategory_preds.append(unique_super_preds[:10])  # Keep top 10 unique supercategories
            # Update counters
            for i in range(batch_size):
                true_super = batch_super_labels[i].item()
                pred_supers = supercategory_preds[i]
                
                if len(pred_supers) >= 1 and pred_supers[0] == true_super:
                    top1_correct += 1
                if len(pred_supers) >= 3 and true_super in pred_supers[:3]:
                    top3_correct += 1
                if len(pred_supers) >= 5 and true_super in pred_supers[:5]:
                    top5_correct += 1
                if len(pred_supers) >= 10 and true_super in pred_supers[:10]:
                    top10_correct += 1
            
            total += batch_size

    # Calculate accuracies
    top1_accuracy = 100 * top1_correct / total
    top3_accuracy = 100 * top3_correct / total
    top5_accuracy = 100 * top5_correct / total
    top10_accuracy = 100 * top10_correct / total
    
    print(f'Supercategory Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Supercategory Top-3 Accuracy: {top3_accuracy:.2f}%')
    print(f'Supercategory Top-5 Accuracy: {top5_accuracy:.2f}%')
    print(f'Supercategory Top-10 Accuracy: {top10_accuracy:.2f}%')

    return top1_accuracy, top3_accuracy, top5_accuracy, top10_accuracy

if __name__ == "__main__":
    # Settings
    batch_size = 64
    embedding_dim = 4 * 32 * 32
    num_classes = 80
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Read test data for compatibility with existing embeddings
    test_df = pd.read_csv('../task_data/objects_from_context_prediction/testing_data.csv')
    
    # Read supercategory test data
    supercategory_df = pd.read_csv('../task_data/objects_from_context_prediction/testing_data_supercategory.csv')
    
    # Prepare embeddings (uses original test_df for compatibility)
    embedding_dir = prepare_test_embeddings(test_df, device)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Test the best model
    model_path = f"../checkpoints/objects_from_context_prediction/classification_net_best.pth"
    print(f"\nTesting model: {model_path}")
    
    # Run normal category test
    print("\n=== Category Classification Test ===")
    cat_top1, cat_top3, cat_top5, cat_top10 = test_normal(
        model_path, embedding_dir, test_df, batch_size, embedding_dim, 
        num_classes, device)
    
    # Run supercategory test
    print("\n=== Supercategory Classification Test ===")
    super_top1, super_top3, super_top5, super_top10 = test_supercategory(
        model_path, embedding_dir, test_df, supercategory_df, batch_size, embedding_dim, 
        num_classes, device)

    # Save combined results to CSV
    results = [{
        'embedding_dim': embedding_dim,
        'category_top1_accuracy': cat_top1,
        'category_top3_accuracy': cat_top3,
        'category_top5_accuracy': cat_top5,
        'category_top10_accuracy': cat_top10,
        'supercategory_top1_accuracy': super_top1,
        'supercategory_top3_accuracy': super_top3,
        'supercategory_top5_accuracy': super_top5,
        'supercategory_top10_accuracy': super_top10
    }]
    
    results_df = pd.DataFrame(results)
    results_file = f'results/test_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nCombined test results saved to {results_file}")