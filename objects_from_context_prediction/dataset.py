import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Dummy dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths1, image_paths2, labels, transform=None):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths1)

    def __getitem__(self, idx):
        image_path1 = self.image_paths1[idx]
        image_path2 = self.image_paths2[idx]
        label = self.labels[idx]

        try:
            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Skipping image {image_path1} or {image_path2} due to error: {e}")
            # Skip to the next valid data point
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
])

if __name__ == "__main__":
    # Example usage
    image_paths1 = ["/path/to/image1_1.png", "/path/to/image1_2.png"]
    image_paths2 = ["/path/to/image2_1.png", "/path/to/image2_2.png"]
    labels = [0, 1]

    dataset = CustomDataset(image_paths1, image_paths2, labels, transform=transform)
    print("Dataset created with length:", len(dataset))
