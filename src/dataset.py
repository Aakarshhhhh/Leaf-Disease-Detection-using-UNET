"""Dataset class for plant disease segmentation."""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PlantDiseaseDataset(Dataset):
    """Dataset for plant disease segmentation."""
    
    def __init__(self, images_dir, masks_dir, image_size=(256, 256), 
                 transform=None, is_training=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (assuming same name with different extension or in masks folder)
        mask_name = os.path.splitext(img_name)[0] + '.png'  # Adjust extension as needed
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create dummy mask if not found (for inference)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Normalize mask to 0-1
        mask = (mask > 0).astype(np.float32)  # Any non-zero value becomes 1
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Ensure mask has correct dimensions for loss function
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask, img_name


def get_transforms(image_size=(256, 256), is_training=True):
    """Get data augmentation transforms."""
    
    if is_training:
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_data_loaders(images_dir, masks_dir, batch_size=16, 
                       train_split=0.8, image_size=(256, 256)):
    """Create train and validation data loaders."""
    
    from torch.utils.data import DataLoader, random_split
    
    # Get transforms
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # Create full dataset
    full_dataset = PlantDiseaseDataset(
        images_dir, masks_dir, image_size, train_transform
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_indices = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset with different transforms
    val_dataset = PlantDiseaseDataset(
        images_dir, masks_dir, image_size, val_transform
    )
    
    # Use the same indices for validation
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset = PlantDiseaseDataset(
        "data/Train/Images", "data/Train/Masks",
        transform=get_transforms(is_training=True)
    )
    
    if len(dataset) > 0:
        image, mask, name = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image name: {name}")
        print(f"Mask min/max: {mask.min():.3f}/{mask.max():.3f}")
    else:
        print("No images found in dataset")