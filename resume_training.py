"""Resume training from the last checkpoint."""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import time
from datetime import datetime
import glob

# Add src to path
sys.path.append('src')

from src.model import get_model
from src.dataset import PlantDiseaseDataset, get_transforms
from src.utils import (
    CombinedLoss, MetricsTracker, calculate_iou, 
    calculate_dice_coefficient, visualize_predictions
)
import config


def find_latest_checkpoint(checkpoint_dir, prefix="checkpoint_augmented_epoch_"):
    """Find the latest checkpoint file."""
    pattern = os.path.join(checkpoint_dir, f"{prefix}*.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    epochs = []
    for cp in checkpoints:
        try:
            epoch_num = int(cp.split(prefix)[-1].split('.pth')[0])
            epochs.append((epoch_num, cp))
        except:
            continue
    
    if epochs:
        latest_epoch, latest_checkpoint = max(epochs, key=lambda x: x[0])
        return latest_checkpoint, latest_epoch
    
    return None


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint."""
    # Fix for PyTorch 2.6+ compatibility
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss


def create_data_loaders_presplit(train_images_dir, train_masks_dir, 
                                val_images_dir, val_masks_dir, 
                                batch_size=16, image_size=(256, 256)):
    """Create train and validation data loaders from pre-split dataset."""
    
    # Get transforms
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        train_images_dir, train_masks_dir, image_size, train_transform, True
    )
    
    val_dataset = PlantDiseaseDataset(
        val_images_dir, val_masks_dir, image_size, val_transform, False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        iou = calculate_iou(outputs, masks)
        dice = calculate_dice_coefficient(outputs, masks)
        
        metrics.update(loss.item(), iou, dice)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{iou:.4f}',
            'Dice': f'{dice:.4f}'
        })
    
    return metrics.get_averages()


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice_coefficient(outputs, masks)
            
            metrics.update(loss.item(), iou, dice)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}'
            })
    
    return metrics.get_averages()


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--total_epochs', type=int, default=30,
                       help='Total number of epochs to train (including already completed)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  Using device: {device}')
    
    # Find latest checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        # Extract epoch from filename
        try:
            start_epoch = int(checkpoint_path.split('epoch_')[-1].split('.pth')[0])
        except:
            start_epoch = 0
    else:
        checkpoint_info = find_latest_checkpoint(config.CHECKPOINTS_DIR)
        if checkpoint_info is None:
            print("❌ No checkpoint found! Starting from scratch...")
            start_epoch = 0
            checkpoint_path = None
        else:
            checkpoint_path, start_epoch = checkpoint_info
    
    if checkpoint_path:
        print(f'📂 Found checkpoint: {checkpoint_path}')
        print(f'🔄 Resuming from epoch: {start_epoch + 1}')
    
    # Create data loaders
    print('📂 Creating data loaders...')
    train_loader, val_loader = create_data_loaders_presplit(
        config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR,
        config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR,
        batch_size=args.batch_size,
        image_size=config.IMAGE_SIZE
    )
    
    # Create model
    model = get_model(n_channels=3, n_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'📥 Loading checkpoint from: {checkpoint_path}')
        loaded_epoch, loaded_loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f'✅ Loaded checkpoint from epoch {loaded_epoch} with loss {loaded_loss:.4f}')
        start_epoch = loaded_epoch + 1
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    print(f'\n🚀 Resuming training...')
    print(f'📅 Resume time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'🎯 Starting from epoch: {start_epoch}')
    print(f'🎯 Target epochs: {args.total_epochs}')
    print('='*80)
    
    for epoch in range(start_epoch, args.total_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f'Epoch {epoch:3d}/{args.total_epochs-1} ({epoch_time:.1f}s):')
        print(f'  Train - Loss: {train_metrics["loss"]:.4f}, IoU: {train_metrics["iou"]:.4f}, Dice: {train_metrics["dice"]:.4f}')
        print(f'  Val   - Loss: {val_metrics["loss"]:.4f}, IoU: {val_metrics["iou"]:.4f}, Dice: {val_metrics["dice"]:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model_augmented.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_model_path)
            print(f'  ✓ New best model saved! Val Loss: {best_val_loss:.4f}')
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f'checkpoint_augmented_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], checkpoint_path)
            print(f'  ✓ Checkpoint saved')
        
        print('-' * 60)
    
    print('='*80)
    print('🎉 TRAINING COMPLETED!')
    print(f'🏆 Best validation loss: {best_val_loss:.4f}')
    print('='*80)


if __name__ == '__main__':
    main()