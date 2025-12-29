"""Training script for augmented dataset with progress tracking."""

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

# Add src to path
sys.path.append('src')

from src.model import get_model
from src.dataset import PlantDiseaseDataset, get_transforms
from src.utils import (
    CombinedLoss, MetricsTracker, calculate_iou, 
    calculate_dice_coefficient, visualize_predictions
)
import config


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
        shuffle=True, num_workers=2, pin_memory=True  # GPU optimizations
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True  # GPU optimizations
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
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
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
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
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


def save_training_log(history, filepath):
    """Save training history to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)


def print_progress_update(epoch, train_metrics, val_metrics, best_val_loss, epoch_time):
    """Print detailed progress update."""
    print(f'\n{"="*80}')
    print(f'🚀 EPOCH {epoch} PROGRESS UPDATE - {datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*80}')
    print(f'⏱️  Epoch Time: {epoch_time:.1f}s')
    print(f'📈 Training   - Loss: {train_metrics["loss"]:.4f} | IoU: {train_metrics["iou"]:.4f} | Dice: {train_metrics["dice"]:.4f}')
    print(f'📊 Validation - Loss: {val_metrics["loss"]:.4f} | IoU: {val_metrics["iou"]:.4f} | Dice: {val_metrics["dice"]:.4f}')
    
    if val_metrics['loss'] < best_val_loss:
        print(f'🎉 NEW BEST MODEL! Validation Loss improved: {best_val_loss:.4f} → {val_metrics["loss"]:.4f}')
    
    # Performance indicators
    if val_metrics['dice'] > 0.7:
        print(f'🏆 EXCELLENT: Dice Score > 70% ({val_metrics["dice"]:.1%})')
    elif val_metrics['dice'] > 0.6:
        print(f'✅ GOOD: Dice Score > 60% ({val_metrics["dice"]:.1%})')
    else:
        print(f'📈 IMPROVING: Dice Score = {val_metrics["dice"]:.1%}')
    
    print(f'{"="*80}\n')


def main():
    parser = argparse.ArgumentParser(description='Train plant disease segmentation model with augmented data')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  Using device: {device}')
    
    # Create data loaders
    print('📂 Creating data loaders for augmented dataset...')
    train_loader, val_loader = create_data_loaders_presplit(
        config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR,
        config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR,
        batch_size=args.batch_size,
        image_size=config.IMAGE_SIZE
    )
    
    print(f'📊 Dataset Statistics:')
    print(f'   🔹 Train samples: {len(train_loader.dataset):,}')
    print(f'   🔹 Validation samples: {len(val_loader.dataset):,}')
    print(f'   🔹 Train batches: {len(train_loader):,}')
    print(f'   🔹 Validation batches: {len(val_loader):,}')
    
    # Create model
    model = get_model(n_channels=3, n_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'🧠 Model parameters: {total_params:,}')
    
    # Loss function and optimizer
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_ious': [],
        'val_ious': [],
        'train_dice': [],
        'val_dice': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f'\n🚀 Starting training with augmented dataset...')
    print(f'📅 Training started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'🎯 Target epochs: {args.epochs}')
    print('='*80)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Store history
        history['train_losses'].append(train_metrics['loss'])
        history['val_losses'].append(val_metrics['loss'])
        history['train_ious'].append(train_metrics['iou'])
        history['val_ious'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - epoch_start
        
        # Print progress update (detailed for first 5 epochs)
        if epoch < 5:
            print_progress_update(epoch, train_metrics, val_metrics, best_val_loss, epoch_time)
        else:
            # Standard progress for later epochs
            print(f'Epoch {epoch:3d}/{args.epochs-1} ({epoch_time:.1f}s):')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, IoU: {train_metrics["iou"]:.4f}, Dice: {train_metrics["dice"]:.4f}')
            print(f'  Val   - Loss: {val_metrics["loss"]:.4f}, IoU: {val_metrics["iou"]:.4f}, Dice: {val_metrics["dice"]:.4f}')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model with augmented suffix
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model_augmented.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_model_path)
            if epoch < 5:
                print(f'💾 Best model saved: {best_model_path}')
            else:
                print(f'  ✓ New best model saved! Val Loss: {best_val_loss:.4f}')
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f'checkpoint_augmented_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], checkpoint_path)
            print(f'  ✓ Checkpoint saved')
        
        # Visualize predictions periodically
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                images, masks, _ = next(iter(val_loader))
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                
                viz_path = os.path.join(config.PREDICTIONS_DIR, f'predictions_augmented_epoch_{epoch}.png')
                visualize_predictions(images[:4], masks[:4], predictions[:4], viz_path)
                print(f'  ✓ Predictions saved')
        
        if epoch < 5:
            print()  # Extra spacing for first 5 epochs
        else:
            print('-' * 60)
    
    # Save final model
    final_model_path = os.path.join(config.CHECKPOINTS_DIR, 'final_model_augmented.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, history['val_losses'][-1], final_model_path)
    
    # Save training history
    history_path = os.path.join(config.OUTPUTS_DIR, 'training_history_augmented.json')
    save_training_log(history, history_path)
    
    total_time = time.time() - start_time
    
    print('='*80)
    print('🎉 TRAINING COMPLETED WITH AUGMENTED DATASET!')
    print('='*80)
    print(f'⏱️  Total time: {total_time/60:.1f} minutes')
    print(f'🏆 Best validation loss: {best_val_loss:.4f}')
    print(f'📈 Final validation IoU: {history["val_ious"][-1]:.4f}')
    print(f'📊 Final validation Dice: {history["val_dice"][-1]:.4f}')
    print(f'💾 Best model saved to: {best_model_path}')
    print(f'📁 Training history saved to: {history_path}')
    print('='*80)


if __name__ == '__main__':
    main()