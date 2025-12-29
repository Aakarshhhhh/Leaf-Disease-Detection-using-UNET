"""Training script for plant disease segmentation model."""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.dataset import create_data_loaders
from src.utils import (
    CombinedLoss, MetricsTracker, calculate_iou, 
    calculate_dice_coefficient, visualize_predictions,
    plot_training_history
)
import config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        iou = calculate_iou(outputs, masks)
        dice = calculate_dice_coefficient(outputs, masks)
        
        metrics.update(loss.item(), iou, dice)
        
        # Update progress bar
        if batch_idx % config.LOG_EVERY == 0:
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


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def main():
    parser = argparse.ArgumentParser(description='Train plant disease segmentation model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--images_dir', type=str, default=config.IMAGES_DIR,
                       help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, default=config.MASKS_DIR,
                       help='Path to masks directory')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('Creating data loaders...')
    train_loader, val_loader = create_data_loaders(
        args.images_dir, args.masks_dir,
        batch_size=config.BATCH_SIZE,
        train_split=config.TRAIN_SPLIT,
        image_size=config.IMAGE_SIZE
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Create model
    model = get_model(n_channels=3, n_classes=config.NUM_CLASSES)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(config.MODELS_DIR, 'runs'))
    
    # Training history
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dice, val_dice = [], []
    
    best_val_loss = float('inf')
    
    print('Starting training...')
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, optimizer, device, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
        writer.add_scalar('IoU/Validation', val_metrics['iou'], epoch)
        writer.add_scalar('Dice/Train', train_metrics['dice'], epoch)
        writer.add_scalar('Dice/Validation', val_metrics['dice'], epoch)
        
        # Store history
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        train_dice.append(train_metrics['dice'])
        val_dice.append(val_metrics['dice'])
        
        # Print epoch results
        print(f'Epoch {epoch}:')
        print(f'  Train - Loss: {train_metrics["loss"]:.4f}, IoU: {train_metrics["iou"]:.4f}, Dice: {train_metrics["dice"]:.4f}')
        print(f'  Val   - Loss: {val_metrics["loss"]:.4f}, IoU: {val_metrics["iou"]:.4f}, Dice: {val_metrics["dice"]:.4f}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_model_path)
            print(f'  New best model saved with validation loss: {best_val_loss:.4f}')
        
        # Save checkpoint periodically
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], checkpoint_path)
        
        # Visualize predictions periodically
        if (epoch + 1) % (config.SAVE_EVERY * 2) == 0:
            model.eval()
            with torch.no_grad():
                images, masks, _ = next(iter(val_loader))
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                
                viz_path = os.path.join(config.PREDICTIONS_DIR, f'predictions_epoch_{epoch}.png')
                visualize_predictions(images[:4], masks[:4], predictions[:4], viz_path)
    
    # Save final model
    final_model_path = os.path.join(config.CHECKPOINTS_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, config.NUM_EPOCHS - 1, val_losses[-1], final_model_path)
    
    # Plot training history
    history_path = os.path.join(config.OUTPUTS_DIR, 'training_history.png')
    plot_training_history(
        train_losses, val_losses, train_ious, val_ious,
        train_dice, val_dice, history_path
    )
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()