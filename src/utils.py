"""Utility functions for plant disease segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Flatten tensors
        predictions = torch.sigmoid(predictions).view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice_score


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_iou(predictions, targets, threshold=0.5):
    """Calculate Intersection over Union (IoU)."""
    predictions = torch.sigmoid(predictions) > threshold
    targets = targets > threshold
    
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    if targets.sum() == 0 and predictions.sum() == 0:
        return 1.0
    
    return jaccard_score(targets, predictions, average='binary')


def calculate_dice_coefficient(predictions, targets, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient."""
    predictions = torch.sigmoid(predictions) > threshold
    targets = targets > threshold
    
    predictions = predictions.float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )
    
    return dice.item()


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.ious = []
        self.dice_scores = []
    
    def update(self, loss, iou, dice):
        self.losses.append(loss)
        self.ious.append(iou)
        self.dice_scores.append(dice)
    
    def get_averages(self):
        return {
            'loss': np.mean(self.losses),
            'iou': np.mean(self.ious),
            'dice': np.mean(self.dice_scores)
        }


def visualize_predictions(images, masks, predictions, save_path=None, num_samples=4):
    """Visualize original images, ground truth masks, and predictions."""
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 10))
    
    for i in range(min(num_samples, len(images))):
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title('Original Image')
        axes[0, i].axis('off')
        
        # Ground truth mask
        mask = masks[i].cpu().numpy().squeeze()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Prediction
        pred = torch.sigmoid(predictions[i]).cpu().numpy().squeeze()
        axes[2, i].imshow(pred, cmap='gray')
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_overlay_visualization(image, mask, alpha=0.3, color=(255, 0, 0)):
    """Create overlay visualization of disease areas on original image."""
    
    # Ensure image is in correct format
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # CHW format
            image = image.transpose(1, 2, 0)
    
    # Denormalize if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Ensure mask is in correct format
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    
    # Threshold mask if it's not binary
    if mask.max() > 1.0:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = (mask > 0.5).astype(np.uint8)
    
    # Create colored overlay
    overlay = image.copy()
    overlay[mask == 1] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def save_prediction_overlay(image_path, prediction, output_path, 
                          alpha=0.3, color=(255, 0, 0)):
    """Save prediction overlay for a single image."""
    
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to match prediction
    if isinstance(prediction, torch.Tensor):
        pred_size = prediction.shape[-2:]
        image = cv2.resize(image, (pred_size[1], pred_size[0]))
        prediction = torch.sigmoid(prediction).cpu().numpy().squeeze()
    
    # Create overlay
    overlay = create_overlay_visualization(image, prediction, alpha, color)
    
    # Save result
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay_bgr)
    
    return overlay


def plot_training_history(train_losses, val_losses, train_ious, val_ious, 
                         train_dice, val_dice, save_path=None):
    """Plot training history."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    # IoU
    axes[1].plot(train_ious, label='Train IoU')
    axes[1].plot(val_ious, label='Val IoU')
    axes[1].set_title('IoU Score')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    
    # Dice
    axes[2].plot(train_dice, label='Train Dice')
    axes[2].plot(val_dice, label='Val Dice')
    axes[2].set_title('Dice Coefficient')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test loss functions
    predictions = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
    # Test combined loss
    loss_fn = CombinedLoss()
    loss = loss_fn(predictions, targets)
    print(f"Combined loss: {loss.item():.4f}")
    
    # Test metrics
    iou = calculate_iou(predictions, targets)
    dice = calculate_dice_coefficient(predictions, targets)
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")