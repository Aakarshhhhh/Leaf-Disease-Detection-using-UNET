"""
Final evaluation script for the trained model.
Performs:
1. Plot training curves (Loss and Dice Score)
2. Visual comparison on 5 random validation images
3. Metrics summary on validation set
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from PIL import Image
import json
import glob
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from src.model import get_model
from src.dataset import PlantDiseaseDataset, get_transforms
from src.utils import calculate_iou, calculate_dice_coefficient
import config


def load_training_history():
    """Load training history from JSON files."""
    # Try to find training history files
    history_files = glob.glob(os.path.join(config.OUTPUTS_DIR, '*history*.json'))
    
    if not history_files:
        print("WARNING: No training history found. Will skip curve plotting.")
        return None
    
    # Load the most recent history file
    latest_history = max(history_files, key=os.path.getctime)
    print(f"Loading training history from: {latest_history}")
    
    with open(latest_history, 'r') as f:
        history = json.load(f)
    
    return history


def plot_training_curves(history, save_path):
    """Plot training and validation curves."""
    if history is None:
        print("❌ Cannot plot curves - no training history available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(len(history['train_losses']))
    
    # Plot Loss curves
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Dice Score curves
    ax2.plot(epochs, history['train_dice'], 'b-', label='Training Dice', linewidth=2)
    ax2.plot(epochs, history['val_dice'], 'r-', label='Validation Dice', linewidth=2)
    ax2.set_title('Training vs Validation Dice Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


def load_best_model(device):
    """Load the best trained model."""
    model_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model_augmented.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Best model not found at: {model_path}")
        return None
    
    model = get_model(n_channels=3, n_classes=config.NUM_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded best model from: {model_path}")
    return model


def create_val_dataloader(batch_size=1):
    """Create validation dataloader."""
    val_transform = get_transforms(config.IMAGE_SIZE, is_training=False)
    
    val_dataset = PlantDiseaseDataset(
        config.VAL_IMAGES_DIR, 
        config.VAL_MASKS_DIR, 
        config.IMAGE_SIZE, 
        val_transform, 
        False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return val_loader, val_dataset


def denormalize_image(tensor):
    """Denormalize image tensor for visualization."""
    # Assuming ImageNet normalization was used
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)
    return denorm


def create_visual_comparison(model, val_dataset, device, save_path, num_samples=5):
    """Create visual comparison of 5 random validation samples."""
    # Select random samples
    random_indices = random.sample(range(len(val_dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(random_indices):
        # Get sample
        image, mask, _ = val_dataset[idx]
        
        # Add batch dimension and move to device
        image_batch = image.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(image_batch)
            pred = torch.sigmoid(pred)
            pred_mask = (pred > 0.5).float()
        
        # Convert to numpy for visualization
        image_np = denormalize_image(image).permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        pred_np = pred_mask.squeeze().cpu().numpy()
        
        # Plot original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Sample {idx}: Original Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title('Model Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️  Visual comparison saved to: {save_path}")


def calculate_pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy."""
    pred_binary = (pred > 0.5).float()
    correct = (pred_binary == target).float()
    accuracy = correct.mean().item()
    return accuracy


def evaluate_model_metrics(model, val_loader, device):
    """Evaluate model and calculate comprehensive metrics."""
    model.eval()
    
    all_accuracies = []
    all_ious = []
    all_dice_scores = []
    
    print("🔍 Evaluating model on validation set...")
    
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Calculate metrics for each sample in batch
            for i in range(images.size(0)):
                pred = predictions[i:i+1]
                mask = masks[i:i+1]
                
                # Pixel accuracy
                accuracy = calculate_pixel_accuracy(pred, mask)
                all_accuracies.append(accuracy)
                
                # IoU
                iou = calculate_iou(pred, mask)
                all_ious.append(iou)
                
                # Dice score
                dice = calculate_dice_coefficient(pred, mask)
                all_dice_scores.append(dice)
    
    # Calculate mean metrics
    mean_accuracy = np.mean(all_accuracies)
    mean_iou = np.mean(all_ious)
    mean_dice = np.mean(all_dice_scores)
    
    return mean_accuracy, mean_iou, mean_dice


def print_metrics_summary(accuracy, iou, dice):
    """Print a formatted metrics summary table."""
    print("\n" + "="*60)
    print("🏆 FINAL MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Model: best_model_augmented.pth")
    print(f"📁 Dataset: Validation Set ({config.VAL_IMAGES_DIR})")
    print("-"*60)
    print(f"{'Metric':<20} {'Value':<15} {'Percentage':<15}")
    print("-"*60)
    print(f"{'Overall Accuracy':<20} {accuracy:.4f}{'':>10} {accuracy*100:.2f}%")
    print(f"{'Mean IoU':<20} {iou:.4f}{'':>10} {iou*100:.2f}%")
    print(f"{'Dice Score':<20} {dice:.4f}{'':>10} {dice*100:.2f}%")
    print("="*60)
    
    # Performance assessment
    if dice > 0.8:
        print("🎉 EXCELLENT: Model performance is outstanding!")
    elif dice > 0.7:
        print("✅ VERY GOOD: Model performance is very good!")
    elif dice > 0.6:
        print("👍 GOOD: Model performance is good!")
    else:
        print("📈 NEEDS IMPROVEMENT: Consider more training or data augmentation.")
    
    print("="*60 + "\n")


def main():
    """Main evaluation function."""
    print("🚀 Starting Final Model Evaluation")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create outputs directory if it doesn't exist
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    
    # Task 1: Plot training curves
    print("\n📈 Task 1: Plotting Training Curves")
    history = load_training_history()
    curves_path = os.path.join(config.OUTPUTS_DIR, 'training_curves.png')
    plot_training_curves(history, curves_path)
    
    # Task 2: Visual comparison
    print("\n🖼️  Task 2: Creating Visual Comparison")
    model = load_best_model(device)
    
    if model is not None:
        val_loader, val_dataset = create_val_dataloader(batch_size=1)
        comparison_path = os.path.join(config.OUTPUTS_DIR, 'results_samples.png')
        create_visual_comparison(model, val_dataset, device, comparison_path)
        
        # Task 3: Metrics summary
        print("\n📊 Task 3: Calculating Validation Metrics")
        val_loader_batch, _ = create_val_dataloader(batch_size=8)  # Larger batch for efficiency
        accuracy, iou, dice = evaluate_model_metrics(model, val_loader_batch, device)
        print_metrics_summary(accuracy, iou, dice)
        
        # Save metrics to file
        metrics_data = {
            'overall_accuracy': float(accuracy),
            'mean_iou': float(iou),
            'dice_score': float(dice),
            'model_path': 'models/checkpoints/best_model_augmented.pth',
            'evaluation_date': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
        }
        
        metrics_path = os.path.join(config.OUTPUTS_DIR, 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Metrics saved to: {metrics_path}")
    
    print("\n✅ Final evaluation completed successfully!")
    print(f"📁 All results saved in: {config.OUTPUTS_DIR}")


if __name__ == '__main__':
    # Set random seed for reproducible results
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()