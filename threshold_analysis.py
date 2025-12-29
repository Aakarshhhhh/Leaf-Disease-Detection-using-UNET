"""
Threshold Analysis Script for Plant Disease Segmentation
Tests different threshold values to find optimal Dice Score
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Add src to path
sys.path.append('src')

from src.model import get_model
from src.dataset import PlantDiseaseDataset, get_transforms
from src.utils import calculate_iou, calculate_dice_coefficient
import config


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


def create_val_dataloader(batch_size=8):
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
    
    return val_loader


def calculate_metrics_with_threshold(predictions, targets, threshold):
    """Calculate metrics for a specific threshold."""
    # Apply threshold
    pred_binary = (predictions > threshold).float()
    
    # Calculate metrics
    dice_scores = []
    iou_scores = []
    
    for i in range(predictions.size(0)):
        pred = pred_binary[i:i+1]
        target = targets[i:i+1]
        
        dice = calculate_dice_coefficient(pred, target)
        iou = calculate_iou(pred, target)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    return np.mean(dice_scores), np.mean(iou_scores)


def analyze_class_distribution(val_loader, device):
    """Analyze the class distribution in validation set."""
    total_pixels = 0
    disease_pixels = 0
    
    print("🔍 Analyzing class distribution in validation set...")
    
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Analyzing"):
            masks = masks.to(device)
            
            total_pixels += masks.numel()
            disease_pixels += masks.sum().item()
    
    healthy_pixels = total_pixels - disease_pixels
    disease_ratio = disease_pixels / total_pixels
    healthy_ratio = healthy_pixels / total_pixels
    
    print(f"📊 Class Distribution Analysis:")
    print(f"   🟢 Healthy pixels: {healthy_pixels:,} ({healthy_ratio:.2%})")
    print(f"   🔴 Disease pixels: {disease_pixels:,} ({disease_ratio:.2%})")
    print(f"   ⚖️  Disease/Healthy ratio: 1:{healthy_pixels/disease_pixels:.1f}")
    
    return disease_ratio, healthy_ratio


def threshold_analysis(model, val_loader, device, thresholds):
    """Perform threshold analysis on validation set."""
    print(f"\n🎯 Starting threshold analysis with {len(thresholds)} thresholds...")
    
    # Collect all predictions and targets first
    all_predictions = []
    all_targets = []
    
    print("📥 Collecting predictions...")
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc="Predicting"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get raw predictions (before sigmoid)
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"📊 Total samples for analysis: {all_predictions.size(0)}")
    
    # Test different thresholds
    results = []
    
    print("🔬 Testing thresholds...")
    for threshold in tqdm(thresholds, desc="Thresholds"):
        dice, iou = calculate_metrics_with_threshold(all_predictions, all_targets, threshold)
        results.append({
            'threshold': threshold,
            'dice_score': dice,
            'iou_score': iou
        })
    
    return results


def plot_threshold_analysis(results, save_path):
    """Plot threshold analysis results."""
    thresholds = [r['threshold'] for r in results]
    dice_scores = [r['dice_score'] for r in results]
    iou_scores = [r['iou_score'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Dice scores
    ax1.plot(thresholds, dice_scores, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Dice Score vs Threshold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Dice Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(dice_scores) * 1.1)
    
    # Highlight best threshold for Dice
    best_dice_idx = np.argmax(dice_scores)
    best_dice_threshold = thresholds[best_dice_idx]
    best_dice_score = dice_scores[best_dice_idx]
    
    ax1.axvline(x=best_dice_threshold, color='red', linestyle='--', alpha=0.7)
    ax1.annotate(f'Best: {best_dice_threshold:.2f}\nDice: {best_dice_score:.3f}', 
                xy=(best_dice_threshold, best_dice_score),
                xytext=(best_dice_threshold + 0.05, best_dice_score),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot IoU scores
    ax2.plot(thresholds, iou_scores, 'g-o', linewidth=2, markersize=6)
    ax2.set_title('IoU Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('IoU Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(iou_scores) * 1.1)
    
    # Highlight best threshold for IoU
    best_iou_idx = np.argmax(iou_scores)
    best_iou_threshold = thresholds[best_iou_idx]
    best_iou_score = iou_scores[best_iou_idx]
    
    ax2.axvline(x=best_iou_threshold, color='red', linestyle='--', alpha=0.7)
    ax2.annotate(f'Best: {best_iou_threshold:.2f}\nIoU: {best_iou_score:.3f}', 
                xy=(best_iou_threshold, best_iou_score),
                xytext=(best_iou_threshold + 0.05, best_iou_score),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Threshold analysis plot saved to: {save_path}")


def print_threshold_results(results):
    """Print detailed threshold analysis results."""
    print("\n" + "="*80)
    print("🎯 THRESHOLD ANALYSIS RESULTS")
    print("="*80)
    
    # Find best thresholds
    best_dice_result = max(results, key=lambda x: x['dice_score'])
    best_iou_result = max(results, key=lambda x: x['iou_score'])
    
    print(f"🏆 BEST DICE SCORE:")
    print(f"   Threshold: {best_dice_result['threshold']:.2f}")
    print(f"   Dice Score: {best_dice_result['dice_score']:.4f} ({best_dice_result['dice_score']*100:.2f}%)")
    print(f"   IoU Score: {best_dice_result['iou_score']:.4f} ({best_dice_result['iou_score']*100:.2f}%)")
    
    print(f"\n🎯 BEST IoU SCORE:")
    print(f"   Threshold: {best_iou_result['threshold']:.2f}")
    print(f"   Dice Score: {best_iou_result['dice_score']:.4f} ({best_iou_result['dice_score']*100:.2f}%)")
    print(f"   IoU Score: {best_iou_result['iou_score']:.4f} ({best_iou_result['iou_score']*100:.2f}%)")
    
    # Compare with default threshold (0.5)
    default_result = next((r for r in results if r['threshold'] == 0.5), None)
    if default_result:
        dice_improvement = best_dice_result['dice_score'] - default_result['dice_score']
        iou_improvement = best_dice_result['iou_score'] - default_result['iou_score']
        
        print(f"\n📊 IMPROVEMENT OVER DEFAULT THRESHOLD (0.5):")
        print(f"   Default Dice: {default_result['dice_score']:.4f}")
        print(f"   Best Dice: {best_dice_result['dice_score']:.4f}")
        print(f"   Dice Improvement: +{dice_improvement:.4f} ({dice_improvement/default_result['dice_score']*100:+.1f}%)")
        print(f"   IoU Improvement: +{iou_improvement:.4f} ({iou_improvement/default_result['iou_score']*100:+.1f}%)")
    
    print("\n📋 DETAILED RESULTS:")
    print("-"*80)
    print(f"{'Threshold':<12} {'Dice Score':<12} {'IoU Score':<12} {'Dice %':<10} {'IoU %':<10}")
    print("-"*80)
    
    for result in results:
        threshold = result['threshold']
        dice = result['dice_score']
        iou = result['iou_score']
        
        # Highlight best results
        marker = "🏆" if result == best_dice_result else "🎯" if result == best_iou_result else "  "
        
        print(f"{marker} {threshold:<10.2f} {dice:<12.4f} {iou:<12.4f} {dice*100:<10.2f} {iou*100:<10.2f}")
    
    print("="*80)
    
    return best_dice_result, best_iou_result


def save_results_to_json(results, best_dice_result, best_iou_result, disease_ratio, save_path):
    """Save threshold analysis results to JSON."""
    output_data = {
        'analysis_summary': {
            'best_dice_threshold': best_dice_result['threshold'],
            'best_dice_score': best_dice_result['dice_score'],
            'best_iou_threshold': best_iou_result['threshold'],
            'best_iou_score': best_iou_result['iou_score'],
            'disease_pixel_ratio': disease_ratio
        },
        'detailed_results': results,
        'recommendations': {
            'optimal_threshold_for_dice': best_dice_result['threshold'],
            'optimal_threshold_for_iou': best_iou_result['threshold'],
            'improvement_over_default': {
                'dice_improvement': best_dice_result['dice_score'] - next((r['dice_score'] for r in results if r['threshold'] == 0.5), 0),
                'iou_improvement': best_dice_result['iou_score'] - next((r['iou_score'] for r in results if r['threshold'] == 0.5), 0)
            }
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: {save_path}")


def main():
    """Main threshold analysis function."""
    print("🎯 Starting Threshold Analysis for Plant Disease Segmentation")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create outputs directory
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    
    # Load model
    model = load_best_model(device)
    if model is None:
        return
    
    # Create validation dataloader
    val_loader = create_val_dataloader(batch_size=8)
    
    # Analyze class distribution
    disease_ratio, healthy_ratio = analyze_class_distribution(val_loader, device)
    
    # Define threshold range (0.1 to 0.5 with step 0.05)
    thresholds = np.arange(0.10, 0.55, 0.05).round(2).tolist()
    print(f"\n🔬 Testing thresholds: {thresholds}")
    
    # Perform threshold analysis
    results = threshold_analysis(model, val_loader, device, thresholds)
    
    # Print results
    best_dice_result, best_iou_result = print_threshold_results(results)
    
    # Plot results
    plot_path = os.path.join(config.OUTPUTS_DIR, 'threshold_analysis.png')
    plot_threshold_analysis(results, plot_path)
    
    # Save results to JSON
    json_path = os.path.join(config.OUTPUTS_DIR, 'threshold_analysis_results.json')
    save_results_to_json(results, best_dice_result, best_iou_result, disease_ratio, json_path)
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🎯 Use threshold {best_dice_result['threshold']:.2f} for best Dice Score")
    print(f"   🎯 Use threshold {best_iou_result['threshold']:.2f} for best IoU Score")
    
    if best_dice_result['threshold'] < 0.5:
        print(f"   ✅ Lower threshold ({best_dice_result['threshold']:.2f}) helps detect disease better!")
        print(f"   📈 This confirms your hypothesis about class imbalance")
    
    print(f"\n✅ Threshold analysis completed successfully!")
    print(f"📁 Results saved in: {config.OUTPUTS_DIR}")


if __name__ == '__main__':
    main()