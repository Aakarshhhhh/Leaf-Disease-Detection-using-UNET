"""Script to evaluate trained model on test dataset."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.dataset import PlantDiseaseDataset, get_transforms
from src.utils import calculate_iou, calculate_dice_coefficient, MetricsTracker
import config


def evaluate_model(model_path, images_dir, masks_dir, output_dir):
    """Evaluate model on test dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(n_channels=3, n_classes=config.NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dataset
    transform = get_transforms(config.IMAGE_SIZE, is_training=False)
    dataset = PlantDiseaseDataset(images_dir, masks_dir, config.IMAGE_SIZE, transform, False)
    
    if len(dataset) == 0:
        print("No images found in dataset!")
        return
    
    print(f"Evaluating on {len(dataset)} images...")
    
    # Metrics tracking
    metrics = MetricsTracker()
    all_predictions = []
    all_targets = []
    image_results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            image, mask, img_name = dataset[i]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Predict
            prediction = model(image)
            
            # Calculate metrics
            iou = calculate_iou(prediction, mask)
            dice = calculate_dice_coefficient(prediction, mask)
            
            metrics.update(0, iou, dice)  # Loss not needed for evaluation
            
            # Store for confusion matrix
            pred_binary = (torch.sigmoid(prediction) > 0.5).cpu().numpy().flatten()
            target_binary = (mask > 0.5).cpu().numpy().flatten()
            
            all_predictions.extend(pred_binary)
            all_targets.extend(target_binary)
            
            # Store individual results
            image_results.append({
                'name': img_name,
                'iou': iou,
                'dice': dice,
                'prediction': torch.sigmoid(prediction).cpu().numpy().squeeze(),
                'target': mask.cpu().numpy().squeeze()
            })
    
    # Calculate overall metrics
    overall_metrics = metrics.get_averages()
    
    print("\n=== Evaluation Results ===")
    print(f"Mean IoU: {overall_metrics['iou']:.4f}")
    print(f"Mean Dice: {overall_metrics['dice']:.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_targets, all_predictions, 
                              target_names=['Healthy', 'Diseased']))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Diseased'],
                yticklabels=['Healthy', 'Diseased'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot metrics distribution
    ious = [result['iou'] for result in image_results]
    dices = [result['dice'] for result in image_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(ious, bins=20, alpha=0.7, color='blue')
    axes[0].set_title('IoU Score Distribution')
    axes[0].set_xlabel('IoU Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(ious), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(ious):.3f}')
    axes[0].legend()
    
    axes[1].hist(dices, bins=20, alpha=0.7, color='green')
    axes[1].set_title('Dice Score Distribution')
    axes[1].set_xlabel('Dice Score')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(dices), color='red', linestyle='--',
                   label=f'Mean: {np.mean(dices):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    dist_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Show best and worst predictions
    image_results.sort(key=lambda x: x['iou'], reverse=True)
    
    # Best predictions
    print(f"\n=== Top 5 Best Predictions ===")
    for i, result in enumerate(image_results[:5]):
        print(f"{i+1}. {result['name']}: IoU={result['iou']:.4f}, Dice={result['dice']:.4f}")
    
    # Worst predictions
    print(f"\n=== Top 5 Worst Predictions ===")
    for i, result in enumerate(image_results[-5:]):
        print(f"{i+1}. {result['name']}: IoU={result['iou']:.4f}, Dice={result['dice']:.4f}")
    
    # Visualize some examples
    visualize_examples(image_results, images_dir, output_dir)
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'detailed_results.txt')
    with open(results_path, 'w') as f:
        f.write("=== Model Evaluation Results ===\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {images_dir}\n")
        f.write(f"Total images: {len(dataset)}\n\n")
        
        f.write(f"Mean IoU: {overall_metrics['iou']:.4f}\n")
        f.write(f"Mean Dice: {overall_metrics['dice']:.4f}\n\n")
        
        f.write("Individual Results:\n")
        for result in image_results:
            f.write(f"{result['name']}: IoU={result['iou']:.4f}, Dice={result['dice']:.4f}\n")
    
    print(f"\nDetailed results saved to {results_path}")


def visualize_examples(image_results, images_dir, output_dir, num_examples=6):
    """Visualize best and worst prediction examples."""
    
    # Get best and worst examples
    best_examples = image_results[:num_examples//2]
    worst_examples = image_results[-(num_examples//2):]
    
    examples = best_examples + worst_examples
    
    fig, axes = plt.subplots(3, num_examples, figsize=(20, 12))
    
    for i, result in enumerate(examples):
        # Load original image
        img_path = os.path.join(images_dir, result['name'])
        image = plt.imread(img_path)
        
        # Original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"{result['name']}\nIoU: {result['iou']:.3f}")
        axes[0, i].axis('off')
        
        # Ground truth
        axes[1, i].imshow(result['target'], cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')
        
        # Prediction
        axes[2, i].imshow(result['prediction'], cmap='gray')
        axes[2, i].set_title('Prediction')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    examples_path = os.path.join(output_dir, 'prediction_examples.png')
    plt.savefig(examples_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(config.CHECKPOINTS_DIR, 'best_model.pth'),
                       help='Path to trained model')
    parser.add_argument('--images_dir', type=str, default=config.IMAGES_DIR,
                       help='Path to test images directory')
    parser.add_argument('--masks_dir', type=str, default=config.MASKS_DIR,
                       help='Path to test masks directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUTS_DIR,
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_model(args.model_path, args.images_dir, args.masks_dir, args.output_dir)


if __name__ == '__main__':
    main()