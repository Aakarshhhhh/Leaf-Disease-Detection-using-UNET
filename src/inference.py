"""Inference script for plant disease segmentation."""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.dataset import get_transforms
from src.utils import create_overlay_visualization, save_prediction_overlay
import config


class PlantDiseasePredictor:
    """Plant disease prediction class."""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = get_model(n_channels=3, n_classes=config.NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms(config.IMAGE_SIZE, is_training=False)
        
        print(f'Model loaded from {model_path}')
        print(f'Using device: {self.device}')
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference."""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size for later
        original_size = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image, original_size
    
    def predict(self, image_path):
        """Predict disease mask for a single image."""
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(image_tensor)
            prediction = torch.sigmoid(prediction)
        
        # Convert to numpy and resize to original size
        pred_mask = prediction.cpu().numpy().squeeze()
        pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))
        
        return pred_mask, original_image
    
    def predict_batch(self, image_paths):
        """Predict disease masks for multiple images."""
        results = []
        
        for image_path in image_paths:
            try:
                pred_mask, original_image = self.predict(image_path)
                results.append({
                    'image_path': image_path,
                    'prediction': pred_mask,
                    'original_image': original_image,
                    'success': True
                })
            except Exception as e:
                print(f'Error processing {image_path}: {str(e)}')
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None, 
                           alpha=0.3, color=(255, 0, 0), threshold=0.5):
        """Visualize prediction with overlay."""
        pred_mask, original_image = self.predict(image_path)
        
        # Apply threshold
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        
        # Create overlay
        overlay = create_overlay_visualization(
            original_image, binary_mask, alpha, color
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction heatmap
        axes[1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Disease Probability')
        axes[1].axis('off')
        
        # Binary mask
        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title(f'Binary Mask (threshold={threshold})')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(overlay)
        axes[3].set_title('Disease Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Visualization saved to {save_path}')
        
        plt.show()
        
        return overlay, pred_mask, binary_mask
    
    def calculate_disease_severity(self, pred_mask, threshold=0.5):
        """Calculate disease severity metrics."""
        binary_mask = pred_mask > threshold
        
        total_pixels = pred_mask.size
        diseased_pixels = binary_mask.sum()
        
        severity_percentage = (diseased_pixels / total_pixels) * 100
        avg_confidence = pred_mask[binary_mask].mean() if diseased_pixels > 0 else 0
        
        return {
            'total_pixels': total_pixels,
            'diseased_pixels': int(diseased_pixels),
            'severity_percentage': severity_percentage,
            'average_confidence': avg_confidence,
            'severity_level': self._get_severity_level(severity_percentage)
        }
    
    def _get_severity_level(self, percentage):
        """Get severity level based on percentage."""
        if percentage < 1:
            return 'Healthy'
        elif percentage < 5:
            return 'Mild'
        elif percentage < 15:
            return 'Moderate'
        elif percentage < 30:
            return 'Severe'
        else:
            return 'Critical'


def main():
    parser = argparse.ArgumentParser(description='Plant disease inference')
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(config.CHECKPOINTS_DIR, 'best_model.pth'),
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default=config.PREDICTIONS_DIR,
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Overlay transparency')
    parser.add_argument('--color', type=str, default='255,0,0',
                       help='Overlay color (R,G,B)')
    args = parser.parse_args()
    
    # Parse color
    color = tuple(map(int, args.color.split(',')))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = PlantDiseasePredictor(args.model_path)
    
    # Get image name for output files
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Make prediction and visualization
    print(f'Processing image: {args.image_path}')
    
    viz_path = os.path.join(args.output_dir, f'{image_name}_visualization.png')
    overlay, pred_mask, binary_mask = predictor.visualize_prediction(
        args.image_path, viz_path, args.alpha, color, args.threshold
    )
    
    # Save individual outputs
    overlay_path = os.path.join(args.output_dir, f'{image_name}_overlay.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    heatmap_path = os.path.join(args.output_dir, f'{image_name}_heatmap.png')
    plt.imsave(heatmap_path, pred_mask, cmap='hot', vmin=0, vmax=1)
    
    mask_path = os.path.join(args.output_dir, f'{image_name}_mask.png')
    cv2.imwrite(mask_path, (binary_mask * 255).astype(np.uint8))
    
    # Calculate disease severity
    severity = predictor.calculate_disease_severity(pred_mask, args.threshold)
    
    print('\n--- Disease Analysis Results ---')
    print(f'Image: {os.path.basename(args.image_path)}')
    print(f'Total pixels: {severity["total_pixels"]:,}')
    print(f'Diseased pixels: {severity["diseased_pixels"]:,}')
    print(f'Disease coverage: {severity["severity_percentage"]:.2f}%')
    print(f'Average confidence: {severity["average_confidence"]:.3f}')
    print(f'Severity level: {severity["severity_level"]}')
    
    print(f'\nOutputs saved to:')
    print(f'  Visualization: {viz_path}')
    print(f'  Overlay: {overlay_path}')
    print(f'  Heatmap: {heatmap_path}')
    print(f'  Binary mask: {mask_path}')


if __name__ == '__main__':
    main()