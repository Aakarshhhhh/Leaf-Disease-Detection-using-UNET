"""Script to prepare and validate dataset for training."""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def validate_dataset(images_dir, masks_dir):
    """Validate that images and masks are properly paired."""
    
    image_files = set()
    mask_files = set()
    
    # Get image files
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.update([f.stem for f in Path(images_dir).glob(f'*{ext}')])
    
    # Get mask files
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_files.update([f.stem for f in Path(masks_dir).glob(f'*{ext}')])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Check for missing pairs
    missing_masks = image_files - mask_files
    missing_images = mask_files - image_files
    
    if missing_masks:
        print(f"Images without masks: {missing_masks}")
    
    if missing_images:
        print(f"Masks without images: {missing_images}")
    
    valid_pairs = image_files & mask_files
    print(f"Valid image-mask pairs: {len(valid_pairs)}")
    
    return valid_pairs


def check_image_properties(images_dir, masks_dir, sample_size=10):
    """Check properties of sample images and masks."""
    
    valid_pairs = validate_dataset(images_dir, masks_dir)
    
    if not valid_pairs:
        print("No valid pairs found!")
        return
    
    sample_pairs = list(valid_pairs)[:sample_size]
    
    print(f"\nChecking properties of {len(sample_pairs)} sample pairs:")
    
    for name in sample_pairs:
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = Path(images_dir) / f"{name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        # Find mask file
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = Path(masks_dir) / f"{name}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if image_path and mask_path:
            # Load and check image
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            print(f"\n{name}:")
            print(f"  Image: {image.shape} - {image_path.suffix}")
            print(f"  Mask:  {mask.shape} - {mask_path.suffix}")
            print(f"  Mask values: min={mask.min()}, max={mask.max()}")
            
            # Check if mask is binary
            unique_values = np.unique(mask)
            if len(unique_values) <= 2:
                print(f"  Mask appears binary: {unique_values}")
            else:
                print(f"  Mask has multiple values: {unique_values}")


def resize_dataset(images_dir, masks_dir, output_images_dir, output_masks_dir, 
                  target_size=(256, 256)):
    """Resize all images and masks to target size."""
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    valid_pairs = validate_dataset(images_dir, masks_dir)
    
    print(f"\nResizing {len(valid_pairs)} pairs to {target_size}...")
    
    for name in valid_pairs:
        # Find and load image
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = Path(images_dir) / f"{name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        # Find and load mask
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = Path(masks_dir) / f"{name}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if image_path and mask_path:
            # Load images
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Resize
            image_resized = cv2.resize(image, target_size)
            mask_resized = cv2.resize(mask, target_size)
            
            # Save resized images
            output_image_path = Path(output_images_dir) / f"{name}.png"
            output_mask_path = Path(output_masks_dir) / f"{name}.png"
            
            cv2.imwrite(str(output_image_path), image_resized)
            cv2.imwrite(str(output_mask_path), mask_resized)
    
    print(f"Resized dataset saved to {output_images_dir} and {output_masks_dir}")


def create_sample_data(output_images_dir, output_masks_dir, num_samples=10):
    """Create sample synthetic data for testing."""
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    print(f"Creating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        # Create synthetic leaf image (green with some texture)
        image = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        image[:, :, 1] = np.random.randint(100, 200, (256, 256))  # More green
        
        # Add some leaf-like patterns
        center_x, center_y = 128, 128
        y, x = np.ogrid[:256, :256]
        leaf_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < 100 ** 2
        image[~leaf_mask] = [50, 80, 50]  # Background
        
        # Create synthetic disease spots
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Add random disease spots
        num_spots = np.random.randint(1, 5)
        for _ in range(num_spots):
            spot_x = np.random.randint(50, 206)
            spot_y = np.random.randint(50, 206)
            spot_size = np.random.randint(10, 30)
            
            y, x = np.ogrid[:256, :256]
            spot_mask = ((x - spot_x) ** 2 + (y - spot_y) ** 2) < spot_size ** 2
            mask[spot_mask] = 255
            
            # Make disease spots brown/yellow in image
            image[spot_mask] = [40, 100, 150]  # Brown-ish color
        
        # Save sample
        image_path = Path(output_images_dir) / f"sample_{i:03d}.png"
        mask_path = Path(output_masks_dir) / f"sample_{i:03d}.png"
        
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)
    
    print(f"Sample data created in {output_images_dir} and {output_masks_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--images_dir', type=str, default=config.IMAGES_DIR,
                       help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, default=config.MASKS_DIR,
                       help='Path to masks directory')
    parser.add_argument('--action', type=str, choices=['validate', 'resize', 'sample'],
                       default='validate', help='Action to perform')
    parser.add_argument('--output_images_dir', type=str,
                       help='Output directory for resized images')
    parser.add_argument('--output_masks_dir', type=str,
                       help='Output directory for resized masks')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='Target size for resizing (width height)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of synthetic samples to create')
    
    args = parser.parse_args()
    
    if args.action == 'validate':
        check_image_properties(args.images_dir, args.masks_dir)
    
    elif args.action == 'resize':
        if not args.output_images_dir or not args.output_masks_dir:
            print("Please specify --output_images_dir and --output_masks_dir for resize action")
            return
        
        resize_dataset(
            args.images_dir, args.masks_dir,
            args.output_images_dir, args.output_masks_dir,
            tuple(args.target_size)
        )
    
    elif args.action == 'sample':
        create_sample_data(args.images_dir, args.masks_dir, args.num_samples)


if __name__ == '__main__':
    main()