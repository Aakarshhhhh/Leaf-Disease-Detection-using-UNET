# Plant Disease Detection & Localization Pipeline

A deep learning pipeline for automated detection and localization of diseases in plant leaves using semantic segmentation with U-Net architecture.

## Project Structure

```
plant-disease-segmentation/
├── data/
│   ├── images/          # RGB leaf images
│   └── masks/           # Binary disease masks
├── models/
│   └── checkpoints/     # Saved model weights
├── outputs/
│   └── predictions/     # Generated visualizations
├── src/
│   ├── model.py         # U-Net architecture
│   ├── train.py         # Training pipeline
│   ├── inference.py     # Prediction and visualization
│   ├── utils.py         # Data utilities and augmentation
│   └── dataset.py       # Data loading pipeline
├── requirements.txt
└── config.py           # Configuration parameters
```

## Features

- **Semantic Segmentation**: Precise pixel-level disease localization
- **U-Net Architecture**: Skip connections for detailed spatial reconstruction
- **Advanced Data Augmentation**: Albumentations for robust training
- **Combined Loss Function**: BCE + Dice Loss for class imbalance handling
- **Comprehensive Metrics**: mIoU and Dice Coefficient monitoring
- **Visualization Tools**: Disease overlay generation for farmers

## Quick Start

### Option 1: Test with Sample Data
```bash
# Install dependencies
pip install -r requirements.txt

# Run quick start script (creates sample data and tests pipeline)
python quick_start.py

# Train on sample data
python src/train.py

# Test inference
python src/inference.py --image_path data/images/sample_001.png
```

### Option 2: Use Your Own Data
```bash
# Install dependencies
pip install -r requirements.txt

# Prepare your data in the correct structure
# data/images/    - RGB leaf images (.jpg, .png)
# data/masks/     - Binary disease masks (.png)

# Validate your dataset
python scripts/prepare_data.py --action validate

# Train the model
python src/train.py

# Run inference
python src/inference.py --image_path path/to/your/leaf.jpg

# Evaluate model performance
python scripts/evaluate_model.py
```

## Dataset Requirements

- **Images**: RGB leaf images in JPG or PNG format
- **Masks**: Binary masks where white pixels (255) represent diseased areas and black pixels (0) represent healthy tissue
- **Naming**: Image and mask files should have the same base name (e.g., `leaf001.jpg` and `leaf001.png`)
- **Size**: Images will be automatically resized to 256×256 during training

## Training Configuration

Key parameters can be modified in `config.py`:
- `IMAGE_SIZE`: Input image dimensions (default: 256×256)
- `BATCH_SIZE`: Training batch size (default: 16)
- `LEARNING_RATE`: Initial learning rate (default: 1e-4)
- `NUM_EPOCHS`: Number of training epochs (default: 100)
- `BCE_WEIGHT` / `DICE_WEIGHT`: Loss function weights (default: 0.5 each)