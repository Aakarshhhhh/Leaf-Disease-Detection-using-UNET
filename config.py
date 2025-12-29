"""Configuration parameters for the plant disease segmentation pipeline."""

import os
import torch

# Data paths
DATA_DIR = "data"
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "Train", "Images")
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, "Train", "Masks")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, "Val", "Images")
VAL_MASKS_DIR = os.path.join(DATA_DIR, "Val", "Masks")
# Legacy paths for compatibility
IMAGES_DIR = TRAIN_IMAGES_DIR
MASKS_DIR = TRAIN_MASKS_DIR
MODELS_DIR = "models"
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
OUTPUTS_DIR = "outputs"
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

# Model parameters
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 1  # Binary segmentation (disease vs healthy)
ENCODER_NAME = "resnet34"  # For segmentation_models_pytorch
ENCODER_WEIGHTS = "imagenet"

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Loss function weights
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# Augmentation parameters
ROTATION_LIMIT = 30
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2

# Checkpoint and logging
SAVE_EVERY = 10  # Save checkpoint every N epochs
LOG_EVERY = 10   # Log metrics every N batches

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)