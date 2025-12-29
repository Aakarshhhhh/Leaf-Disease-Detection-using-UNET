# Plant Disease Detection - Clean Project Structure

## 📁 **Project Organization**

```
Plant Disease Detection Using UNET/
├── 📂 data/                          # Dataset
│   ├── 📂 Train/                     # Training data (2,998 augmented images)
│   │   ├── 📂 Images/               # Training images
│   │   └── 📂 Masks/                # Training masks
│   └── 📂 Val/                      # Validation data (90 images)
│       ├── 📂 Images/               # Validation images
│       └── 📂 Masks/                # Validation masks
│
├── 📂 src/                          # Core source code
│   ├── 📄 model.py                  # U-Net model architecture
│   ├── 📄 dataset.py                # Dataset loading and transforms
│   ├── 📄 train.py                  # Training utilities
│   ├── 📄 inference.py              # Inference utilities
│   └── 📄 utils.py                  # Loss functions and metrics
│
├── 📂 scripts/                      # Utility scripts
│   ├── 📄 evaluate_model.py         # Model evaluation
│   └── 📄 prepare_data.py           # Data preparation
│
├── 📂 models/                       # Saved models
│   └── 📂 checkpoints/              # Model checkpoints
│       ├── 📄 best_model.pth        # Original best model
│       ├── 📄 final_model.pth       # Original final model
│       └── 📄 best_model_augmented.pth  # New augmented model (training)
│
├── 📂 outputs/                      # Results and visualizations
│   ├── 📄 training_history.json     # Training metrics
│   ├── 📄 detailed_results.txt      # Evaluation results
│   └── 📄 *.png                     # Visualization images
│
├── 📄 train_augmented.py            # 🚀 Main GPU training script
├── 📄 demo.py                       # Demo inference script
├── 📄 quick_eval.py                 # Quick evaluation script
├── 📄 check_training.py             # Training status checker
├── 📄 config.py                     # Configuration settings
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Project documentation
├── 📄 PROJECT_SUMMARY.md            # Detailed project summary
├── 📄 FINAL_EVALUATION_REPORT.md    # Final results report
└── 📄 PROJECT_STRUCTURE.md          # This file
```

## 🚀 **Key Scripts**

### **Training**
- `train_augmented.py` - GPU-optimized training with augmented dataset (2,998 images)

### **Evaluation & Inference**
- `demo.py` - Run inference on sample images with visualizations
- `quick_eval.py` - Quick evaluation on validation set
- `scripts/evaluate_model.py` - Comprehensive model evaluation

### **Utilities**
- `check_training.py` - Monitor training progress
- `config.py` - Central configuration file

### **Documentation**
- `README.md` - Project overview and setup instructions
- `PROJECT_SUMMARY.md` - Detailed project documentation
- `FINAL_EVALUATION_REPORT.md` - Complete evaluation results

## 🎯 **Current Status**
- ✅ **Dataset**: 2,998 augmented training images + 90 validation images
- ✅ **GPU Training**: Active on RTX 3050 with CUDA acceleration
- ✅ **Model**: U-Net architecture (13.4M parameters)
- ✅ **Performance**: Previous model achieved 71.91% Dice score
- 🔄 **Training**: New augmented model in progress (30 epochs)

## 📊 **Usage Examples**

### Start Training
```bash
python train_augmented.py --epochs 30 --batch_size 8
```

### Check Training Status
```bash
python check_training.py
```

### Run Evaluation
```bash
python quick_eval.py
```

### Run Demo
```bash
python demo.py
```

---
*Clean project structure - All unnecessary files removed*
*GPU training active with augmented dataset*