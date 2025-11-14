# ODIR-5K Dual CNN - Ocular Disease Recognition

A deep learning model for multi-label classification of ocular diseases using dual-eye fundus images from the ODIR-5K dataset.

## Overview

This project implements a sophisticated dual-eye CNN architecture that processes both left and right eye images simultaneously, using cross-attention mechanisms to capture inter-eye relationships for improved disease classification.

## Dataset

**ODIR-5K (Ocular Disease Intelligent Recognition)**
- Source: [Kaggle - ODIR-5K Dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- Contains paired fundus images (left and right eyes)
- Multi-label classification across 8 disease categories

### Disease Classes
- **N**: Normal
- **D**: Diabetes
- **G**: Glaucoma
- **C**: Cataract
- **A**: Age-related Macular Degeneration
- **H**: Hypertension
- **M**: Myopia
- **O**: Other diseases/abnormalities

## Model Architecture

### Dual CNN with Cross-Attention

The model features:
- **Shared Backbone**: ConvNeXt-Small pretrained encoder for feature extraction
- **Individual Eye Branches**: Separate processing paths for left and right eyes
- **Cross-Attention Mechanism**: Multi-head attention allowing each eye to attend to the other
- **Fusion Head**: Deep classifier combining all eye representations

```
Left Eye  ──► Backbone ──► Left Branch  ──┐
                                           ├──► Cross-Attention ──► Fusion Head ──► Predictions
Right Eye ──► Backbone ──► Right Branch ──┘
```

### Key Features
- **Image Size**: 384×384 pixels
- **Model Parameters**: ~27M parameters
- **Architecture**: ConvNeXt-Small backbone with custom dual-eye head
- **Attention**: 8-head cross-attention between eye representations

## Training Configuration

### Hyperparameters
- **Batch Size**: 6 (with gradient accumulation of 2 steps)
- **Epochs**: 25
- **Learning Rate**: 5e-5 (with differential learning rates for different layers)
- **Optimizer**: AdamW with weight decay 2e-5
- **Scheduler**: Cosine annealing with 3-epoch warmup

### Data Augmentation
Native PyTorch transforms including:
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Color jitter - brightness and contrast (60%)
- Random rotation ±12° (60%)

### Loss Function
Hybrid loss combining:
- Weighted Binary Cross-Entropy (50%)
- Focal Loss with α=0.3, γ=2.5 (50%)

### Class Balancing
- Weighted random sampling based on label frequency
- Class-specific positive weights for BCE loss
- Focal loss to handle class imbalance

## Requirements

```bash
pip install torch torchvision timm openpyxl pillow scikit-learn pandas numpy tqdm
```

### Dependencies
- PyTorch (CUDA support recommended)
- timm (PyTorch Image Models)
- scikit-learn
- pandas, numpy
- Pillow
- openpyxl (for Excel file reading)

## Usage

### 1. Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("andrewmvd/ocular-disease-recognition-odir5k")
```

### 2. Update Paths
Modify the `BASE` variable in the script to point to your dataset location:
```python
BASE = "/path/to/ODIR-5K/ODIR-5K"
```

### 3. Train Model
```bash
python best_model.py
```

The script will:
- Load and prepare the dual-eye dataset
- Split into train (85%) and validation (15%) sets
- Train for 25 epochs with early stopping based on F1-score
- Save the best model as `best_dual_f1_model.pth`

## Performance Metrics

The model is evaluated using:
- **F1-micro**: Overall performance across all classes
- **F1-macro**: Average F1 across individual classes
- **Per-class F1**: Individual performance for each disease category

Training outputs per-class F1 scores for detailed performance analysis.

## Model Optimization Techniques

1. **Gradient Accumulation**: Effective batch size of 12 (6×2)
2. **Mixed Precision Training**: AMP for faster training and reduced memory
3. **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
4. **Differential Learning Rates**: Lower LR for backbone (0.1×), higher for head
5. **Learning Rate Warmup**: 3 epochs linear warmup
6. **Cosine Annealing**: Smooth LR decay over training

## Advanced Features

### Cross-Attention Between Eyes
The model learns to correlate information between left and right eyes, which is clinically relevant as many ocular diseases affect both eyes with correlated patterns.

### Weighted Sampling
Training uses weighted random sampling to ensure balanced exposure to rare disease classes during training.

### Hybrid Loss Function
Combines BCE for stable gradients with Focal Loss for hard example mining, addressing class imbalance effectively.

## Output

The training loop displays:
- Real-time training loss per batch
- Per-epoch train and validation F1 scores
- Per-class F1 scores for each disease category
- Best model checkpoint saved when validation F1 improves

## GPU Recommendations

- **Minimum**: 8GB VRAM (may require batch size reduction)
- **Recommended**: 16GB+ VRAM for optimal training speed

## Future Improvements

- [ ] Test-time augmentation (TTA)
- [ ] Ensemble with different backbones
- [ ] Attention visualization
- [ ] External validation on other datasets
- [ ] Clinical deployment pipeline

## License

This implementation is for research and educational purposes. Please cite the original ODIR-5K dataset appropriately in any publications.

## Citation

If you use this code, please cite the ODIR-5K dataset:
```
@dataset{odir5k,
  title={Ocular Disease Intelligent Recognition (ODIR-5K)},
  author={Peking University},
  year={2019}
}
```

## Acknowledgments

- ODIR-5K dataset providers
- PyTorch and timm library contributors
- ConvNeXt architecture developers
