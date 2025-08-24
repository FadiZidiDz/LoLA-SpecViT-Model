# LoLA-SpecViT: Local Attention SwiGLU Vision Transformer with LoRA for Hyperspectral Imaging


A state-of-the-art hyperspectral image classification framework that implements a **custom-built Global Context Vision Transformer (GCViT) architecture** with **enhanced Low-Rank Adaptation (LoRA)** for efficient and accurate classification of hyperspectral data.

## 🚀 Features

- **Custom GCViT Architecture**: Fully custom-built transformer backbone
- **Enhanced LoRA Integration**: Custom LoRA implementation with gate and residual mechanisms
- **Multi-Dataset Support**: LongKou, Salinas, HongHu, and  QUH-Qingyun hyperspectral datasets
- **Efficient Training**: Parameter-efficient fine-tuning with up to 71% parameter reduction
- **Comprehensive Ablation Studies**: Systematic analysis of LoRA placement and components
- **Professional Evaluation**: OA, AA, Kappa metrics with detailed performance analysis

## 📁 Repository Structure

```
cls_SSFTT_IP/
├── enhanced_training.py          # Main training script with PEFT integration
├── improved_GCPE.py         # Core model architecture (Enhanced LoRA + modified GCViT)
├── gate_residual_ablation_real_data.py  # Gate/Residual ablation study
├── lora_ablation_study_real_data.py     # LoRA placement ablation study
├── get_cls_map.py               # Classification map generation
├── data/                        # Dataset storage
│   ├── WHU-Hi-LongKou/         # LongKou dataset
│   ├── Salinas/                 # Salinas dataset
│   └── WHU-Hi-HongHu/          # HongHu dataset
├── enhanced_peft_checkpoints/   # PEFT LoRA checkpoints
├── enhanced_plots/              # Training curves and results
├── enhanced_cls_result/         # Classification results
├── classification_maps/         # Generated classification maps
└── ablation_results/            # Ablation study results
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory recommended

### Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd cls_SSFTT_IP

# Create conda environment
conda create -n hsi_classification python=3.8
conda activate hsi_classification

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install transformers peft wandb scikit-learn matplotlib seaborn tqdm
```

## 🚀 Quick Start

### Option 1: Training with Optional Pretrained Weight Initialization (Recommended)

**Note**: The model architecture is **fully custom-built**. Pretrained weights are only used for initialization if available:

```bash
# Train with pretrained weight initialization (if available)
python enhanced_training.py --dataset LongKou --hf_backbone nvidia/GCViT

# Train with pretrained weight initialization (if available)
python enhanced_training.py --dataset Salinas --hf_backbone nvidia/GCViT

# Train with pretrained weight initialization (if available)
python enhanced_training.py --dataset HongHu --hf_backbone nvidia/GCViT
```

### Option 2: Training from Scratch 

**The model is designed to work without pretrained weights**:

```bash
# Train on LongKou dataset (from scratch - recommended)
python enhanced_training.py --dataset LongKou --skip-pretrained

# Train on Salinas dataset (from scratch - recommended)
python enhanced_training.py --dataset Salinas --skip-pretrained

# Train on HongHu dataset (from scratch - recommended)
python enhanced_training.py --dataset HongHu --skip-pretrained
```

## 📊 Model Architecture

### Custom-Built GCViT Architecture
- **Spectral Processing**: 3D convolution with BandDropout and Adaptive SE
- **Custom Transformer Backbone**: Fully custom-built GCViT architecture (not pretrained)
- **Enhanced LoRA Integration**: Custom LoRA implementation throughout the architecture
- **Classification Head**: Efficient classification with LoRA-adapted layers

### Enhanced LoRA Implementation
- **Custom LoRA**: `EnhancedLoRALinear` with gate and residual mechanisms
- **Rank**: 16 (configurable)
- **Alpha**: 32 (configurable)
- **Targets**: Attention layers, MLPs, and spectral processing
- **Parameter Reduction**: Up to 71% compared to full fine-tuning
- **Gate & Residual**: Optional sigmoid gating and residual connections

## 🔬 Ablation Studies

### LoRA Placement Analysis
```bash
# Run LoRA placement ablation study
python lora_ablation_study_real_data.py --dataset Salinas
```

**Configurations:**
- Spectral Only: LoRA in spectral processing only
- + Attention: Add LoRA to attention layers  
- + MLPs: Add LoRA to MLP layers
- All Components: Full LoRA integration

### Enhanced LoRA Component Analysis
```bash
# Run gate/residual ablation study
python gate_residual_ablation_real_data.py --dataset LongKou
```

**Variants:**
- Base LoRA: Standard LoRA implementation (`output = main + lora`)
- + Residual only: Add residual connections (`output = main + lora + residual`)
- + Gate only: Add sigmoid gating (`output = main + gate * lora`)
- + Gate + Residual: Full enhanced LoRA (`output = main + gate * lora + residual`)

## 📈 Training Configuration

### Key Parameters
```python
config = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'lora_rank': 16,
    'lora_alpha': 32,
    'warmup_epochs': 10,
    'gradient_accumulation_steps': 1,
    'use_amp': True,  # Mixed precision training
    'grad_clip': 1.0,
    'use_wandb': False  # Set to True for logging
}
```

### Dataset Splits
- **Training**: 2% of total samples (for low-label regime)
- **Testing**: 98% of total samples
- **Patch Size**: 15×15×15 (spatial × spectral)

## 📊 Results and Performance

### Classification Metrics
- **Overall Accuracy (OA)**: Primary performance measure
- **Average Accuracy (AA)**: Mean per-class accuracy
- **Kappa Coefficient**: Agreement measure

### Parameter Efficiency
- **Total Parameters**: ~254M (custom-built model)
- **Trainable Parameters**: ~74M (LoRA only)
- **Parameter Reduction**: 71.10%
- **Memory Usage**: ~971MB (training)
- **Architecture**: Fully custom-built, no pretrained backbone required

## 🎯 Usage Examples

### Generate Classification Maps
```bash
# Generate classification map for LongKou
python get_cls_map.py --dataset LongKou --model_path enhanced_peft_checkpoints/model_best.pth
```

### Custom Training Configuration
```bash
# Custom LoRA configuration
python enhanced_training.py \
    --dataset LongKou \
    --lora_rank 32 \
    --lora_alpha 64 \
    --batch_size 64 \
    --num_epochs 200
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python enhanced_training.py --batch_size 16

# Enable gradient accumulation
python enhanced_training.py --gradient_accumulation_steps 2
```

**2. Model Architecture Understanding**
- **This is NOT a pretrained GCViT model** - it's a custom-built architecture
- **Pretrained weights are optional** - only used for initialization if available
- **Use `--skip-pretrained`** for training the custom model from scratch

**3. Dataset Loading Issues**
- Ensure datasets are in the `data/` folder
- Check file permissions and paths
- Verify dataset format compatibility

### Performance Optimization
- **GPU Training**: Ensure CUDA is properly installed
- **Mixed Precision**: Enable AMP for faster training
- **Memory Management**: Use gradient accumulation for large models

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Enhanced LoRA for Hyperspectral Image Classification with GCViT},
  author={Your Name},
  journal={Pattern Recognition},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GCViT architecture design inspired by NVIDIA's work
- LoRA implementation inspired by Microsoft's PEFT library
- Hyperspectral datasets from WHU-Hi and Salinas
- **Note**: This implementation uses a fully custom-built architecture, not pretrained models

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the code comments for implementation details

---

**Note**: This framework is designed for research and educational purposes. For production use, please ensure proper validation and testing on your specific datasets.





