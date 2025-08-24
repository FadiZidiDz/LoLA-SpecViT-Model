#!/usr/bin/env python3
"""
Quick Start Script for Hyperspectral Image Classification
========================================================

This script demonstrates the main functionalities of the repository:
1. Model creation and parameter analysis
2. Data loading and preprocessing
3. Training configuration
4. Ablation study execution

Usage:
    python quick_start.py --mode [demo|train|ablation]
"""

import argparse
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'torch', 'transformers', 'peft', 'numpy', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def demo_model_creation():
    """Demonstrate model creation and parameter analysis."""
    print("\n🔧 Model Creation Demo")
    print("=" * 50)
    
    try:
        from improved_GCPE_old import create_enhanced_model
        
        # Create model with default settings
        model, efficiency_results = create_enhanced_model(
            spatial_size=15,
            num_classes=9,  # LongKou dataset
            lora_rank=16,
            lora_alpha=32,
            freeze_non_lora=True
        )
        
        print(f"✅ Model created successfully!")
        print(f"📊 Total Parameters: {efficiency_results['total_params']:,}")
        print(f"🎯 Trainable Parameters: {efficiency_results['trainable_params']:,}")
        print(f"📉 Parameter Reduction: {efficiency_results['parameter_reduction_percent']:.2f}%")
        print(f"💾 Memory Usage: {efficiency_results['memory_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    return True

def demo_data_loading():
    """Demonstrate data loading and preprocessing."""
    print("\n📊 Data Loading Demo")
    print("=" * 50)
    
    try:
        # Check if data directory exists
        data_dir = Path("data")
        if not data_dir.exists():
            print("⚠️  Data directory not found. Please ensure datasets are in the 'data/' folder.")
            print("   Expected structure:")
            print("   data/")
            print("   ├── WHU-Hi-LongKou/")
            print("   ├── Salinas/")
            print("   └── WHU-Hi-HongHu/")
            return False
        
        # List available datasets
        available_datasets = []
        for item in data_dir.iterdir():
            if item.is_dir():
                available_datasets.append(item.name)
        
        if available_datasets:
            print(f"✅ Found datasets: {', '.join(available_datasets)}")
            print("📁 You can now run training with:")
            print("   python enhanced_training.py --dataset LongKou --skip-pretrained")
        else:
            print("⚠️  No dataset folders found in 'data/' directory")
            return False
            
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False
    
    return True

def demo_training_config():
    """Show training configuration options."""
    print("\n⚙️  Training Configuration")
    print("=" * 50)
    
    print("🚀 Quick Training Commands:")
    print()
    print("1. Train with pretrained GCViT (if available):")
    print("   python enhanced_training.py --dataset LongKou --hf_backbone nvidia/GCViT")
    print()
    print("2. Train from scratch (no pretrained model):")
    print("   python enhanced_training.py --dataset LongKou --skip-pretrained")
    print()
    print("3. Custom LoRA configuration:")
    print("   python enhanced_training.py --dataset LongKou --lora_rank 32 --lora_alpha 64")
    print()
    print("4. Ablation studies:")
    print("   python lora_ablation_study_real_data.py --dataset Salinas")
    print("   python gate_residual_ablation_real_data.py --dataset LongKou")
    
    return True

def run_ablation_demo():
    """Run a quick ablation study demo."""
    print("\n🔬 Ablation Study Demo")
    print("=" * 50)
    
    try:
        # Check if ablation script exists
        if not Path("lora_ablation_study_real_data.py").exists():
            print("❌ Ablation script not found")
            return False
        
        print("✅ Ablation script found!")
        print("📋 To run ablation study:")
        print("   python lora_ablation_study_real_data.py --dataset LongKou")
        print()
        print("📊 This will test:")
        print("   - Spectral Only LoRA")
        print("   - + Attention LoRA")
        print("   - + MLPs LoRA")
        print("   - All Components LoRA")
        
    except Exception as e:
        print(f"❌ Error in ablation demo: {e}")
        return False
    
    return True

def main():
    """Main function to run the quick start demo."""
    parser = argparse.ArgumentParser(description="Quick Start for Hyperspectral Classification")
    parser.add_argument("--mode", choices=["demo", "train", "ablation"], 
                       default="demo", help="Mode to run")
    
    args = parser.parse_args()
    
    print("🚀 Hyperspectral Image Classification - Quick Start")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    if args.mode == "demo":
        # Run all demos
        demo_model_creation()
        demo_data_loading()
        demo_training_config()
        run_ablation_demo()
        
    elif args.mode == "train":
        # Show training options
        demo_training_config()
        
    elif args.mode == "ablation":
        # Show ablation options
        run_ablation_demo()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Ensure your datasets are in the 'data/' folder")
    print("2. Run training: python enhanced_training.py --dataset LongKou --skip-pretrained")
    print("3. Check results in enhanced_plots/ and enhanced_cls_result/ folders")
    print("4. Run ablation studies for comprehensive analysis")
    print("\n📚 For detailed usage, see README.md")
    print("🔧 For issues, check the troubleshooting section")

if __name__ == "__main__":
    main()
