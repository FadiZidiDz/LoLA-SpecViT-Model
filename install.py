#!/usr/bin/env python3
"""
Installation Script for Hyperspectral Image Classification
========================================================

This script helps set up the environment and check all dependencies.
It will:
1. Check Python version
2. Verify CUDA availability
3. Install required packages
4. Test basic functionality

Usage:
    python install.py [--check-only|--install]
"""

import argparse
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_cuda():
    """Check CUDA availability."""
    print("\n🔧 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
            print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - training will be slow on CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'numpy': 'NumPy',
        'scikit-learn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing_packages.append(package)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages."""
    print(f"\n📥 Installing {len(packages)} missing packages...")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def test_imports():
    """Test if all imports work correctly."""
    print("\n🧪 Testing imports...")
    
    try:
        import torch
        import transformers
        import peft
        import numpy as np
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        
        print("✅ All core packages imported successfully")
        
        # Test basic functionality
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("✅ Basic PyTorch operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist."""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "data",
        "enhanced_peft_checkpoints",
        "enhanced_plots",
        "enhanced_cls_result",
        "classification_maps",
        "ablation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    print("📁 All directories created")

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Installation script for Hyperspectral Classification")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies, don't install")
    parser.add_argument("--install", action="store_true", help="Install missing packages")
    
    args = parser.parse_args()
    
    print("🚀 Hyperspectral Image Classification - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Check packages
    missing_packages = check_packages()
    
    if missing_packages:
        print(f"\n⚠️  {len(missing_packages)} packages missing: {', '.join(missing_packages)}")
        
        if args.install:
            if install_packages(missing_packages):
                print("✅ All packages installed successfully")
            else:
                print("❌ Some packages failed to install")
                sys.exit(1)
        elif not args.check_only:
            print("\n💡 To install missing packages, run:")
            print("   python install.py --install")
            print("   or")
            print("   pip install -r requirements.txt")
    else:
        print("✅ All required packages are installed")
    
    # Test imports
    if test_imports():
        print("✅ All functionality tests passed")
    else:
        print("❌ Some functionality tests failed")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    
    if cuda_available:
        print("🚀 GPU training ready - you can start training immediately")
    else:
        print("⚠️  CPU training only - consider installing CUDA for faster training")
    
    print("\n🎯 Next steps:")
    print("1. Place your datasets in the 'data/' folder")
    print("2. Run: python quick_start.py --mode demo")
    print("3. Start training: python enhanced_training.py --dataset LongKou --skip-pretrained")
    print("\n📚 For detailed usage, see README.md")

if __name__ == "__main__":
    main()
