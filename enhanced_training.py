import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from operator import truediv
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_class_weight
from get_cls_map import get_cls_map, classification_map, list_to_colormap, get_classification_map
import wandb
from improved_GCPE_old import (
    EnhancedPEFTHyperspectralGCViT,
    analyze_model_efficiency,
    create_enhanced_model,
    load_hf_gcvit_into_model,
    prepare_model_for_lora_finetuning,
    merge_lora_for_inference,
)
import os
import timm
from torchvision import transforms
import warnings
import argparse
warnings.filterwarnings("ignore")

# Set environment variable for wandb
os.environ["WANDB_API_KEY"] = "b68375e4a0cbcbc284700f0627c966e5d78181b6"

# ADDING MISSING DATA PROCESSING FUNCTIONS FROM ORIGINAL
def loadData(dataset_name="LongKou"):
    """
    Load hyperspectral data and labels with explicit dataset selection
    
    Args:
        dataset_name (str): Dataset to load. Options:
            - "LongKou": WHU-Hi-LongKou dataset (9 classes)
            - "IndianPines": Indian Pines dataset (16 classes)
            - "PaviaU": Pavia University dataset (9 classes)
            - "PaviaC": Pavia Center dataset (9 classes)
            - "Salinas": Salinas dataset (16 classes)
            - "HongHu": WHU-Hi-HongHu dataset (22 classes)
            - "auto": Automatically detect available datasets
    
    Returns:
        tuple: (data, labels, dataset_info)
    """
    dataset_configs = {
        "LongKou": {
            "data_path": "data/WHU_Hi_LongKou.mat",
            "label_path": "data/WHU_Hi_LongKou_gt.mat",
            "data_key": "WHU_Hi_LongKou",
            "label_key": "WHU_Hi_LongKou_gt",
            "num_classes": 9,
            "target_names": ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                           'Narrow-leaf soybean', 'Rice', 'Water',
                           'Roads and houses', 'Mixed weed'],
            "description": "WHU-Hi-LongKou dataset with 9 crop classes"
        },
        "IndianPines": {
            "data_path": "Indian_pines_corrected.mat",
            "label_path": "Indian_pines_gt.mat",
            "data_key": "indian_pines_corrected",
            "label_key": "indian_pines_gt",
            "num_classes": 16,
            "target_names": ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                           'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                           'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                           'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                           'Stone-Steel-Towers'],
            "description": "Indian Pines dataset with 16 land cover classes"
        },
        "PaviaU": {
            "data_path": "PaviaU.mat",
            "label_path": "PaviaU_gt.mat",
            "data_key": "paviaU",
            "label_key": "paviaU_gt",
            "num_classes": 9,
            "target_names": ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                           'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows'],
            "description": "Pavia University dataset with 9 urban classes"
        },
        "PaviaC": {
            "data_path": "PaviaC.mat",
            "label_path": "PaviaC_gt.mat",
            "data_key": "paviaC",
            "label_key": "paviaC_gt",
            "num_classes": 9,
            "target_names": ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen',
                           'Tiles', 'Shadows', 'Meadows', 'Bare Soil'],
            "description": "Pavia Center dataset with 9 urban classes"
        },
        "Salinas": {
            "data_path": "data/Salinas_corrected.mat",
            "label_path": "data/Salinas_gt.mat",
            "data_key": "salinas_corrected",
            "label_key": "salinas_gt",
            "num_classes": 16,
            "target_names": ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                           'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
                           'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                           'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                           'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'],
            "description": "Salinas dataset with 16 agricultural classes"
        },
        "HongHu": {
            "data_path": "data/WHU_Hi_HongHu.mat",
            "label_path": "data/WHU_Hi_HongHu_gt.mat",
            "data_key": "WHU_Hi_HongHu",
            "label_key": "WHU_Hi_HongHu_gt",
            "num_classes": 22,
            "target_names": ['Red roof', 'Road', 'Bare soil', 'Red roof 2', 'Red roof 3',
                           'Gray roof', 'Red roof 4', 'White roof', 'Bright roof', 'Trees',
                           'Grass', 'Red roof 5', 'Red roof 6', 'Red roof 7', 'Red roof 8',
                           'Red roof 9', 'Red roof 10', 'Red roof 11', 'Red roof 12', 'Red roof 13',
                           'Red roof 14', 'Red roof 15'],
            "description": "WHU-Hi-HongHu dataset with 22 urban classes"
        },
        "Qingyun": {
            "data_path": "data/QUH-Qingyun.mat",
            "label_path": "data/QUH-Qingyun_GT.mat", 
            "data_key": "Chengqu",
            "label_key": "ChengquGT",
            "num_classes": 6,
            "target_names": ["Trees", "Concrete building", "Car", "Ironhide building",
                           "Plastic playground", "Asphalt road"],
            "description": "Qingyun dataset with 6 urban classes"
        }
    }
    
    if dataset_name == "auto":
        # Try to automatically detect available datasets
        available_datasets = []
        for name, config in dataset_configs.items():
            try:
                sio.loadmat(config["data_path"])
                sio.loadmat(config["label_path"])
                available_datasets.append(name)
                print(f"✓ Found {name} dataset")
            except FileNotFoundError:
                print(f"✗ {name} dataset not found")
        
        if not available_datasets:
            raise FileNotFoundError("No datasets found! Please ensure dataset files are in the correct location.")
        
        # Use the first available dataset
        dataset_name = available_datasets[0]
        print(f"Auto-selected dataset: {dataset_name}")
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available options: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    try:
        data = sio.loadmat(config["data_path"])[config["data_key"]]
        labels = sio.loadmat(config["label_path"])[config["label_key"]]
        
        print(f"✓ Successfully loaded {dataset_name} dataset")
        print(f"  - Data shape: {data.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Number of classes: {config['num_classes']}")
        print(f"  - Description: {config['description']}")
        
        return data, labels, config
        
    except FileNotFoundError as e:
        print(f"✗ Error loading {dataset_name} dataset: {e}")
        print(f"Please ensure the following files exist:")
        print(f"  - {config['data_path']}")
        print(f"  - {config['label_path']}")
        raise

def padWithZeros(X, margin=2):
    """Pad the hyperspectral data with zeros"""
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=15, removeZeroLabels=True):
    """Create image cubes for hyperspectral data processing with memory-efficient approach"""
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    
    # PROFESSIONAL FIX: Use lists to collect only valid patches, avoiding massive pre-allocation
    patches_list = []
    labels_list = []
    
    print(f"Creating patches for {X.shape[0]}x{X.shape[1]} image with {windowSize}x{windowSize} window...")
    
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            # Get the original position in the unpadded image
            orig_r = r - margin
            orig_c = c - margin
            label = y[orig_r, orig_c]
            
            # Only collect patches with non-zero labels (if removeZeroLabels=True)
            if not removeZeroLabels or label > 0:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patches_list.append(patch)
                labels_list.append(label)
    
    # Convert to numpy arrays only after filtering
    patchesData = np.array(patches_list, dtype=np.float32)
    patchesLabels = np.array(labels_list, dtype=np.int32)
    
    if removeZeroLabels:
        # Labels are already filtered, just convert to 0-based indexing
        patchesLabels -= 1
    
    # Validate label range
    min_label = np.min(patchesLabels)
    max_label = np.max(patchesLabels)
    num_classes = len(np.unique(patchesLabels))
    print(f"Created {len(patchesData)} patches")
    print(f"Label range: [{min_label}, {max_label}], Number of classes: {num_classes}")
    
    # Check for non-finite values
    assert np.isfinite(patchesData).all(), "Patch data contains non-finite values"
    
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=testRatio,
                                                       random_state=randomState,
                                                       stratify=y)
    
    # Validate split results
    assert len(np.unique(y_train)) == len(np.unique(y)), "Training set missing some classes"
    assert len(np.unique(y_test)) == len(np.unique(y)), "Test set missing some classes"
    
    return X_train, X_test, y_train, y_test

class EnhancedHyperspectralDataset:
    def __init__(self, hyperspectral_data, labels, pretrained_transform=None):
        self.hyperspectral_data = hyperspectral_data
        self.labels = labels
        self.pretrained_transform = pretrained_transform or self._get_default_transform()
        
    def _get_default_transform(self):
        """Get default transform for hyperspectral data (15 channels)"""
        # For hyperspectral data, we'll use a simple normalization
        # No need for RGB-specific transforms since we're using hyperspectral directly
        return None  # We'll handle normalization in the model forward pass
    
    def __len__(self):
        return len(self.hyperspectral_data)
    
    def __getitem__(self, idx):
        # Get hyperspectral data - ensure it's properly shaped
        hyperspectral = self.hyperspectral_data[idx]  # Shape: (H, W, 15)
        
        # Convert to tensor and transpose to (C, H, W) for PyTorch
        if len(hyperspectral.shape) == 3:
            hyperspectral = np.transpose(hyperspectral, (2, 0, 1))  # (15, H, W)
        
        # Convert hyperspectral data to RGB-like input for pretrained model
        # Select 3 representative bands from the 15 hyperspectral bands
        if hyperspectral.shape[0] == 15:  # (15, H, W)
            # Select bands that roughly correspond to RGB wavelengths
            r_band = hyperspectral[0]  # First band as red
            g_band = hyperspectral[7]  # Middle band as green  
            b_band = hyperspectral[14]  # Last band as blue
            
            # Stack to create RGB image
            rgb = np.stack([r_band, g_band, b_band], axis=0)  # (3, H, W)
            
            # Normalize to [0, 1] range
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            
            # Convert to tensor
            pretrained_input = torch.FloatTensor(rgb)
            
            # Resize from 15x15 to 224x224 for pretrained model
            pretrained_input = torch.nn.functional.interpolate(
                pretrained_input.unsqueeze(0),  # Add batch dimension
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
        else:
            # Fallback: use first 3 channels if not 15
            pretrained_input = torch.FloatTensor(hyperspectral[:3])
            
            # Resize to 224x224
            pretrained_input = torch.nn.functional.interpolate(
                pretrained_input.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Get label
        label = self.labels[idx]
        
        return torch.FloatTensor(hyperspectral), pretrained_input, torch.LongTensor([label]).squeeze()
    


def applyPCA(X, numComponents=15):
    """
    Apply PCA to reduce dimensionality to exactly 15 components for hyperspectral data
    This ensures consistent input size for the model architecture.
    
    Args:
        X: Input hyperspectral data (H, W, C) where C is original number of bands
        numComponents: Number of PCA components (fixed at 15 for model consistency)
    
    Returns:
        numpy.ndarray: PCA-transformed data with shape (H, W, 15)
    """
    print(f"\nApplying PCA to reduce from {X.shape[2]} bands to {numComponents} components...")
    
    # Reshape for PCA: (H*W, C)
    original_shape = X.shape
    newX = np.reshape(X, (-1, X.shape[2]))
    
    # Apply PCA with whitening for better performance
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    
    # Reshape back to spatial dimensions: (H, W, numComponents)
    newX = np.reshape(newX, (original_shape[0], original_shape[1], numComponents))
    
    # Calculate explained variance for information
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA completed: {numComponents} components explain {explained_variance:.3f} of total variance")
    
    return newX

def create_enhanced_data_loader(batch_size=64, test_ratio=0.90, patch_size=15, dataset_name="Qingyun"):
    """
    Create enhanced data loaders with comprehensive processing
    
    Args:
        batch_size (int): Batch size for data loaders
        test_ratio (float): Ratio of data to use for testing (0.02 = 2% test, 98% train)
        patch_size (int): Size of image patches
        dataset_name (str): Dataset to use (see loadData for options)
    
    Returns:
        tuple: (train_loader, test_loader, all_loader, y_all, pca_components, dataset_info)
    """
    # Load and validate raw data
    X, y, dataset_info = loadData(dataset_name)
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    print('Original label range:', np.min(y), 'to', np.max(y))
    print('Unique labels:', np.unique(y))
    
    # Apply PCA to reduce to exactly 15 components (standard for hyperspectral)
    X_pca = applyPCA(X, numComponents=15)
    pca_components = 15
    print('Data shape after PCA: ', X_pca.shape)
    
    # Create image cubes and validate labels
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Processed label range:', np.min(y_all), 'to', np.max(y_all))
    print('Unique processed labels:', np.unique(y_all))
    
    # Split dataset
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest shape: ', Xtest.shape)
    
    # Reshape data for processing (FIXED to match working model exactly)
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    
    X = X.transpose(0, 4, 3, 1, 2).squeeze(1)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2).squeeze(1)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2).squeeze(1)
    
    # Create enhanced datasets
    trainset = EnhancedHyperspectralDataset(Xtrain, ytrain)
    testset = EnhancedHyperspectralDataset(Xtest, ytest)
    allset = EnhancedHyperspectralDataset(X, y_all)
    
    # Create data loaders with memory-optimized settings (per MEMORY_OPTIMIZATION_CHANGES.md)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    all_loader = torch.utils.data.DataLoader(
        allset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    
    return train_loader, test_loader, all_loader, y_all, pca_components, dataset_info

# ADDING MISSING EVALUATION AND SAVING FUNCTIONS
def AA_andEachClassAccuracy(confusion_matrix):
    """Calculate average accuracy and per-class accuracy"""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test, dataset_name="Qingyun"):
    """Generate comprehensive accuracy reports with target names"""
    dataset_configs = {
        "LongKou": {
            "target_names": ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                           'Narrow-leaf soybean', 'Rice', 'Water',
                           'Roads and houses', 'Mixed weed'],
            "num_classes": 9
        },
        "IndianPines": {
            "target_names": ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                           'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                           'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                           'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                           'Stone-Steel-Towers'],
            "num_classes": 16
        },
        "PaviaU": {
            "target_names": ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                           'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows'],
            "num_classes": 9
        },
        "PaviaC": {
            "target_names": ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen',
                           'Tiles', 'Shadows', 'Meadows', 'Bare Soil'],
            "num_classes": 9
        },
        "Salinas": {
            "target_names": ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                           'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
                           'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                           'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                           'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'],
            "num_classes": 16
        },
        "HongHu": {
            "target_names": ['Red roof', 'Road', 'Bare soil', 'Red roof 2', 'Red roof 3',
                           'Gray roof', 'Red roof 4', 'White roof', 'Bright roof', 'Trees',
                           'Grass', 'Red roof 5', 'Red roof 6', 'Red roof 7', 'Red roof 8',
                           'Red roof 9', 'Red roof 10', 'Red roof 11', 'Red roof 12', 'Red roof 13',
                           'Red roof 14', 'Red roof 15'],
            "num_classes": 22
        },
        "Qingyun": {
            "target_names": ["Trees", "Concrete building", "Car", "Ironhide building",
                           "Plastic playground", "Asphalt road"],
            "num_classes": 6
            }
    }
    
    if dataset_name in dataset_configs:
        target_names = dataset_configs[dataset_name]["target_names"]
    else:
        # Fallback for unknown datasets
        target_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
    
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names, zero_division=0)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100, target_names

@torch.no_grad()
def test(device, model, test_loader):
    """Test model and get predictions"""
    model.eval()
    count = 0
    y_pred_test = 0
    y_test = 0
    
    for hyperspectral, pretrained, labels in test_loader:
        hyperspectral = hyperspectral.to(device)
        pretrained = pretrained.to(device)
        labels = labels.squeeze().to(device)
        
        # FIXED: Ensure correct tensor format for hyperspectral data
        # Input should be [B, C, H, W] where C is the number of channels (15 for hyperspectral)
        if hyperspectral.dim() == 4 and hyperspectral.shape[-1] == 15:  # [B, H, W, C]
            # [B, H, W, C] -> [B, C, H, W]
            hyperspectral = hyperspectral.permute(0, 3, 1, 2)
        
        # Normalize inputs (FIXED to match working model exactly)
        hyperspectral = (hyperspectral - hyperspectral.mean(dim=(2,3), keepdim=True)) / (hyperspectral.std(dim=(2,3), keepdim=True) + 1e-8)
        
        outputs = model(hyperspectral)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        
        if count == 0:
            y_pred_test = outputs
            y_test = labels.cpu().numpy()
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels.cpu().numpy()))
    
    return y_pred_test, y_test

def plot_enhanced_training_curves(train_losses, train_accuracies, eval_accuracies, epoch):
    """Plot enhanced training curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Enhanced Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(eval_accuracies, label='Validation Accuracy')
    plt.title('Enhanced Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('enhanced_plots', exist_ok=True)
    plt.savefig(f'enhanced_plots/enhanced_training_curves_{epoch}.png')
    plt.close()

def save_enhanced_results(model, config, best_acc, best_epoch, training_time, test_time, 
                         classification, oa, confusion, each_acc, aa, kappa, target_names,
                         pretrained_model_name, efficiency_results):
    """Save comprehensive enhanced results"""
    os.makedirs('enhanced_cls_result', exist_ok=True)
    file_name = f"enhanced_cls_result/classification_report_enhanced_{pretrained_model_name.replace('/', '_')}.txt"
    
    with open(file_name, 'w') as x_file:
        x_file.write(f'Enhanced Model Configuration:\n')
        x_file.write(f'Pretrained Model: {pretrained_model_name}\n')
        x_file.write(f'LoRA rank (r): 16\n')
        x_file.write(f'LoRA alpha: 32\n')
        x_file.write(f'Learning rate: {config["learning_rate"]}\n')
        x_file.write(f'Batch size: {config["batch_size"]}\n')
        x_file.write(f'Parameter reduction: {efficiency_results["parameter_reduction_percent"]:.2f}%\n\n')
        
        x_file.write(f'Training Time (s): {training_time:.2f}\n')
        x_file.write(f'Test Time (s): {test_time:.2f}\n')
        x_file.write(f'Best epoch: {best_epoch}\n\n')
        
        x_file.write(f'Enhanced Performance Metrics:\n')
        x_file.write(f'Overall Accuracy (%): {oa:.2f}\n')
        x_file.write(f'Average Accuracy (%): {aa:.2f}\n')
        x_file.write(f'Kappa Score (%): {kappa:.2f}\n\n')
        
        x_file.write(f'Per-Class Accuracies (%):\n')
        for name, acc in zip(target_names, each_acc):
            x_file.write(f'{name}: {acc:.2f}\n')
        x_file.write(f'\nDetailed Classification Report:\n{classification}\n')
        x_file.write(f'\nConfusion Matrix:\n{confusion}\n')
    
    print(f"\nEnhanced results saved to {file_name}")
    return file_name

def save_enhanced_model(model, config, best_acc, kappa, training_time, test_time, 
                       best_epoch, each_acc, confusion, pretrained_model_name, efficiency_results):
    """Save enhanced model with comprehensive metadata"""
    os.makedirs('enhanced_peft_checkpoints', exist_ok=True)
    
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'performance': {
            'accuracy': best_acc,
            'kappa': kappa,
            'training_time': training_time,
            'test_time': test_time,
            'best_epoch': best_epoch,
            'per_class_accuracy': each_acc.tolist(),
            'confusion_matrix': confusion.tolist()
        },
        'model_config': {
            'pretrained_model': pretrained_model_name,
            'lora_rank': 16,
            'lora_alpha': 32,
            'dim': 96,
            'depths': [3, 4, 19],
            'num_heads': [4, 8, 16],
            'window_size': [7, 7, 7]
        },
        'efficiency_results': efficiency_results
    }, f'enhanced_peft_checkpoints/enhanced_final_model_{pretrained_model_name.replace("/", "_")}.pth')
    
    print(f"Enhanced model saved with comprehensive metadata")

class EnhancedTrainer:
    def __init__(self, config):
        self.config = config
        
        # FORCE CUDA USAGE - bypass torch.cuda.is_available() check
        try:
            # Force CUDA device creation
            test_tensor = torch.zeros(1, device='cuda')
            self.device = torch.device('cuda')
            print(f"✓ CUDA forced: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Device count: {torch.cuda.device_count()}")
            torch.cuda.empty_cache()
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        except Exception as e:
            # Fallback to CPU only if CUDA completely fails
            self.device = torch.device('cpu')
            print(f"⚠️  CUDA failed, using CPU: {e}")
            print("⚠️  CPU training will be extremely slow!")
        
        # Initialize wandb (disabled by default for stability)
        if config.get('use_wandb', False):
            try:
                wandb.init(
                    project="enhanced-lola-specvit",
                    config=config,
                    name=f"enhanced_{config.get('hf_backbone','GCViT-LoRA')}_{config['dataset_name']}_{config['num_epochs']}epochs"
                )
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                print("Continuing without wandb logging...")
                config['use_wandb'] = False
        
        # Load and preprocess data
        self.load_data()
        
        # Create model
        self.create_model()
        
        # Setup training components
        self.setup_training()
        
        # Training metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.eval_accuracies = []
        self.learning_rates = [] # Added for warmup and adjustment
        
    def load_data(self):
        """Load and preprocess hyperspectral data"""
        print("Loading hyperspectral data...")
        # Use enhanced data loader with comprehensive processing - MEMORY OPTIMIZED
        self.train_loader, self.test_loader, self.all_loader, self.y_all, self.pca_components, self.dataset_info = create_enhanced_data_loader(
            batch_size=self.config['batch_size'],
            test_ratio=0.90,  # 90% for testing, 10% for training
            patch_size=15,
            dataset_name=self.config['dataset_name']
        )
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Total samples: {len(self.all_loader.dataset)}")
        print(f"PCA components: {self.pca_components}")
        print(f"Dataset: {self.config['dataset_name']} ({self.dataset_info['num_classes']} classes)")
        
    def create_model(self):
        """Create enhanced model with pretrained integration"""
        print(f"Creating enhanced model...")
        
        # Update num_classes based on dataset
        self.config['num_classes'] = self.dataset_info['num_classes']
        
        self.model, self.efficiency_results = create_enhanced_model(
            spatial_size=15,
            num_classes=self.config['num_classes'],
            lora_rank=self.config.get('lora_rank', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            freeze_non_lora=True
        )

        # Step 1: Load pretrained GCViT weights (partial, non-strict)
        if self.config.get('hf_backbone', None) and not self.config.get('skip_pretrained', False):
            print(f"Loading GCViT backbone (transformers): {self.config['hf_backbone']}")
            load_hf_gcvit_into_model(self.model, self.config['hf_backbone'])
        elif self.config.get('skip_pretrained', False):
            print("Skipping pretrained weights loading (training from scratch)")
        else:
            print("No pretrained backbone specified, training from scratch")

        # Step 2: Ensure only LoRA parameters are trainable
        prepare_model_for_lora_finetuning(self.model)
        
        self.model = self.model.to(self.device)
        
        # Print comprehensive parameter efficiency analysis (RESTORED from working version)
        print(f"\n{'='*60}")
        print(f"=== Enhanced GCViT (LoRA) Efficiency Analysis ===")
        print(f"Total Parameters: {self.efficiency_results['total_params']:,}")
        print(f"Trainable Parameters: {self.efficiency_results['trainable_params']:,}")
        print(f"LoRA Parameters: {self.efficiency_results['lora_params']:,}")
        print(f"Pretrained Parameters: {self.efficiency_results['pretrained_params']:,}")
        print(f"Parameter Reduction: {self.efficiency_results['parameter_reduction_percent']:.2f}%")
        print(f"LoRA Ratio: {self.efficiency_results['lora_ratio_percent']:.2f}%")
        print(f"Pretrained Ratio: {self.efficiency_results['pretrained_ratio_percent']:.2f}%")
        print(f"Memory Usage: {self.efficiency_results['memory_mb']:.2f} MB")
        print(f"Trainable Memory: {self.efficiency_results['trainable_memory_mb']:.2f} MB")
        print(f"Model efficiency analysis:")
        print(f"  - Total parameters: {self.efficiency_results['total_params']:,}")
        print(f"  - Trainable parameters: {self.efficiency_results['trainable_params']:,}")
        print(f"  - Parameter reduction: {self.efficiency_results['parameter_reduction_percent']:.2f}%")
        
    def setup_training(self):
        """Setup training components - OPTIMIZED for CPU/GPU compatibility"""
        # MEMORY OPTIMIZATION: Clear cache before setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory efficient settings
            torch.backends.cudnn.benchmark = False  # More memory efficient
            torch.backends.cudnn.deterministic = True
        
        # CRITICAL FIX: Disable AMP for CPU training
        if self.device.type == 'cpu':
            self.config['use_amp'] = False
            print("AMP disabled for CPU training")
        
        # Simple parameter grouping like working model (LongKu_train_fadi.py)
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora_' in name:
                    lora_params.append(param)
                else:
                    other_params.append(param)
        
        # Use working model's exact optimizer setup for stability
        self.optimizer = optim.AdamW([
            {'params': lora_params, 'lr': self.config['learning_rate']},
            {'params': other_params, 'lr': self.config['learning_rate'] * 0.1}
        ], weight_decay=self.config['weight_decay'])
        
        # Create scheduler (use standard scheduler like LongKu_train_fadi.py)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler']['T_0'],
            T_mult=self.config['scheduler']['T_mult'],
            eta_min=self.config['scheduler']['eta_min']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
        
        # Gradient scaler for mixed precision (only for GPU)
        self.scaler = GradScaler() if (self.config['use_amp'] and self.device.type == 'cuda') else None
        
        # Training metrics with best model state tracking
        self.best_accuracy = 0
        self.patience_counter = 0
        self.best_epoch = 0
        self.best_model_state = None  # Track best model state for restoration
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Learning rate warmup with stability (like in LongKu_train_fadi.py)
        if epoch < self.config.get('warmup_epochs', 10):
            warmup_factor = (epoch + 1) / self.config.get('warmup_epochs', 10)
            current_lr = self.config['learning_rate'] * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, data in enumerate(pbar):
            # Handle different data loader outputs
            if len(data) == 3:
                hyperspectral, pretrained, labels = data
                pretrained = pretrained.to(self.device)
            elif len(data) == 2:
                hyperspectral, labels = data
            else:
                raise ValueError(f"Unexpected data format: {len(data)} items")
            
            hyperspectral = hyperspectral.to(self.device)
            labels = labels.to(self.device)
            
            # DEBUG: Check tensor shapes
            if batch_idx == 0:
                print(f"DEBUG - hyperspectral shape: {hyperspectral.shape}, labels shape: {labels.shape}")
            
            # FIXED: Ensure correct tensor format for hyperspectral data
            if hyperspectral.dim() == 4 and hyperspectral.shape[-1] == 15:  # [B, H, W, C]
                hyperspectral = hyperspectral.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # CRITICAL FIX: Ensure labels are not empty and have correct shape
            if labels.numel() == 0:
                print(f"ERROR: Empty labels at batch {batch_idx}, skipping...")
                continue
                
            # Ensure labels are 1D for cross_entropy
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            # Enhanced normalization during training (like working model)
            hyperspectral = (hyperspectral - hyperspectral.mean(dim=(2,3), keepdim=True)) / (hyperspectral.std(dim=(2,3), keepdim=True) + 1e-8)
            
            # Zero gradients only at the start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # CRITICAL FIX: Handle AMP properly for CPU/GPU
            if self.config['use_amp'] and self.device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = self.model(hyperspectral)
                    loss = self.criterion(outputs, labels)
                    loss = loss / accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.config.get('grad_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # CPU training or no AMP
                outputs = self.model(hyperspectral)
                loss = self.criterion(outputs, labels)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                    self.optimizer.step()
            
            # Update scheduler (like in LongKu_train_fadi.py)
            if epoch >= self.config.get('warmup_epochs', 10):
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Use actual loss for metrics (undo accumulation scaling)
            actual_loss = loss.item() * accumulation_steps if accumulation_steps > 1 else loss.item()
            total_loss += actual_loss
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb (disabled by default for memory)
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': actual_loss,
                    'train_accuracy': 100.*correct/total,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # MEMORY OPTIMIZATION: Clear cache and delete tensors
            if batch_idx % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Delete intermediate variables to free memory
            del predicted
            if 'actual_loss' in locals():
                del actual_loss
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # Store metrics
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """Evaluate model on the full test set (uses larger eval batch size)."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_all = []
        eval_batch_size = self.config.get('eval_batch_size', 64)
        
        # CRITICAL FIX: Reduce batch size for CPU training
        if self.device.type == 'cpu':
            eval_batch_size = min(eval_batch_size, 16)  # Smaller batches for CPU
            print(f"CPU training detected, reducing eval batch size to {eval_batch_size}")
        
        # FULL EVALUATION: entire test set
        eval_loader = torch.utils.data.DataLoader(
            self.test_loader.dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        print(f"Full evaluation: {len(self.test_loader.dataset)} samples with batch size: {eval_batch_size}")
        
        with torch.no_grad():
            for hyperspectral, pretrained, labels in tqdm(eval_loader, desc='Evaluating'):
                hyperspectral = hyperspectral.to(self.device, non_blocking=True)
                pretrained = pretrained.to(self.device, non_blocking=True)
                labels = labels.squeeze().to(self.device, non_blocking=True)
                
                if hyperspectral.dim() == 4 and hyperspectral.shape[-1] == 15:  # [B, H, W, C]
                    hyperspectral = hyperspectral.permute(0, 3, 1, 2)  # [B, C, H, W]
                
                hyperspectral = (hyperspectral - hyperspectral.mean(dim=(2,3), keepdim=True)) / (hyperspectral.std(dim=(2,3), keepdim=True) + 1e-8)
                outputs = self.model(hyperspectral)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())
                targets_all.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        targets_all = np.array(targets_all)
        accuracy = accuracy_score(targets_all, predictions) * 100
        kappa = cohen_kappa_score(targets_all, predictions) * 100
        self.eval_accuracies.append(accuracy)
        return total_loss / len(eval_loader), accuracy, kappa, predictions, targets_all
    
    def train(self):
        """Main training loop"""
        print("Starting enhanced training...")
        
        # Initialize wandb (like in LongKu_train_fadi.py)
        if self.config.get('use_wandb', False):
            try:
                wandb.init(
                    project="enhanced-lola-specvit",
                    config=self.config,
                    name=f"enhanced_{self.config['pretrained_model_name']}_{self.config['dataset_name']}_{self.config['num_epochs']}epochs"
                )
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                print("Continuing without wandb logging...")
                self.config['use_wandb'] = False
        
        # Training time measurement
        tic1 = time.perf_counter()
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate per eval_interval for speed
            should_eval = ((epoch + 1) % self.config.get('eval_interval', 1) == 0) or (epoch + 1 == self.config['num_epochs'])
            if should_eval:
                print(f"\nEvaluating epoch {epoch+1}...")
                test_loss, test_acc, kappa, predictions, labels = self.evaluate()
            else:
                test_loss, test_acc, kappa = None, None, None
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            if should_eval:
                print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Kappa: {kappa:.4f}')
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    **({'test_loss': test_loss, 'test_accuracy': test_acc, 'kappa': kappa} if should_eval else {}),
                    'epoch': epoch
                })
            
            # Save best model with stability monitoring
            if should_eval and test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model state for potential restoration
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model with full state (like in LongKu_train_fadi.py)
                checkpoint_data = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'accuracy': test_acc,
                    'config': self.config
                }
                
                # Only save scheduler if it has state_dict method
                if hasattr(self.scheduler, 'state_dict'):
                    checkpoint_data['scheduler'] = self.scheduler.state_dict()
                
                # Save best model
                best_model_path = f'best_enhanced_model_{self.config["dataset_name"]}.pth'
                torch.save(checkpoint_data, best_model_path)
                print(f'Best model saved: {best_model_path}')
            elif should_eval:
                self.patience_counter += 1
                
                # If accuracy drops significantly, restore best model
                if test_acc < self.best_accuracy - 2.0:  # 2% drop threshold
                    print(f"Accuracy dropped significantly from {self.best_accuracy:.2f}% to {test_acc:.2f}%, restoring best model from epoch {self.best_epoch}")
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        print("Best model state restored")
                
            # Early stopping
            if should_eval and self.patience_counter >= self.config['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        toc1 = time.perf_counter()
        training_time = toc1 - tic1
        
        print(f"Training completed. Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        
        # Finish wandb (like in LongKu_train_fadi.py)
        if self.config.get('use_wandb', False):
            wandb.finish()
        
        # Load best model for final evaluation
        try:
            print(f"Loading best model from: best_enhanced_model_{self.config['dataset_name']}.pth")
            checkpoint = torch.load(f'best_enhanced_model_{self.config["dataset_name"]}.pth')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                print("Loading full checkpoint with state...")
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                print("Loading state dict only...")
                self.model.load_state_dict(checkpoint)
            print("Successfully loaded best model for final evaluation")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Attempting to continue with current model state...")
        self.model.eval()
        print("Model set to evaluation mode for final testing")

        # Final evaluation (always on best model)
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(self.device, self.model, self.test_loader)
        toc2 = time.perf_counter()
        test_time = toc2 - tic2

        # Calculate comprehensive metrics
        classification, oa, confusion, each_acc, aa, kappa, target_names = acc_reports(
            y_test, y_pred_test, dataset_name=self.config['dataset_name']
        )

        # Save results (always for best model)
        file_name = save_enhanced_results(
            self.model, self.config, self.best_accuracy, self.best_epoch, 
            training_time, test_time, classification, oa, confusion, 
            each_acc, aa, kappa, target_names, 
            'GCViT-LoRA', self.efficiency_results
        )

        # Save enhanced model (best model)
        save_enhanced_model(
            self.model, self.config, self.best_accuracy, kappa, 
            training_time, test_time, self.best_epoch, each_acc, confusion,
            'GCViT-LoRA', self.efficiency_results
        )

        # Optional Step 3: Merge LoRA into base for inference
        if self.config.get('merge_lora_for_inference', False):
            print("Merging LoRA adapters into base linear layers for inference...")
            merge_lora_for_inference(self.model)
            print("LoRA merged. You can now save a merged checkpoint if desired.")
        
        # Plot training curves
        plot_enhanced_training_curves(self.train_losses, self.train_accuracies, self.eval_accuracies, epoch)
        
        # Generate classification maps
        print('\nGenerating enhanced classification maps...')
        try:
            cls_labels = get_cls_map(self.model, self.device, self.all_loader, self.y_all)
            print('Enhanced classification maps have been saved to the classification_maps directory')
        except Exception as e:
            print(f"Error generating classification maps: {e}")
        
        # Display comprehensive final results
        print(f"\n{'='*60}")
        print(f"FINAL TRAINING RESULTS - {self.config['dataset_name']}")
        print(f"{'='*60}")
        print(f"Best Accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        print(f"Overall Accuracy: {oa:.2f}%")
        print(f"Average Accuracy: {aa:.2f}%")
        print(f"Kappa Score: {kappa:.2f}%")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Test Time: {test_time:.2f} seconds")
        print(f"\nPer-Class Accuracies:")
        for name, acc in zip(target_names, each_acc):
            print(f"  {name}: {acc:.2f}%")
        print(f"\nDetailed Classification Report:")
        print(classification)
        print(f"\nConfusion Matrix:")
        print(confusion)
        print(f"\nResults saved to: {file_name}")
        print(f"{'='*60}")
        
        return self.best_accuracy, self.best_epoch

def main():
    """Main function with dataset selection"""
    parser = argparse.ArgumentParser(description='Enhanced Hyperspectral Image Classification')
    parser.add_argument('--dataset', type=str, default='Qingyun', 
                       choices=['LongKou', 'IndianPines', 'PaviaU', 'PaviaC', 'Salinas', 'HongHu','Qingyun'],
                       help='Dataset to use for training/evaluation')
    parser.add_argument('--model', type=str, default='nvidia/GCViT',
                       help='Hugging Face model to use (e.g., nvidia/GCViT, nvidia/GCViT-Tiny)')
    parser.add_argument('--skip-pretrained', action='store_true',
                       help='Skip loading pretrained weights (train from scratch)')
    parser.add_argument('--eval-only', action='store_true', 
                       help='Run evaluation only (skip training)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model for evaluation-only mode')
    
    args = parser.parse_args()
    
    # Configuration with dataset selection - OPTIMIZED for CPU/GPU compatibility
    config = {
        'num_epochs': 100,                  # Reduced for faster training
        'batch_size': 32,                  # Smaller batch size for stability
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'dataset_name': args.dataset,
        'hf_backbone': args.model,
        'skip_pretrained': args.skip_pretrained,
        'lora_rank': 16,
        'lora_alpha': 32,
        'merge_lora_for_inference': False,
        'eval_interval': 5,                # Evaluate every epoch
        'use_amp': True,                   # Will be disabled for CPU automatically
        'label_smoothing': 0.1,
        'patience': 15,                    # More patience for CPU training
        'use_wandb': False,                # Disabled by default for stability
        'scheduler': {
            'T_0': 10,                     # Shorter warm restarts
            'T_mult': 2,
            'eta_min': 1e-6
        },
        'warmup_epochs': 5,                # Reduced warmup
        'grad_clip': 0.5,
        'lora_dropout': 0.2,
        'fast_eval': False,                # Evaluate on full test set during training
        'eval_batch_size': 32,             # Smaller eval batch size
        'gradient_accumulation_steps': 2   # Reduced for CPU training
    }
    
    # Create trainer
    trainer = EnhancedTrainer(config)
    
    if args.eval_only:
        # Evaluation-only mode
        if args.model_path is None:
            # Try to find the best model automaticallyy
            model_path = f'best_enhanced_model_{config["dataset_name"]}.pth'
            if not os.path.exists(model_path):
                print(f"Error: No saved model found at {model_path}")
                print("Please provide a model path using --model-path or train the model first.")
                return
        else:
            model_path = args.model_path
            
        print(f"Loading model from: {model_path}")
        trainer.model.load_state_dict(torch.load(model_path))
        
        # Run evaluation
        print(f"\n{'='*60}")
        print(f"EVALUATION MODE - Dataset: {config['dataset_name']}")
        print(f"{'='*60}")
        
        tic = time.perf_counter()
        y_pred_test, y_test = test(trainer.device, trainer.model, trainer.test_loader)
        toc = time.perf_counter()
        test_time = toc - tic
        
        # Calculate comprehensive metrics
        classification, oa, confusion, each_acc, aa, kappa, target_names = acc_reports(
            y_test, y_pred_test, dataset_name=config['dataset_name']
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {config['dataset_name']}")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {oa:.2f}%")
        print(f"Average Accuracy: {aa:.2f}%")
        print(f"Kappa Score: {kappa:.2f}%")
        print(f"Test Time: {test_time:.2f} seconds")
        print(f"\nPer-Class Accuracies:")
        for name, acc in zip(target_names, each_acc):
            print(f"  {name}: {acc:.2f}%")
        print(f"\nDetailed Classification Report:")
        print(classification)
        print(f"\nConfusion Matrix:")
        print(confusion)
        
        # Save results with comprehensive analysis (RESTORED from working version)
        file_name = save_enhanced_results(
            trainer.model, config, oa, 0, 0, test_time, classification, oa, confusion, 
            each_acc, aa, kappa, target_names, config['pretrained_model_name'], 
            trainer.efficiency_results  # Use actual efficiency results
        )
        print(f"\nResults saved to: {file_name}")
        
    else:
        # Training mode (includes evaluation during and after training)
        print(f"\n{'='*60}")
        print(f"TRAINING MODE - Dataset: {config['dataset_name']}")
        print(f"{'='*60}")
        best_acc, best_epoch = trainer.train()

if __name__ == "__main__":
    main() 