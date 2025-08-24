import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import scipy.io 
def get_classification_map(y_pred, y):
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1
    return cls_labels

import numpy as np

def list_to_colormap(x_list):
    """Map WHU-Hi-HongHu class labels to GT RGB colors."""
    y = np.zeros((len(x_list), 3))
    for idx, label in enumerate(x_list):
        if label == 0:
            y[idx] = [0, 0, 0]  # Background
        elif label == 1:
            y[idx] = [72, 34, 25]
        elif label == 2:
            y[idx] = [0, 44, 167]
        elif label == 3:
            y[idx] = [240, 138, 6]
        elif label == 4:
            y[idx] = [0, 194, 0]
        elif label == 5:
            y[idx] = [194, 174, 209]
        elif label == 6:
            y[idx] = [120, 196, 255]
        elif label == 7:
            y[idx] = [0, 255, 140]
        elif label == 8:
            y[idx] = [60, 115, 65]
        elif label == 9:
            y[idx] = [204, 255, 204]
        elif label == 10:
            y[idx] = [255, 255, 0]
        elif label == 11:
            y[idx] = [255, 0, 255]
        elif label == 12:
            y[idx] = [63, 0, 161]
        elif label == 13:
            y[idx] = [132, 255, 255]
        elif label == 14:
            y[idx] = [247, 255, 255]
        elif label == 15:
            y[idx] = [188, 255, 127]
        elif label == 16:
            y[idx] = [173, 255, 0]
        elif label == 17:
            y[idx] = [255, 208, 108]
        elif label == 18:
            y[idx] = [255, 122, 0]
        elif label == 19:
            y[idx] = [255, 255, 193]
        elif label == 20:
            y[idx] = [153, 255, 255]
        elif label == 21:
            y[idx] = [121, 133, 121]
        elif label == 22:
            y[idx] = [255, 153, 0]
    return y / 255.0


def classification_map(map_data, ground_truth, dpi, save_path):
    """Generate classification map image"""
    # Clear any existing plots
    plt.clf()
    
    # Set the backend to Agg
    plt.switch_backend('agg')
    
    # Create new figure with tight layout
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)
    
    # Create axis that takes up the entire figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display the image
    im = ax.imshow(map_data, interpolation='nearest')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with tight layout
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # Cleanup
    plt.close(fig)
    plt.close('all')
'''
def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    with torch.no_grad():  # Add no_grad context for inference
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
    return y_pred_test, y_test

def get_cls_map(net, device, all_data_loader, y):
    """Generate classification maps for the entire dataset"""
    # Ensure network is in evaluation mode
    net.eval()
    
    # Clear any existing plots
    plt.close('all')
    
    # Create output directory
    os.makedirs('classification_maps', exist_ok=True)
    
    # Get predictions
    with torch.no_grad():
        y_pred, _ = test(device, net, all_data_loader)
    
    # Generate classification map
    cls_labels = get_classification_map(y_pred, y)
    
    # Convert to colormaps
    x = np.ravel(cls_labels)
    gt = y.flatten()
    
    # Generate color mappings
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    
    # Reshape to image dimensions
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    
    # Save prediction and ground truth maps
    classification_map(y_re, y, 300, 'classification_maps/predictions.png')
    classification_map(gt_re, y, 300, 'classification_maps/ground_truth.png')
    
    print('Classification maps saved successfully to classification_maps directory')
    return cls_labels
'''
def test(device, net, test_loader):
    net.eval()
    count = 0
    y_pred_test = 0
    y_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Use the same normalization as during training
            inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
            
            outputs = net(inputs)
            outputs = torch.softmax(outputs, dim=1)  # Add softmax for better probability distribution
            outputs = outputs.max(1)[1].cpu().numpy()  # Get class predictions
            
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
    
    return y_pred_test, y_test

def get_cls_map(net, device, all_data_loader, y):
    """Generate classification maps with enhanced progress tracking"""
    net.eval()
    
    # Get predictions with enhanced progress tracking
    predictions = []
    total_samples = len(all_data_loader.dataset)
    
    with torch.no_grad():
        progress_bar = tqdm(all_data_loader, desc="Generating classification maps", 
                           leave=True, position=0, ncols=100)
        
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)
            # Basic normalization
            inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
            
            # Get predictions with softmax
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.max(1)[1].cpu().numpy()
            predictions.extend(preds)
            
            # Update progress bar with informative metrics
            progress_bar.set_postfix({
                'Samples': f'{len(predictions)}/{total_samples}',
                'Classes': f'{len(np.unique(predictions))}',
                'Progress': f'{len(predictions)/total_samples*100:.1f}%'
            })
    
    predictions = np.array(predictions)
    
    # Generate classification map
    cls_labels = get_classification_map(predictions, y)
    
    # Basic spatial filtering
    from scipy import ndimage
    cls_labels = ndimage.median_filter(cls_labels, size=3)
    
    # Convert to colormaps
    x = np.ravel(cls_labels)
    gt = y.flatten()
    
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    
    # Save maps
    os.makedirs('classification_maps', exist_ok=True)
    
    # Save classification maps
    plt.figure(figsize=(10, 10))
    plt.imshow(y_re)
    plt.title('Prediction Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('classification_maps/prediction_hong.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(gt_re)
    plt.title('Ground Truth Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('classification_maps/ground_truth-hong.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save difference map
    diff_map = np.zeros_like(cls_labels)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j] != 0:  # Only consider non-background pixels
                diff_map[i,j] = 1 if cls_labels[i,j] != y[i,j] else 0
    
    plt.figure(figsize=(10, 10))
    plt.imshow(diff_map, cmap='coolwarm')
    plt.title('Difference Map')
    plt.colorbar(label='Incorrect Predictions')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('classification_maps/difference_map_hong.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f'Classification maps saved to: classification_maps/')
    return cls_labels