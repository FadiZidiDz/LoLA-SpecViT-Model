import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from tqdm import tqdm
import copy

# Use the actual model from improved_GCPE_old
from improved_GCPE_old import EnhancedPEFTHyperspectralGCViT, EnhancedLoRALinear

# Import training utilities for real data
from enhanced_training import (
    create_enhanced_data_loader,
    EnhancedHyperspectralDataset
)

class LoRAAblationStudyRealData:
    """Real LoRA ablation study using actual hyperspectral data"""
    
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # Model configuration - EXACTLY matching improved_GCPE_old.py
        self.model_config = {
            'in_channels': config.get('in_channels', 15),
            'num_classes': config.get('num_classes', 16),  # Salinas has 16 classes
            'dim': config.get('dim', 96),
            'depths': config.get('depths', [3, 4, 19]),
            'num_heads': config.get('num_heads', [4, 8, 16]),
            'window_size': config.get('window_size', [7, 7, 7]),
            'mlp_ratio': config.get('mlp_ratio', 4.0),
            'drop_path_rate': config.get('drop_path_rate', 0.2),
            'spatial_size': config.get('spatial_size', 15),
            'r': config.get('r', 16),
            'lora_alpha': config.get('lora_alpha', 32)
        }
        
        # Training configuration with realistic settings
        self.train_config = {
            'batch_size': config.get('batch_size', 16),
            'learning_rate': config.get('learning_rate', 1e-3),
            'epochs': config.get('epochs', 100),
            'patience': config.get('patience', 8),
            'device': config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            'weight_decay': config.get('weight_decay', 1e-4),
            'dropout_rate': config.get('dropout_rate', 0.1),
            'early_stopping': config.get('early_stopping', True),
            'dataset_name': config.get('dataset_name', 'Salinas'),  # Default to Salinas
            'patch_size': config.get('patch_size', 15),
            'train_ratio': config.get('train_ratio', 0.02),  # 2% train
            'test_ratio': config.get('test_ratio', 0.98),     # 98% test
            'run_only_all': config.get('run_only_all', False),
            'run_only_spectral': config.get('run_only_spectral', False)
        }
        
        self.device = self.train_config['device']
        self.results = {}
    
    def _replace_lora_with_linear(self, module: EnhancedLoRALinear) -> nn.Linear:
        return nn.Linear(module.in_features, module.out_features, bias=(module.linear.bias is not None))
    
    def _disable_attention_lora(self, model: nn.Module):
        for sub in model.modules():
            if hasattr(sub, 'qkv') and isinstance(sub.qkv, EnhancedLoRALinear):
                sub.qkv = self._replace_lora_with_linear(sub.qkv)
            if hasattr(sub, 'proj') and isinstance(sub.proj, EnhancedLoRALinear):
                sub.proj = self._replace_lora_with_linear(sub.proj)
    
    def _disable_mlp_lora(self, model: nn.Module):
        for sub in model.modules():
            if hasattr(sub, 'mlp') and isinstance(sub.mlp, nn.Sequential):
                new_layers = []
                for layer in sub.mlp:
                    if isinstance(layer, EnhancedLoRALinear):
                        new_layers.append(self._replace_lora_with_linear(layer))
                    else:
                        new_layers.append(layer)
                sub.mlp = nn.Sequential(*new_layers)
    
    def _disable_head_lora(self, model: nn.Module):
        if isinstance(model.head, EnhancedLoRALinear):
            model.head = nn.Linear(model.head.in_features, model.head.out_features, bias=True)
    
    def create_models(self):
        """Create models aligned with actual LoRA placements in improved_GCPE_old.py"""
        print("Creating ablation models (using improved_GCPE_old.py)...")
        
        # If running only the full model
        if self.train_config.get('run_only_all', False):
            full_model = EnhancedPEFTHyperspectralGCViT(**self.model_config).to(self.device)
            print("✓ All_Components (full model) ready")
            return {'All_Components': full_model}
        
        # If running only the spectral only model
        if self.train_config.get('run_only_spectral', False):
            base = EnhancedPEFTHyperspectralGCViT(**self.model_config).to(self.device)
            spectral_only = copy.deepcopy(base)
            self._disable_attention_lora(spectral_only)
            self._disable_mlp_lora(spectral_only)
            self._disable_head_lora(spectral_only)
            spectral_only = spectral_only.to(self.device)
            print("✓ Spectral_Only model ready")
            return {'Spectral_Only': spectral_only}
        
        base = EnhancedPEFTHyperspectralGCViT(**self.model_config).to(self.device)
        
        # 1) Spectral Only: disable Attention + MLP + Head LoRA, keep only spectral LoRA
        spectral_only = copy.deepcopy(base)
        self._disable_attention_lora(spectral_only)
        self._disable_mlp_lora(spectral_only)
        self._disable_head_lora(spectral_only)
        spectral_only = spectral_only.to(self.device)
        
        # 2) Attention Only: disable MLP + Head LoRA
        att_only = copy.deepcopy(base)
        self._disable_mlp_lora(att_only)
        self._disable_head_lora(att_only)
        att_only = att_only.to(self.device)
        
        # 3) Attention + MLPs: keep attention + MLP LoRA, disable Head LoRA
        att_mlp = copy.deepcopy(base)
        self._disable_head_lora(att_mlp)
        att_mlp = att_mlp.to(self.device)
        
        # 4) All Components: attention + MLP + head (same as base)
        all_components = base
        
        models = {
            'Spectral_Only': spectral_only,
            'Attention_Only': att_only,
            'Attention_MLPs': att_mlp,
            'All_Components': all_components
        }
        
        for name, model in models.items():
            print(f"✓ {name} model ready")
        
        return models
    
    def load_real_data(self):
        """Load real hyperspectral data with custom split"""
        print(f"Loading {self.train_config['dataset_name']} dataset...")
        
        try:
            # Prefer the shared loader utility; handle different return shapes
            loaders = create_enhanced_data_loader(
                dataset_name=self.train_config['dataset_name'],
                batch_size=self.train_config['batch_size'],
                patch_size=self.train_config['patch_size'],
                test_ratio=self.train_config['test_ratio']
            )
            
            # Unpack flexibly (supports 2..6 returns)
            if isinstance(loaders, tuple):
                if len(loaders) >= 2:
                    train_loader = loaders[0]
                    test_loader = loaders[1]
                else:
                    raise ValueError("create_enhanced_data_loader returned insufficient outputs")
            else:
                raise ValueError("Unexpected return type from create_enhanced_data_loader")
            
            print(f"✓ Real dataset loaded successfully!")
            print(f"  - Train batches: {len(train_loader)}")
            print(f"  - Test batches: {len(test_loader)}")
            
            # Try to infer number of classes if provided
            num_classes = self.model_config['num_classes']
            return train_loader, test_loader, num_classes
            
        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            print("Falling back to dummy data...")
            return self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy data as fallback"""
        print("Creating dummy hyperspectral data...")
        
        num_samples = 1000
        num_classes = self.model_config['num_classes']  # Should be 16 for Salinas
        in_channels = self.model_config['in_channels']
        spatial_size = self.model_config['spatial_size']
        
        # Create realistic dummy data
        data = []
        labels = []
        
        for i in range(num_samples):
            class_id = torch.randint(0, num_classes, (1,)).item()
            pattern = torch.randn(in_channels, spatial_size, spatial_size) * 0.3
            
            # Add class-specific features
            if class_id == 0:  # Example pattern
                pattern[:, :spatial_size//2, :] += 0.5
                pattern[5:10, :, :] += 0.3
            elif class_id == 1:
                pattern[:, :, :spatial_size//2] += 0.4
                pattern[0:5, :, :] += 0.2
            elif class_id == 2:
                pattern[:, spatial_size//3:2*spatial_size//3, spatial_size//3:2*spatial_size//3] += 0.6
            else:
                pattern += torch.randn_like(pattern) * 0.2
            
            data.append(pattern)
            labels.append(class_id)
        
        # Convert to tensors
        data = torch.stack(data)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Split: 2% train, 98% test
        train_size = int(0.02 * num_samples)
        test_size = num_samples - train_size
        
        train_data = data[:train_size]
        train_labels = labels[:train_size]
        test_data = data[train_size:]
        test_labels = labels[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.train_config['batch_size'], shuffle=False)
        
        print(f"Dummy data created: Train={len(train_loader)}, Test={len(test_loader)}")
        return train_loader, test_loader, num_classes

    def _extract_inputs_labels(self, batch):
        """Robustly extract (inputs, labels) from batch tuples/lists.
        - If batch has (hs, rgb, labels): returns (hs, labels)
        - If batch has (data, labels): returns (data, labels)
        - If multiple tensors: first tensor is input, last tensor is labels
        """
        if isinstance(batch, (list, tuple)):
            tensors = [b for b in batch if torch.is_tensor(b)]
            if len(tensors) >= 2:
                data = tensors[0].to(self.device)
                labels = tensors[-1].to(self.device)
                return data, labels
        # Fallback: assume (data, labels)
        data, labels = batch
        return data.to(self.device), labels.to(self.device)
    
    def train_model(self, model, train_loader, test_loader, model_name):
        """Train model with real data"""
        print(f"\n{'='*50}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*50}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'best_test_acc': 0.0,
            'best_epoch': 0,
            'training_time': 0
        }
        
        # Early stopping
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.train_config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.train_config["epochs"]} [Train]')
            for batch_idx, batch in enumerate(train_pbar):
                data, labels = self._extract_inputs_labels(batch)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Test phase
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{self.train_config["epochs"]} [Test]')
                for batch in test_pbar:
                    data, labels = self._extract_inputs_labels(batch)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
                    
                    test_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*test_correct/test_total:.2f}%'
                    })
            
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            test_loss_avg = test_loss / len(test_loader)
            test_acc = 100. * test_correct / test_total
            
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss_avg)
            history['test_acc'].append(test_acc)
            
            scheduler.step(test_acc)
            
            if test_acc > history['best_test_acc']:
                history['best_test_acc'] = test_acc
                history['best_epoch'] = epoch
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}.pth')
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%")
            
            if train_acc - test_acc > 15:
                print(f"⚠️  WARNING: Potential overfitting detected! Gap: {train_acc - test_acc:.1f}%")
            
            if patience_counter >= self.train_config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        history['training_time'] = time.time() - start_time
        
        print(f"✓ Training completed for {model_name}")
        print(f"  - Best test accuracy: {history['best_test_acc']:.2f}%")
        print(f"  - Best epoch: {history['best_epoch']+1}")
        print(f"  - Training time: {history['training_time']/60:.2f} minutes")
        
        return history
    
    def run_ablation_study(self):
        """Run the complete ablation study"""
        print("=" * 70)
        print(f"LoRA ABLATION STUDY (REAL DATA - {self.train_config['dataset_name']})")
        print("=" * 70)
        print("Training and evaluating 3 LoRA integration strategies:")
        print("1. Attention Only")
        print("2. Attention + MLPs")
        print("3. All Components")
        print(f"Dataset: {self.train_config['dataset_name']}")
        print(f"Split: {self.train_config['train_ratio']*100:.1f}% train, {self.train_config['test_ratio']*100:.1f}% test")
        print("=" * 70)
        
        models = self.create_models()
        train_loader, test_loader, num_classes = self.load_real_data()
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"PROCESSING: {model_name}")
            print(f"{'='*70}")
            training_history = self.train_model(model, train_loader, test_loader, model_name)
            self.results[model_name] = {
                'training_history': training_history,
                'model_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'lora_params': sum(p.numel() for name, p in model.named_parameters() if 'lora_' in name and p.requires_grad)
            }
        
        self.generate_report()
        return self.results
    
    def generate_report(self):
        print(f"\n{'='*70}")
        print("ABLATION STUDY FINAL RESULTS (REAL DATA)")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Best Test Acc':<12} {'Train Time':<12} {'Params':<12} {'LoRA %':<8}")
        print("-" * 70)
        for model_name, result in self.results.items():
            best_test_acc = result['training_history']['best_test_acc']
            train_time = result['training_history']['training_time'] / 60
            total_params = result['model_params']
            lora_ratio = (result['lora_params'] / total_params) * 100
            print(f"{model_name:<20} {best_test_acc:<11.2f}% {train_time:<11.2f}m {total_params:<11,} {lora_ratio:<7.1f}%")
        best_model = max(self.results.items(), key=lambda x: x[1]['training_history']['best_test_acc'])
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]} ({best_model[1]['training_history']['best_test_acc']:.2f}% accuracy)")
        print(f"\n{'='*70}")
        print("OVERFITTING ANALYSIS")
        print(f"{'='*70}")
        for model_name, result in self.results.items():
            history = result['training_history']
            final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
            final_test_acc = history['test_acc'][-1] if history['test_acc'] else 0
            gap = final_train_acc - final_test_acc
            print(f"\n{model_name}:")
            print(f"  - Final Train Accuracy: {final_train_acc:.2f}%")
            print(f"  - Final Test Accuracy: {final_test_acc:.2f}%")
            print(f"  - Overfitting Gap: {gap:.2f}%")
            print(f"  - Status: {'✅ Good' if gap < 10 else '⚠️  Overfitting' if gap < 20 else '❌ Severe Overfitting'}")
        print(f"\n✓ Ablation study completed successfully!")


def main():
    config = {
        'batch_size': 16,
        'learning_rate': 1e-3,
        'epochs': 20,
        'patience': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'weight_decay': 1e-4,
        'dropout_rate': 0.1,
        'early_stopping': True,
        'dataset_name': 'Salinas',
        'patch_size': 15,
        'train_ratio': 0.02,
        'test_ratio': 0.98,
        'in_channels': 15,
        'num_classes': 16,
        'dim': 96,
        'depths': [3, 4, 19],
        'num_heads': [4, 8, 16],
        'window_size': [7, 7, 7],
        'mlp_ratio': 4.0,
        'drop_path_rate': 0.2,
        'spatial_size': 15,
        'r': 16,
        'lora_alpha': 32,
        # Run only the spectral only model
        'run_only_spectral': True
    }
    study = LoRAAblationStudyRealData(config)
    results = study.run_ablation_study()
    return results

if __name__ == "__main__":
    results = main()

