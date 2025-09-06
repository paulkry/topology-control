import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import base64
from io import BytesIO
import numpy as np
import os
import time
from glob import glob
from deepsdf.Model import DeepSDF

torch.set_default_dtype(torch.float32)

class SDFDataset(Dataset):
    """
    SDF Dataset class that uses preprocessed data from CDataProcessor.
    Compatible with DeepSDF training pipeline - supports latent code learning.
    """
    
    def __init__(self, dataset_info, split='train', fix_seed=False, volume_coords=None):
        """
        Initialize dataset from CDataProcessor output.
        
        Parameters:
            dataset_info (dict): Output from CDataProcessor.generate_sdf_dataset()
            split (str): 'train' or 'val'
            fix_seed (bool): Whether to fix random seed for reproducibility
            volume_coords (torch.Tensor): Volume coordinates for additional sampling
        """
        self.split = split
        self.fix_seed = fix_seed
        self.dataset_params = dataset_info['dataset_params']
        
        # Get file list for this split
        self.files = dataset_info[f'{split}_files']
        
        # Initialize latent vectors for each mesh
        z_dim = self.dataset_params['z_dim']
        latent_mean = self.dataset_params['latent_mean']
        latent_sd = self.dataset_params['latent_sd']
        
        self.latent_vectors = torch.randn(len(self.files), z_dim)
        self.latent_vectors = (self.latent_vectors * latent_sd) + latent_mean
        self.latent_vectors.requires_grad = True
        
        # Store volume coordinates for additional sampling if needed
        self.volume_coords = volume_coords or dataset_info.get('volume_coords')
    
    def __getitem__(self, idx):
        if self.fix_seed:
            np.random.seed(42)
        
        # Load preprocessed data from PointCloudProcessor
        file_info = self.files[idx]
        sampled_points = np.load(file_info['points_file'])
        sdf_values = np.load(file_info['distances_file'])
        
        # Convert to tensors
        sampled_points = torch.tensor(sampled_points, dtype=torch.float32)
        sdf_values = torch.tensor(sdf_values, dtype=torch.float32)
        
        return sampled_points, self.latent_vectors[idx], sdf_values
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences"""
        # Find minimum sample length, but ensure it's at least 1
        sample_lengths = [len(i[0]) for i in batch]
        min_sample_len = max(1, min(sample_lengths))
        
        # Also check if we have enough samples for meaningful training
        if min_sample_len < 100:  # Adjust threshold as needed
            print(f"Warning: Very small sample size detected: {min_sample_len}")
        
        x_vals = [i[0][:min_sample_len].unsqueeze(0) for i in batch]
        vec = [i[1].unsqueeze(0) for i in batch]
        y_vals = [i[2][:min_sample_len].unsqueeze(0) for i in batch]
        return torch.cat(x_vals, dim=0), torch.cat(vec, dim=0), torch.cat(y_vals, dim=0)
    
    def __len__(self):
        return len(self.files)

class ModelTrainer:
    """Model trainer for 3D shape classification pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.optimizer_name = config.get('optimizer', 'adam')
        self.loss_function = config.get('loss_function', 'mse')
        self.epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.network_learning_rate = config.get('network_learning_rate', 1e-3)
        self.latent_learning_rate = config.get('latent_learning_rate', 0.01)
        
        # Data paths
        self.processed_data_path = config.get('processed_data_path', 'data/processed')
        self.max_points = config.get('max_points', 10000)
        
        # Experiment integration - check if experiment_id is provided by orchestrator
        self.experiment_id = config.get('experiment_id', None)
        self.artifacts_base = config.get('artifacts_base', None)
        
        # Model saving configuration
        if self.experiment_id and self.artifacts_base:
            # Use orchestrator's experiment structure
            self.experiment_dir = os.path.join(self.artifacts_base, f'experiment_{self.experiment_id}')
            self.save_dir = os.path.join(self.experiment_dir, 'training_artifacts')
        else:
            # Fallback to standalone mode
            self.save_dir = config.get('save_dir', 'models/checkpoints')
            print("Running in standalone mode")
        
        self.save_best_only = config.get('save_best_only', True)
        self.save_frequency = config.get('save_frequency', 5)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.dataset_type = 'sdf'
        # Keep a generic learning rate attribute for legacy optimizer path
        self.learning_rate = self.network_learning_rate
        print(f"    ✓ Trainer Device: {self.device}")
        print(f"    ✓ Training artifacts will be saved to: {self.save_dir}")
        
    def create_datasets(self, dataset_info=None):
        """
        Create training dataset.
        
        Args:
            dataset_info: Optional dataset info from CDataProcessor (for SDF datasets)
            
        Returns:
            train_dataset
        """
        
        # Always use SDFDataset; dataset_info must be provided
        if dataset_info is None:
            raise ValueError("dataset_info is required for SDF training")
        train_dataset = SDFDataset(dataset_info, split='train')
        return train_dataset
    
    def create_data_loaders(self, train_dataset):
        """Create data loader for training (no validation)."""
        # Explicitly use the SDFDataset collate function for SDF datasets
        if isinstance(train_dataset, SDFDataset):
            collate_fn = train_dataset.collate_fn
            # Test the collate function directly
            try:
                sample1 = train_dataset[0]
                sample2 = train_dataset[1] if len(train_dataset) > 1 else train_dataset[0]
                test_batch = [sample1, sample2]
                _ = collate_fn(test_batch)
            except Exception:
                import traceback
                traceback.print_exc()
        else:
            collate_fn = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        return train_loader
            
    def load_checkpoint(self, filepath, model, optimizer=None):
        """
        Load model checkpoint from disk.
        
        Args:
            filepath: Path to checkpoint file
            model: PyTorch model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            dict: Loaded checkpoint data
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"✓ Checkpoint loaded from: {filepath}")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
            
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def train(self, model, train_loader, criterion, optimizer, latent_vectors=None, dataset_info=None):
        """
        Training loop for DeepSDF. Only optimizes network and training latents.
        Now also saves checkpoints and best model during training.
        Returns: history, best_model_state, best_optimizer_state, best_latent_state, best_epoch
        """
        history = []
        best_train_loss = float('inf')
        best_model_state = None
        best_optimizer_state = None
        best_latent_state = None
        best_epoch = 0
        
        for epoch in tqdm(range(self.epochs), desc="Epochs", unit="epoch"):
            start_time = time.time()
            train_loss = self._run_epoch(
                train_loader, model, criterion, optimizer, 
                is_training=True, epoch=epoch+1
            )
            epoch_time = time.time() - start_time
            is_best = train_loss < best_train_loss
            if is_best:
                best_train_loss = train_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                best_optimizer_state = optimizer.state_dict().copy()
                if latent_vectors is not None:
                    best_latent_state = latent_vectors.detach().clone()
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'is_best': is_best,
                'epoch_time': epoch_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            history.append(epoch_data)
        return history, best_model_state, best_optimizer_state, best_latent_state, best_epoch

    def load_best_model(self, run_directory, model, optimizer=None):
        """
        Convenience method to load the best model from a training run.
        
        Args:
            run_directory: Path to training run directory
            model: Model instance to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            dict: Loaded checkpoint data
        """
        best_model_path = os.path.join(run_directory, 'best_model.pth')
        return self.load_checkpoint(best_model_path, model, optimizer)
    
    def load_latest_checkpoint(self, run_directory, model, optimizer=None):
        """
        Convenience method to load the latest checkpoint for resuming training.
        
        Args:
            run_directory: Path to training run directory  
            model: Model instance to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            dict: Loaded checkpoint data
        """
        latest_checkpoint_path = os.path.join(run_directory, 'latest_checkpoint.pth')
        return self.load_checkpoint(latest_checkpoint_path, model, optimizer)
    
    def _get_sdf_loss(self):
        """Get loss function for SDF learning."""
        delta = self.config.get('sdf_delta', 1.0)
        latent_sd = self.config.get('latent_sd', 0.01)
        
        class DeepSDFLoss(nn.Module):
            def __init__(self, delta, sd):
                super().__init__()
                self.mae = nn.L1Loss()
                self.delta = delta
                self.sd = sd
            
            def forward(self, yhat, y, latent=None):
                clamped_yhat = torch.clamp(yhat, -self.delta, self.delta)
                clamped_y = torch.clamp(y, -self.delta, self.delta)
                l1 = torch.mean(torch.abs(clamped_yhat - clamped_y))
                
                # Add latent regularization if latent codes are provided
                if latent is not None:
                    l2 = self.sd**2 * torch.mean(torch.linalg.norm(latent, dim=1, ord=2))
                    return l1 + l2
                return l1
        
        return DeepSDFLoss(delta, latent_sd)
    
    def _get_optimizer(self, model, latent_vectors=None):
        """Get optimizer with separate learning rates for network and latent vectors."""
        
        if latent_vectors is not None:
            # Use separate learning rates for SDF training
            lr_net = self.config.get('network_learning_rate', 0.001)
            lr_latent = self.config.get('latent_learning_rate', 0.01)
            
            if self.optimizer_name.lower() == 'adam':
                optimizer = optim.Adam([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": latent_vectors, "lr": lr_latent}
                ])
            elif self.optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": latent_vectors, "lr": lr_latent}
                ])
            else:
                # Default to Adam for SDF training
                optimizer = optim.Adam([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": latent_vectors, "lr": lr_latent}
                ])
            
            return optimizer
        
        else:
            # Single learning rate for traditional training
            params = list(model.parameters())
            lr = self.learning_rate
            
            if self.optimizer_name.lower() == 'adam':
                return optim.Adam(params, lr=lr)
            elif self.optimizer_name.lower() == 'adamw':
                return optim.AdamW(params, lr=lr)
            elif self.optimizer_name.lower() == 'sgd':
                return optim.SGD(params, lr=lr, momentum=0.9)
            else:
                return optim.Adam(params, lr=lr)  # Default
    
    def _run_epoch(self, data_loader, model, loss_fn, optimizer, is_training, epoch=1):
        """
        Run a single epoch of training or validation.
        """
        if is_training:
            model.train()
        else:
            model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, batch_data in enumerate(data_loader):
                # Only handle SDF dataset: (coords, latents, targets)
                if len(batch_data) == 3:
                    coords, latents, targets = batch_data
                    coords = coords.to(self.device)  # [batch, num_points, 3]
                    latents = latents.to(self.device)  # [batch, z_dim]
                    targets = targets.to(self.device)  # [batch, num_points]
                    
                    # Pass 3D tensors directly to model
                    outputs = model(latents, coords)  # outputs: [batch, num_points]
                    
                    # Flatten only for loss computation
                    outputs_flat = outputs.view(-1)  # [batch*num_points]
                    targets_flat = targets.view(-1)  # [batch*num_points]
                    
                    # Compute loss
                    if hasattr(loss_fn, 'forward') and loss_fn.__class__.__name__ == 'DeepSDFLoss':
                        loss = loss_fn(outputs_flat, targets_flat, latents)
                    else:
                        loss = loss_fn(outputs_flat, targets_flat)
                else:
                    raise ValueError(f"Expected SDF dataset with 3 elements (coords, latents, targets), got {len(batch_data)} elements")
                
                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _create_report(self, training_history, best_train_loss):
        """Create training report with metrics and visualization (training only)."""
        if not training_history:
            return {"error": "No training history"}
        total_time = sum(h['epoch_time'] for h in training_history)
        report = {
            "total_epochs": len(training_history),
            "best_train_loss": best_train_loss,
            "final_train_loss": training_history[-1]['train_loss'],
            "total_training_time": total_time,
            "avg_epoch_time": total_time / len(training_history),
            "training_history": training_history,
            "dataset_type": self.dataset_type
        }
        # Add loss plot
        loss_plot = self._create_loss_plot(training_history)
        if loss_plot:
            report["loss_plot"] = loss_plot
        return report
    
    def _create_loss_plot(self, training_history):
        """Create loss curve visualization for training only."""
        if not training_history:
            return None

        try:
            epochs = [h['epoch'] for h in training_history]
            train_losses = [h['train_loss'] for h in training_history]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'o-', label='Training Loss', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {self.dataset_type.upper()} Dataset (No Validation)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()

            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"Warning: Could not create loss plot: {e}")
            return None
    
    def run_training(self, model, dataset_info=None):
        """
        Main training loop (training only) with model saving.
        Args:
            model: PyTorch model to train
            dataset_info: Optional dataset info from CDataProcessor (for SDF datasets)
        Returns:
            tuple: (trained_model, training_report)
        """
        print(f"    ✓ Starting training on {self.device}")
        model = model.to(self.device)
        # Create dataset and data loader
        try:
            train_dataset = self.create_datasets(dataset_info)
            train_loader = self.create_data_loaders(train_dataset)
            # Report total sampled points (raw) used for training
            if hasattr(train_dataset, 'total_points'):
                print(f"    ✓ Training point samples (raw): {train_dataset.total_points} (avg {train_dataset.avg_points:.1f} per mesh, {len(train_dataset.files)} meshes)")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return model, {"error": f"Dataset loading failed: {e}"}
        # Setup loss function and optimizer
        criterion = self._get_sdf_loss()
        if hasattr(train_dataset, 'latent_vectors'):
            latent_vectors = train_dataset.latent_vectors.to(self.device)
            optimizer = self._get_optimizer(model, latent_vectors)  
            print(f"    ✓ SDF training setup complete with dual learning rates\n")
        else:
            print(" No latent vectors found in SDF dataset, using single learning rate")
            optimizer = self._get_optimizer(model)
            latent_vectors = None
            
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.save_dir, f'run_{timestamp}')
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        model_dir = os.path.join(run_dir, 'models')
        config_dir = os.path.join(run_dir, 'configs')
        metrics_dir = os.path.join(run_dir, 'metrics')
        for directory in [run_dir, checkpoint_dir, model_dir, config_dir, metrics_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Save initial configuration
        config_path = os.path.join(config_dir, 'training_config.json')
        initial_config = {
            'training_config': self.config,
            'model_config': {
                'model_type': model.__class__.__name__,
                'parameters': sum(p.numel() for p in model.parameters()),
                'device': str(self.device)
            },
            'dataset_info': dataset_info,
            'experiment_id': self.experiment_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        import json
        with open(config_path, 'w') as f:
            json.dump(initial_config, f, indent=2)

        total_start_time = time.time()
        training_history, best_model_state, best_optimizer_state, best_latent_state, best_epoch = self.train(
            model, train_loader, criterion, optimizer, latent_vectors, dataset_info
        )
        best_train_loss = min([h['train_loss'] for h in training_history]) if training_history else None
        total_training_time = time.time() - total_start_time

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if best_optimizer_state is not None:
                optimizer.load_state_dict(best_optimizer_state)
                print(f"\n    ✓ Optimizer state restored")
            if best_latent_state is not None and latent_vectors is not None:
                latent_vectors.data = best_latent_state.to(latent_vectors.device)
                print(f"    ✓ Latent vectors restored")
                
            print(f"    ✓ Best model restored from epoch {best_epoch}")
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            best_train_loss_val = None
            if training_history:
                for h in training_history:
                    if h['epoch'] == best_epoch:
                        best_train_loss_val = h['train_loss']
                        break
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'latent_vectors': best_latent_state.detach().cpu() if best_latent_state is not None else None,
                'config': self.config,
                'dataset_type': self.dataset_type,
                'train_loss': best_train_loss_val,
                'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
            }, best_model_path)

        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        last_train_loss = training_history[-1]['train_loss'] if training_history else None
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'latent_vectors': latent_vectors.detach().cpu() if latent_vectors is not None else None,
            'config': self.config,
            'dataset_type': self.dataset_type,
            'train_loss': last_train_loss,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }, latest_checkpoint_path)

        final_model_data = {
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_train_loss': best_train_loss,
            'training_history': training_history,
            'config': self.config,
            'dataset_type': self.dataset_type,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'total_training_time': total_training_time,
            'experiment_id': self.experiment_id,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        if latent_vectors is not None:
            final_model_data['latent_vectors'] = latent_vectors.detach().cpu()
        if dataset_info is not None:
            final_model_data['dataset_info'] = dataset_info
        final_path = os.path.join(model_dir, 'final_model_complete.pth')
        torch.save(final_model_data, final_path)

        training_summary = {
            'experiment_id': self.experiment_id,
            'total_training_time': total_training_time,
            'best_epoch': best_epoch,
            'best_train_loss': best_train_loss,
            'final_train_loss': training_history[-1]['train_loss'] if training_history else None,
            'total_epochs': len(training_history),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'run_directory': run_dir,
            'saved_artifacts': {
                'best_model': os.path.join(model_dir, 'best_model.pth'),
                'final_complete': final_path,
                'latest_checkpoint': latest_checkpoint_path,
                'training_config': config_path,
            },
            'status': 'completed',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        summary_path = os.path.join(metrics_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)

        report = self._create_report(training_history, best_train_loss)
        report['run_directory'] = run_dir
        report['experiment_id'] = self.experiment_id
        report['model_saved'] = True
        report['best_epoch'] = best_epoch
        report['total_training_time'] = total_training_time
        report['training_artifacts'] = {
            'best_model_path': os.path.join(model_dir, 'best_model.pth'),
            'final_model_path': os.path.join(model_dir, 'final_model_complete.pth'),
            'latest_checkpoint_path': latest_checkpoint_path,
            'training_summary_path': summary_path,
            'run_directory': run_dir
        }
        return model, report