"""
Model training pipeline for 3D shape classification.
Handles training, validation, and model checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import os
import time
from glob import glob
torch.set_default_dtype(torch.float32)
from src.CArchitectureManager import DeepSDF

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
        
        self.latent_vectors = torch.randn(len(self.files), z_dim, device='cuda')
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
        
        # Optional: Add additional random sampling from volume coords
        if self.volume_coords is not None:
            num_volume_samples = min(1000, len(self.volume_coords))
            volume_indices = np.random.choice(len(self.volume_coords), num_volume_samples, replace=False)
            # volume_points = self.volume_coords[volume_indices].numpy()
        
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

class ShapeDataset(Dataset):
    """Dataset class for 3D shape data (points and signed distances)."""
    
    def __init__(self, data_path, split='train', max_points=10000):
        """
        Initialize dataset from processed data.
        
        Args:
            data_path: Path to processed data directory
            split: Either 'train' or 'val'
            max_points: Maximum number of points to sample from each point cloud
        """
        self.data_path = data_path
        self.split = split
        self.max_points = max_points
        
        # Load all .npy files for this split
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            raise ValueError(f"Split directory not found: {split_path}")
        
        # Find all point and distance files
        point_files = sorted(glob(os.path.join(split_path, "*_sampled_points.npy")))
        distance_files = sorted(glob(os.path.join(split_path, "*_signed_distances.npy")))
        
        if len(point_files) != len(distance_files):
            raise ValueError(f"Mismatch between point files ({len(point_files)}) and distance files ({len(distance_files)})")
        
        self.data_pairs = list(zip(point_files, distance_files))
        
        if not self.data_pairs:
            raise ValueError(f"No data files found in {split_path}")
        
        print(f"Loaded {len(self.data_pairs)} samples for {split} split (max_points: {max_points})")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        point_file, distance_file = self.data_pairs[idx]
        
        # Load data
        points = np.load(point_file)
        distances = np.load(distance_file)
        
        # Handle sampling if we have more points than max_points
        if len(points) > self.max_points:
            # Randomly sample max_points indices
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            distances = distances[indices]
        elif len(points) < self.max_points:
            # Pad with zeros if we have fewer points
            num_pad = self.max_points - len(points)
            pad_points = np.zeros((num_pad, points.shape[1]))
            pad_distances = np.zeros(num_pad)
            
            points = np.concatenate([points, pad_points], axis=0)
            distances = np.concatenate([distances, pad_distances], axis=0)
        
        # Convert to tensors
        points = torch.tensor(points, dtype=torch.float32)
        distances = torch.tensor(distances, dtype=torch.float32)
        
        # Flatten points for MLP input (assuming points are [N, 3])
        if len(points.shape) > 1:
            points = points.reshape(-1)
        
        return points, distances

class CModelTrainer:
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
            print(f"Using orchestrator experiment: {self.experiment_id}")
        else:
            # Fallback to standalone mode
            self.save_dir = config.get('save_dir', 'models/checkpoints')
            print("Running in standalone mode")
        
        self.save_best_only = config.get('save_best_only', True)
        self.save_frequency = config.get('save_frequency', 5)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Dataset type configuration
        self.dataset_type = config.get('dataset_type', 'shape')
        
        print(f"Trainer initialized - Device: {self.device}, Dataset Type: {self.dataset_type}")
        print(f"Training artifacts will be saved to: {self.save_dir}")
        
    def create_datasets(self, dataset_info=None):
        """
        Create datasets based on configuration.
        
        Args:
            dataset_info: Optional dataset info from CDataProcessor (for SDF datasets)
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        
        if self.dataset_type == 'sdf' or self.dataset_type == 'lipschitz_sdf':  # <- REMOVE the "and dataset_info is not None" condition
            # Use SDFDataset - let it handle None dataset_info gracefully
            train_dataset = SDFDataset(dataset_info, split='train')
            val_dataset = SDFDataset(dataset_info, split='val')
        else:
            # Default to ShapeDataset for traditional point cloud learning
            train_dataset = ShapeDataset(self.processed_data_path, split='train', max_points=self.max_points)
            val_dataset = ShapeDataset(self.processed_data_path, split='val', max_points=self.max_points)
        
        return train_dataset, val_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset):
        """Create data loaders from datasets."""
        
        # Explicitly use the SDFDataset collate function for SDF datasets
        if isinstance(train_dataset, SDFDataset):
            collate_fn = train_dataset.collate_fn
            
            # Test the collate function directly
            try:
                # Get a sample from the dataset
                sample1 = train_dataset[0]
                sample2 = train_dataset[1] if len(train_dataset) > 1 else train_dataset[0]
                test_batch = [sample1, sample2]
                                
                # Test collate function
                result = collate_fn(test_batch)
                
            except Exception as e:
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
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader
        
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, 
                       is_best=False, latent_vectors=None, dataset_info=None):
        """
        Save model checkpoint to disk.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            latent_vectors: Latent vectors for SDF models
            dataset_info: Dataset information for SDF models
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'dataset_type': self.dataset_type,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        # Add latent vectors for SDF models
        if latent_vectors is not None:
            checkpoint['latent_vectors'] = latent_vectors.detach().cpu()
        
        # Add dataset info for reproducibility
        if dataset_info is not None:
            checkpoint['dataset_info'] = dataset_info
        
        # Create filename
        if is_best:
            filename = f'best_model_{self.dataset_type}_epoch_{epoch}.pth'
        else:
            filename = f'checkpoint_{self.dataset_type}_epoch_{epoch}.pth'
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Save checkpoint
        try:
            torch.save(checkpoint, filepath)
            print(f"{'✓ Best model' if is_best else '📁 Checkpoint'} saved: {filename}")
            
            # Also save a 'latest' checkpoint for easy resuming
            if not is_best:
                latest_path = os.path.join(self.save_dir, f'latest_{self.dataset_type}.pth')
                torch.save(checkpoint, latest_path)
            
        except Exception as e:
            print(f" Error saving checkpoint: {e}")
    
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
            print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
            
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def train_and_validate(self, model, dataset_info=None):
        """
        Main training loop with validation and model saving.
        
        Args:
            model: PyTorch model to train
            dataset_info: Optional dataset info from CDataProcessor (for SDF datasets)
            
        Returns:
            tuple: (trained_model, training_report)
        """
        print(f"Starting training on {self.device}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Create datasets and data loaders
        try:
            train_dataset, val_dataset = self.create_datasets(dataset_info)
            train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return model, {"error": f"Dataset loading failed: {e}"}
        
        # Setup loss function and optimizer
        if self.dataset_type == 'sdf' or self.dataset_type == 'lipschitz_sdf':
            if self.dataset_type == 'sdf' :
                criterion = self._get_sdf_loss()
            elif self.dataset_type == 'lipschitz_sdf':
                criterion = self._get_lipschitz_sdf_loss()
            else:
                raise Exception
            
            # Get latent vectors from dataset for SDF training
            if hasattr(train_dataset, 'latent_vectors'):
                latent_vectors = train_dataset.latent_vectors #.to(self.device)
                optimizer = self._get_optimizer(model, latent_vectors)
                print(f"✓ SDF training setup complete with dual learning rates")
            else:
                print("⚠️  No latent vectors found in SDF dataset, using single learning rate")
                optimizer = self._get_optimizer(model)
                latent_vectors = None
        else:
            criterion = nn.MSELoss() if self.loss_function.lower() == 'mse' else nn.L1Loss()
            optimizer = self._get_optimizer(model)
            latent_vectors = None
        
        # Training state tracking
        history = []
        best_val_loss = float('inf')
        best_model_state = None
        best_optimizer_state = None
        best_latent_state = None
        best_epoch = 0
        
        # Create training run directory structure
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.save_dir, f'run_{timestamp}')
        
        # Create organized subdirectories
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
        
        print(f"Training run directory: {run_dir}")
        
        # Training loop
        total_start_time = time.time()
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training phase
            train_loss = self._run_epoch(
                train_loader, model, criterion, optimizer, 
                is_training=True, epoch=epoch+1
            )
            
            # Validation phase
            val_loss = self._run_epoch(
                val_loader, model, criterion, optimizer, 
                is_training=False, epoch=epoch+1
            )
            
            epoch_time = time.time() - start_time
            
            # Track best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                best_optimizer_state = optimizer.state_dict().copy()
                
                if latent_vectors is not None:
                    best_latent_state = latent_vectors.detach().clone()
            
            # Record history
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'is_best': is_best,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            history.append(epoch_data)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s"
                  f"{' (Best Model)' if is_best else ''}")
            
            # Save checkpoints and models
            # if epoch // self.save_frequency == 0:
            try:
                # Always save latest checkpoint
                latest_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'training_history': history,
                    'experiment_id': self.experiment_id,
                    'config': self.config,
                    'dataset_type': self.dataset_type,
                    'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
                }
                
                if latent_vectors is not None:
                    latest_checkpoint['latent_vectors'] = latent_vectors.detach().cpu()
                if dataset_info is not None:
                    latest_checkpoint['dataset_info'] = dataset_info
                
                latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
                torch.save(latest_checkpoint, latest_path)
                
                # Save best model when found
                if is_best:
                    best_checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': best_optimizer_state,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'experiment_id': self.experiment_id,
                        'config': self.config,
                        'dataset_type': self.dataset_type,
                        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
                    }
                    
                    if best_latent_state is not None:
                        best_checkpoint['latent_vectors'] = best_latent_state.cpu()
                    if dataset_info is not None:
                        best_checkpoint['dataset_info'] = dataset_info
                    
                    best_path = os.path.join(model_dir, 'best_model.pth')
                    torch.save(best_checkpoint, best_path)
                    print(f"✓ Best model saved: best_model.pth (epoch {epoch+1})")
                
                # Save periodic metrics
                if (epoch + 1) % 5 == 0:
                    metrics_data = {
                        'current_epoch': epoch + 1,
                        'training_history': history,
                        'best_epoch': best_epoch,
                        'best_val_loss': best_val_loss,
                        'experiment_id': self.experiment_id
                    }
                    metrics_path = os.path.join(metrics_dir, f'metrics_epoch_{epoch+1}.json')
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics_data, f, indent=2)
                
            except Exception as e:
                print(f"Warning: Could not save artifacts at epoch {epoch+1}: {e}")
        
        total_training_time = time.time() - total_start_time
        
        # Restore best model state
        if best_model_state is not None:
            print(f"\n Restoring best model state from epoch {best_epoch}...")
            model.load_state_dict(best_model_state)
            
            if best_optimizer_state is not None:
                optimizer.load_state_dict(best_optimizer_state)
                print(f"✓ Optimizer state restored")
            
            if best_latent_state is not None and latent_vectors is not None:
                latent_vectors.data = best_latent_state.to(latent_vectors.device)
                print(f"✓ Latent vectors restored")
            
            print(f"✓ Best model restored from epoch {best_epoch}")
        
        # Save final comprehensive model
        final_model_data = {
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'training_history': history,
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
        
        # Save final training summary
        training_summary = {
            'experiment_id': self.experiment_id,
            'total_training_time': total_training_time,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': history[-1]['train_loss'] if history else None,
            'final_val_loss': history[-1]['val_loss'] if history else None,
            'total_epochs': len(history),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'run_directory': run_dir,
            'saved_artifacts': {
                'best_model': os.path.join(model_dir, 'best_model.pth'),
                'final_complete': os.path.join(model_dir, 'final_model_complete.pth'),
                'latest_checkpoint': os.path.join(checkpoint_dir, 'latest_checkpoint.pth'),
                'training_config': os.path.join(config_dir, 'training_config.json'),
            },
            'status': 'completed',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(metrics_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        # Create training report for orchestrator
        report = self._create_report(history, best_val_loss)
        report['run_directory'] = run_dir
        report['experiment_id'] = self.experiment_id
        report['model_saved'] = True
        report['best_epoch'] = best_epoch
        report['total_training_time'] = total_training_time
        
        # Add paths to artifacts for orchestrator integration
        report['training_artifacts'] = {
            'best_model_path': os.path.join(model_dir, 'best_model.pth'),
            'final_model_path': os.path.join(model_dir, 'final_model_complete.pth'),
            'latest_checkpoint_path': os.path.join(checkpoint_dir, 'latest_checkpoint.pth'),
            'training_summary_path': summary_path,
            'run_directory': run_dir
        }
        
        print(f"\nTraining artifacts saved to: {run_dir}")
        print(f"   Models: {model_dir}")
        print(f"   Checkpoints: {checkpoint_dir}")
        print(f"   Configs: {config_dir}")
        print(f"   Metrics: {metrics_dir}")
        
        return model, report

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
    
    def _get_lipschitz_sdf_loss(self):
        """Get loss function for Lipschitz-regularized SDF learning."""
        delta = self.config.get('sdf_delta', 1.0)
        latent_sd = self.config.get('latent_sd', 0.01)
        
        class LipschitzSDFLoss(nn.Module):
            def __init__(self, delta, sd):
                super().__init__()
                self.delta = delta
                self.sd = sd
            
            def forward(self, yhat, y, latent=None):
                clamped_yhat = torch.clamp(yhat, -self.delta, self.delta)
                clamped_y = torch.clamp(y, -self.delta, self.delta)
                sdf_loss = torch.mean(torch.abs(clamped_yhat - clamped_y))
                
                if latent is not None:
                    latent_loss = self.sd**2 * torch.mean(torch.linalg.norm(latent, dim=1, ord=2))
                    return sdf_loss + latent_loss
                return sdf_loss
                
        return LipschitzSDFLoss(delta, latent_sd)
    
    def _get_optimizer(self, model, latent_vectors=None):
        """Get optimizer with separate learning rates for network and latent vectors."""
        
        if latent_vectors is not None:
            # Use separate learning rates for SDF training
            lr_net = self.config.get('network_learning_rate', 0.001)
            lr_latent = self.config.get('latent_learning_rate', 0.01)
            
            if self.optimizer_name.lower() == 'adam':
                optimizer = optim.Adam([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": [latent_vectors], "lr": lr_latent}
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

        lambda_lip = self.config.get('lipschitz_lambda', 0.01)
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, batch_data in enumerate(data_loader):
                # Only handle SDF dataset: (coords, latents, sdfs)
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

                    primary_loss = loss_fn(outputs_flat, targets_flat, latents)
             
                    if is_training and self.dataset_type == 'lipschitz_sdf' and hasattr(model, 'get_lipschitz_loss'):
                        # Get the Lipschitz regularization term directly from the model.
                        lipschitz_loss = model.get_lipschitz_loss()
                        
                        # 3. Combine the losses into the final total loss for backpropagation.
                        total_loss_for_backward = primary_loss + lambda_lip * lipschitz_loss
                    else:
                        # For validation or non-Lipschitz models, just use the primary loss.
                        total_loss_for_backward = primary_loss
                    
                    
                    if is_training:
                        optimizer.zero_grad()
                        total_loss_for_backward.backward()
                        optimizer.step()
                    
                    total_loss += total_loss_for_backward.item()
                    num_batches += 1
                else:
                    raise ValueError(f"Expected SDF dataset with 3 elements (coords, latents, targets), got {len(batch_data)} elements")
                
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _create_report(self, history, best_val_loss):
        """Create training report with metrics and visualization."""
        if not history:
            return {"error": "No training history"}
        
        total_time = sum(h['epoch_time'] for h in history)
        
        report = {
            "total_epochs": len(history),
            "best_val_loss": best_val_loss,
            "final_train_loss": history[-1]['train_loss'],
            "final_val_loss": history[-1]['val_loss'],
            "total_training_time": total_time,
            "avg_epoch_time": total_time / len(history),
            "training_history": history,
            "dataset_type": self.dataset_type
        }
        
        # Add loss plot
        loss_plot = self._create_loss_plot(history)
        if loss_plot:
            report["loss_plot"] = loss_plot
        
        return report
    
    def _create_loss_plot(self, history):
        """Create loss curve visualization."""
        if not history:
            return None
        
        try:
            epochs = [h['epoch'] for h in history]
            train_losses = [h['train_loss'] for h in history]
            val_losses = [h['val_loss'] for h in history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'o-', label='Training Loss', alpha=0.8)
            # plt.plot(epochs, val_losses, 's-', label='Validation Loss', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss - {self.dataset_type.upper()} Dataset')
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