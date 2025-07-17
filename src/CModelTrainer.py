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
from CArchitectureManager import DeepSDF

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
        
        # Optional: Add additional random sampling from volume coords
        if self.volume_coords is not None:
            num_volume_samples = min(1000, len(self.volume_coords))
            volume_indices = np.random.choice(len(self.volume_coords), num_volume_samples, replace=False)
            # volume_points = self.volume_coords[volume_indices].numpy()
        
        # Convert to tensors
        sampled_points = torch.tensor(sampled_points, dtype=torch.float32)
        sdf_values = torch.tensor(sdf_values, dtype=torch.float32)
        
        return sampled_points, self.latent_vectors[idx], sdf_values
    
    def collate_fn(self, batch):
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
        self.learning_rate = config.get('learning_rate', 1e-3)
        
        # Data paths
        self.processed_data_path = config.get('processed_data_path', 'data/processed')
        self.max_points = config.get('max_points', 10000)  # Fixed number of points per sample
        
        # Dataset type configuration
        self.dataset_type = config.get('dataset_type', 'shape')  # 'shape' or 'sdf' for now
        
        print(f"Trainer initialized - Device: {self.device}, Dataset Type: {self.dataset_type}")
        
    def create_datasets(self, dataset_info=None):
        """
        Create datasets based on configuration.
        
        Args:
            dataset_info: Optional dataset info from CDataProcessor (for SDF datasets)
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        if self.dataset_type == 'sdf' and dataset_info is not None:
            # Use SDFDataset with dataset_info from CDataProcessor
            train_dataset = SDFDataset(dataset_info, split='train')
            val_dataset = SDFDataset(dataset_info, split='val')
        else:
            # Default to ShapeDataset for traditional point cloud learning
            train_dataset = ShapeDataset(self.processed_data_path, split='train', max_points=self.max_points)
            val_dataset = ShapeDataset(self.processed_data_path, split='val', max_points=self.max_points)
        
        return train_dataset, val_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset):
        """
        Create data loaders from datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Use collate function if available (for SDF datasets)
        collate_fn = getattr(train_dataset, 'collate_fn', None)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_and_validate(self, model, dataset_info=None):
        """
        Main training loop with validation.
        
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
        
        # Setup loss function based on dataset type
        if self.dataset_type == 'sdf':
            # For SDF learning, might want custom loss
            criterion = self._get_sdf_loss()
        else:
            # Standard losses for other types
            if self.loss_function.lower() == 'mse':
                criterion = nn.MSELoss()
            elif self.loss_function.lower() == 'mae':
                criterion = nn.L1Loss()
            else:
                criterion = nn.MSELoss()  # Default
        
        # Setup optimizer
        if self.dataset_type == 'sdf' and hasattr(train_dataset, 'latent_vectors'):
            # Include latent vectors in optimization for SDF learning
            optimizer = self._get_optimizer(model, train_dataset.latent_vectors)
        else:
            optimizer = self._get_optimizer(model)
        
        # Training history
        history = []
        best_val_loss = float('inf')
        
        # Training loop
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Record history
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time
            }
            history.append(epoch_data)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s")
        
        # Create training report
        report = self._create_report(history, best_val_loss)
        
        return model, report
    
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
        """Get optimizer, optionally including latent vectors."""
        params = list(model.parameters())
        if latent_vectors is not None:
            params.append(latent_vectors)
        
        if self.optimizer_name.lower() == 'adam':
            return optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
            return optim.AdamW(params, lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            return optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        else:
            return optim.Adam(params, lr=self.learning_rate)  # Default
    
    def _run_epoch(self, data_loader, model, loss_fn, optimizer, is_training, epoch=1):
        """
        Run a single epoch of training or validation.
        
        Args:
            data_loader: DataLoader for the data
            model: PyTorch model
            loss_fn: Loss function
            optimizer: Optimizer (only used if is_training=True)
            is_training: Whether this is a training epoch
            epoch: Current epoch number
            
        Returns:
            float: Average loss for the epoch
        """
        if is_training:
            model.train()
        else:
            model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, batch_data in enumerate(data_loader):
                # Handle different dataset types
                if len(batch_data) == 3:  # SDF dataset: (coords, latents, sdfs)
                    coords, latents, targets = batch_data
                    coords = coords.to(self.device)
                    latents = latents.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Flatten coords if they're 3D: [batch, num_points, 3] -> [batch*num_points, 3]
                    if coords.dim() == 3:
                        batch_size, num_points, coord_dim = coords.shape
                        coords = coords.view(-1, coord_dim)  # [batch*num_points, 3]
                        
                        # Expand latents to match: [batch, z_dim] -> [batch*num_points, z_dim]
                        latents = latents.unsqueeze(1).expand(-1, num_points, -1).contiguous()
                        latents = latents.view(-1, latents.shape[-1])  # [batch*num_points, z_dim]
                        
                        # Flatten targets to match: [batch, num_points] -> [batch*num_points]
                        targets = targets.view(-1)
                    
                    # Concatenate latents and coords for DeepSDF input
                    model_input = torch.cat([latents, coords], dim=1)  # [batch*num_points, z_dim + coord_dim]
                    
                    # Forward pass
                    outputs = model(model_input)
                    outputs = outputs.squeeze(-1) if outputs.dim() > 1 else outputs  # Remove extra dimensions
                    
                    # Compute loss with latent regularization if applicable
                    if hasattr(loss_fn, 'forward') and loss_fn.__class__.__name__ == 'DeepSDFLoss':
                        loss = loss_fn(outputs, targets, latents)
                    else:
                        loss = loss_fn(outputs, targets)
                        
                elif len(batch_data) == 2:  # Traditional dataset: (inputs, targets)
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                else:  # Volumetric dataset: (volumes,)
                    volumes = batch_data[0].to(self.device)
                    outputs = model(volumes)
                    # For volumetric, you might need a different loss computation
                    loss = loss_fn(outputs, volumes)  # Reconstruction loss example
                
                if is_training:
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
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
            plt.plot(epochs, val_losses, 's-', label='Validation Loss', alpha=0.8)
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

# Test the CModelTrainer class
if __name__ == "__main__":
    print("Testing CModelTrainer with real data...")
    
    # First, generate processed data using CDataProcessor
    from CDataProcessor import CDataProcessor
    
    # Configuration for data processing
    data_config = {
        'dataset_paths': {
            'raw': 'data/raw',
            'processed': 'data/processed'
        },
        'train_val_split': 0.7,  # 70% train, 30% validation
        'point_cloud_params': {
            'radius': 0.02,
            'sigma': 0.01,
            'mu': 0.0,
            'n_gaussian': 5,  # Fewer samples for faster testing
            'n_uniform': 1000  # Fewer samples for faster testing
        },
        'volume_processor_params': {
            'device': 'cpu',  # Use CPU for compatibility
            'resolution': 16   # Smaller resolution for faster testing
        }
    }
    
    print("Step 1: Processing data with CDataProcessor...")
    try:
        processor = CDataProcessor(data_config)
        print(f"Found {len(processor.mesh_files)} mesh files: {processor.mesh_files}")
        
        # Generate SDF dataset
        dataset_info = processor.generate_sdf_dataset(
            z_dim=64,  # Smaller latent dimension for testing
            latent_mean=0.0,
            latent_sd=0.01
        )
        print(f"✓ Data processing complete!")
        print(f"  Train files: {len(dataset_info['train_files'])}")
        print(f"  Val files: {len(dataset_info['val_files'])}")
        
    except Exception as e:
        print(f"Error in data processing: {e}")
        print("Make sure you have mesh files in data/raw/")
        exit(1)
    
    print("\nStep 2: Setting up model training...")
    
    # Configuration for model training
    trainer_config = {
        'processed_data_path': 'data/processed',
        'dataset_type': 'sdf',
        'num_epochs': 3,  # Fewer epochs for testing
        'batch_size': 2,   # Smaller batch size for testing
        'learning_rate': 0.001,
        'max_points': 10000,
        'optimizer': 'adam',
        'loss_function': 'mse',
        'sdf_delta': 1.0,
        'latent_sd': 0.01
    }
    
    trainer = CModelTrainer(trainer_config)
    
    # Create a config for the model
    model_config = {
        'z_dim': 64,  # Match the dataset latent dimension
        'layer_size': 128,  # Smaller network for testing
        'coord_dim': 3,  # 3D coordinates
        'dropout_p': 0.2
    }
    
    try:
        model = DeepSDF(model_config)
        print("✓ Model created successfully")
        
        print("\nStep 3: Starting training...")
        trained_model, report = trainer.train_and_validate(model, dataset_info)
        
        print("\n" + "="*50)
        print("Training Completed!")
        print("="*50)
        if "error" in report:
            print(f"Training failed: {report['error']}")
        else:
            print("✓ Training successful!")
            print(f"  Total epochs: {report['total_epochs']}")
            print(f"  Best validation loss: {report['best_val_loss']:.6f}")
            print(f"  Final train loss: {report['final_train_loss']:.6f}")
            print(f"  Final validation loss: {report['final_val_loss']:.6f}")
            print(f"  Total training time: {report['total_training_time']:.2f}s")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()