"""
Model training pipeline for 3D shape classification.
Handles training, validation, and model checkpointing with device management and memory optimization.
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

# --------------------------------------
# Datasets
# --------------------------------------  
class VolumeSDFDataset(Dataset):
    """
    Volume-aware SDF Dataset class that includes shape volume information.
    Extends SDFDataset functionality with volume data for topology-aware learning.
    Raises error if volume data is not available.
    """

    def __init__(self, dataset_info, split='train', fix_seed=False, volume_coords=None, device='cpu'):
        """
        Initialize volume-aware dataset from CDataProcessor output.
        """
        self.split = split
        self.fix_seed = fix_seed
        self.device = device  # Store the device
        self.dataset_params = dataset_info['dataset_params']
        
        # Get file list for this split
        self.files = dataset_info[f'{split}_files']
        
        # FIXED: Initialize latent vectors properly as leaf tensors
        self.z_dim = self.dataset_params['z_dim']
        self.latent_mean = self.dataset_params['latent_mean']
        self.latent_sd = self.dataset_params['latent_sd']
        
        self.latent_vectors = self._initialize_latent_vectors(device=device)  # Pass device here
        
        # Store volume coordinates for additional sampling if needed
        self.volume_coords = volume_coords or dataset_info.get('volume_coords')
        
        # Extract shape volumes - raise error if not available
        self.shape_volumes = self._extract_shape_volumes(dataset_info)
        
        print(f"VolumeSDFDataset initialized with {len(self.files)} samples and volume data")

    def _initialize_latent_vectors(self, device='cpu'):
        """
        Initialize latent vectors for DeepSDF training.
        Each shape gets its own latent code that will be optimized during training.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            torch.nn.Parameter: Latent vectors of shape [num_shapes, z_dim] with requires_grad=True
        """
        num_shapes = len(self.files)
        
        if self.fix_seed:
            torch.manual_seed(42)
        
        # FIXED: Create Parameter directly on the target device
        latent_vectors = torch.nn.Parameter(
            torch.normal(
                mean=self.latent_mean, 
                std=self.latent_sd, 
                size=(num_shapes, self.z_dim),
                device=device  # Create directly on target device
            )
        )
        
        print(f"✓ Initialized {num_shapes} latent vectors of dim {self.z_dim} on {device}")
        print(f"  Mean: {self.latent_mean}, Std: {self.latent_sd}")
        print(f"  Requires grad: {latent_vectors.requires_grad}")
        print(f"  Is leaf: {latent_vectors.is_leaf}")
        print(f"  Device: {latent_vectors.device}")
        
        return latent_vectors

    def _extract_shape_volumes(self, dataset_info):
        """
        Extract shape volume information from dataset_info for the current split.
        
        Parameters:
            dataset_info (dict): Dataset information containing volume data
            
        Returns:
            torch.Tensor: Tensor of shape volumes for each sample in current split
            
        Raises:
            ValueError: If volume data is not found in dataset_info
        """
        volumes = None
        
        # Check for split-specific volumes first (preferred approach)
        split_volumes_key = f'{self.split}_volumes'  # 'train_volumes' or 'val_volumes'
        
        if split_volumes_key in dataset_info:
            volumes = dataset_info[split_volumes_key]
            print(f"✓ Found {self.split} volumes: {len(volumes)} volumes for {len(self.files)} files")
        
        # Fallback: extract volumes from file info for current split
        elif f'{self.split}_files' in dataset_info:
            split_files = dataset_info[f'{self.split}_files']
            volumes = []
            
            for file_info in split_files:
                if isinstance(file_info, dict) and 'volume' in file_info:
                    volumes.append(file_info['volume'])
                else:
                    raise ValueError(f"File info missing volume data for {self.split} split: {file_info}")
            
            print(f"✓ Extracted {len(volumes)} volumes from {self.split}_files")
        
        # Last resort: use all volumes and slice for current split
        elif 'shape_volumes' in dataset_info:
            all_volumes = dataset_info['shape_volumes']
            
            # Get file indices for current split
            if self.split == 'train':
                # Assume first N files are train files
                num_train = len(dataset_info.get('train_files', []))
                volumes = all_volumes[:num_train]
            elif self.split == 'val':
                # Assume remaining files are val files
                num_train = len(dataset_info.get('train_files', []))
                volumes = all_volumes[num_train:]
            else:
                volumes = all_volumes
            
            print(f"✓ Sliced {len(volumes)} volumes from shape_volumes for {self.split} split")
        
        else:
            raise ValueError(
                f"No volume data found in dataset_info. "
                f"Expected one of: '{split_volumes_key}', '{self.split}_files' with volume info, or 'shape_volumes'"
            )
        
        # Convert to tensor
        if isinstance(volumes, (list, tuple)):
            volumes = torch.tensor(volumes, dtype=torch.float32)
        elif isinstance(volumes, np.ndarray):
            volumes = torch.from_numpy(volumes).float()
        elif not isinstance(volumes, torch.Tensor):
            volumes = torch.tensor(volumes, dtype=torch.float32)
        
        # Verify we have the right number of volumes
        if len(volumes) != len(self.files):
            raise ValueError(
                f"Volume count mismatch for {self.split} split: "
                f"{len(volumes)} volumes for {len(self.files)} files. "
                f"Files: {[f['mesh_name'] if isinstance(f, dict) else f for f in self.files[:3]]}... "
                f"Volumes shape: {volumes.shape if hasattr(volumes, 'shape') else type(volumes)}"
            )
        
        print(f"✓ Successfully loaded {len(volumes)} volumes for {self.split} split")
        return volumes
    
    def __getitem__(self, idx):
        if self.fix_seed:
            np.random.seed(42)
        
        # Load preprocessed data from PointCloudProcessor
        file_info = self.files[idx]
        sampled_points = np.load(file_info['points_file'])
        sdf_values = np.load(file_info['distances_file'])
        
        # Get shape volume for this sample - now available in file_info
        if 'volume' in file_info:
            shape_volume = file_info['volume']
        else:
            # Fallback to indexed volumes
            shape_volume = self.shape_volumes[idx]
        
        # Optional: Add additional random sampling from volume coords
        if self.volume_coords is not None:
            num_volume_samples = min(1000, len(self.volume_coords))
            volume_indices = np.random.choice(len(self.volume_coords), num_volume_samples, replace=False)
            # volume_points = self.volume_coords[volume_indices]
        
        # Convert to tensors
        sampled_points = torch.tensor(sampled_points, dtype=torch.float32)
        sdf_values = torch.tensor(sdf_values, dtype=torch.float32)
        shape_volume = torch.tensor(shape_volume, dtype=torch.float32)
        
        # Return points, latent vector, sdf values, and shape volume
        return sampled_points, self.latent_vectors[idx], sdf_values, shape_volume

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences with volume data"""
        # Find minimum sample length, but ensure it's at least 1
        sample_lengths = [len(i[0]) for i in batch]
        min_sample_len = max(1, min(sample_lengths))
        
        # Also check if we have enough samples for meaningful training
        if min_sample_len < 100:  # Adjust threshold as needed
            print(f"Warning: Very small sample size detected: {min_sample_len}")
        
        x_vals = [i[0][:min_sample_len].unsqueeze(0) for i in batch]
        latent_vecs = [i[1].unsqueeze(0) for i in batch]
        y_vals = [i[2][:min_sample_len].unsqueeze(0) for i in batch]
        volumes = [i[3].unsqueeze(0) for i in batch]  # Shape volumes
        
        return (torch.cat(x_vals, dim=0), 
                torch.cat(latent_vecs, dim=0), 
                torch.cat(y_vals, dim=0),
                torch.cat(volumes, dim=0))
    
    def __len__(self):
        return len(self.files)

class SDFDataset(Dataset):
    """
    SDF Dataset class that uses preprocessed data from CDataProcessor.
    Compatible with DeepSDF training pipeline - supports latent code learning.
    """
    
    def __init__(self, dataset_info, split='train', fix_seed=False, volume_coords=None, device='cpu'):
        """
        Initialize dataset from CDataProcessor output.
        
        Parameters:
            dataset_info (dict): Output from CDataProcessor.generate_sdf_dataset()
            split (str): 'train' or 'val'
            fix_seed (bool): Whether to fix random seed for reproducibility
            volume_coords (torch.Tensor): Volume coordinates for additional sampling
            device (str): Device to create tensors on
        """
        self.split = split
        self.fix_seed = fix_seed
        self.device = device
        self.dataset_params = dataset_info['dataset_params']
        
        # Get file list for this split
        self.files = dataset_info[f'{split}_files']
        
        # Initialize latent vectors for each mesh - FIXED
        self.z_dim = self.dataset_params['z_dim']
        self.latent_mean = self.dataset_params['latent_mean']
        self.latent_sd = self.dataset_params['latent_sd']
        
        self.latent_vectors = self._initialize_latent_vectors(device=device)
        
        # Store volume coordinates for additional sampling if needed
        self.volume_coords = volume_coords or dataset_info.get('volume_coords')

    def _initialize_latent_vectors(self, device='cpu'):
        """
        Initialize latent vectors for DeepSDF training.
        Each shape gets its own latent code that will be optimized during training.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            torch.nn.Parameter: Latent vectors of shape [num_shapes, z_dim] with requires_grad=True
        """
        num_shapes = len(self.files)
        
        if self.fix_seed:
            torch.manual_seed(42)
        
        # FIXED: Create Parameter directly on the target device
        latent_vectors = torch.nn.Parameter(
            torch.normal(
                mean=self.latent_mean, 
                std=self.latent_sd, 
                size=(num_shapes, self.z_dim),
                device=device  # Create directly on target device
            )
        )
        
        print(f"✓ Initialized {num_shapes} latent vectors of dim {self.z_dim} on {device}")
        print(f"  Mean: {self.latent_mean}, Std: {self.latent_sd}")
        print(f"  Requires grad: {latent_vectors.requires_grad}")
        print(f"  Is leaf: {latent_vectors.is_leaf}")
        print(f"  Device: {latent_vectors.device}")
        
        return latent_vectors
    
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
        latent_vecs = [i[1].unsqueeze(0) for i in batch]
        y_vals = [i[2][:min_sample_len].unsqueeze(0) for i in batch]
        
        return (torch.cat(x_vals, dim=0), 
                torch.cat(latent_vecs, dim=0), 
                torch.cat(y_vals, dim=0))
    
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

# --------------------------------------
# Loss Functions
# --------------------------------------

class CombinedSDFVolumeLoss(torch.nn.Module):
    """
    Combined loss for SDF and volume prediction.
    50% SDF loss (L1) + 50% Volume loss (MAE)
    """
    
    def __init__(self, sdf_weight=0.5, volume_weight=0.5, delta=1.0):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.volume_weight = volume_weight
        self.delta = delta  # Clamp threshold for SDF loss
        
    def forward(self, predictions, targets):
        """
        Calculate combined loss
        
        Args:
            predictions: dict with 'sdf' and 'volume' keys
            targets: dict with 'sdf' and 'volume' keys
        """
        # SDF Loss (L1 with clamping, similar to DeepSDF)
        sdf_pred = predictions['sdf']
        sdf_target = targets['sdf']
        sdf_loss = torch.clamp(torch.abs(sdf_pred - sdf_target), max=self.delta).mean()
        
        # Volume Loss (MAE) - CHANGED from MSE to MAE
        volume_pred = predictions['volume']
        volume_target = targets['volume']
        volume_loss = torch.nn.functional.l1_loss(volume_pred, volume_target)  # MAE instead of MSE
        
        # Combined loss
        total_loss = self.sdf_weight * sdf_loss + self.volume_weight * volume_loss
        
        return {
            'total_loss': total_loss,
            'sdf_loss': sdf_loss,
            'volume_loss': volume_loss
        }

class CModelTrainer:
    """Model trainer with memory management and device fallback capabilities."""
    
    def __init__(self, config):
        """Initialize trainer with device configuration and memory management."""
        self.config = config
        
        # Device configuration with fallback
        self.device = self._configure_device(
            config.get('device', 'auto'),
            config.get('allow_cpu_fallback', True)
        )
        
        # Memory management settings
        self.memory_efficient = config.get('memory_efficient', True)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_points_per_batch = config.get('max_points_per_batch', None)
        
        print(f"Trainer initialized - Device: {self.device}")
        if self.memory_efficient:
            print("✓ Memory efficient mode enabled")
        if self.gradient_accumulation_steps > 1:
            print(f"✓ Gradient accumulation: {self.gradient_accumulation_steps} steps")
        
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
        
        # Dataset type configuration
        self.dataset_type = config.get('dataset_type', 'shape')
        
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
        
        print(f"Training artifacts will be saved to: {self.save_dir}")

    def _configure_device(self, device_config, allow_fallback=True):
        """
        Configure device with fallback options for memory constraints.
        
        Args:
            device_config: Device configuration ('auto', 'cuda', 'cpu', etc.)
            allow_fallback: Whether to fall back to CPU on CUDA OOM
            
        Returns:
            torch.device: Configured device
        """
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ Auto-detected CUDA device: {device}")
                
                # Test CUDA memory availability
                if allow_fallback:
                    try:
                        # Test allocation to detect memory issues early
                        test_tensor = torch.randn(100, 100, device=device)
                        del test_tensor
                        torch.cuda.empty_cache()
                        print(f"✓ CUDA memory test passed")
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        print(f"⚠️  CUDA memory test failed: {e}")
                        print("⚠️  Falling back to CPU")
                        device = torch.device('cpu')
                
                return device
            else:
                print("⚠️  CUDA not available, using CPU")
                return torch.device('cpu')
        
        elif device_config == 'cpu':
            return torch.device('cpu')
        
        elif device_config.startswith('cuda'):
            if torch.cuda.is_available():
                return torch.device(device_config)
            else:
                if allow_fallback:
                    print(f"⚠️  {device_config} not available, falling back to CPU")
                    return torch.device('cpu')
                else:
                    raise RuntimeError(f"CUDA device {device_config} requested but not available")
        
        else:
            raise ValueError(f"Unknown device configuration: {device_config}")

    def _apply_memory_optimizations(self):
        """Apply memory optimization techniques."""
        if not self.memory_efficient:
            return
            
        print("🔧 Applying memory optimizations...")
        
        # Set memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("  ✓ Flash attention enabled")
        except:
            pass
        
        # Enable memory efficient inference
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            print("  ✓ CUDNN benchmark enabled")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✓ CUDA cache cleared")

    def train_and_validate(self, model, dataset_info=None):
        """
        Main training loop with memory management and device fallback.
        """
        try:
            return self._train_with_memory_management(model, dataset_info)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "CUDA out of memory" in str(e) and self.config.get('allow_cpu_fallback', True):
                print(f"\n❌ CUDA OOM Error: {e}")
                print("🔄 Attempting fallback to CPU...")
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Switch to CPU
                original_device = self.device
                self.device = torch.device('cpu')
                
                print(f"✓ Switched from {original_device} to {self.device}")
                
                try:
                    return self._train_with_memory_management(model, dataset_info)
                except Exception as fallback_error:
                    print(f"❌ CPU fallback also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e

    def _train_with_memory_management(self, model, dataset_info=None):
            """
            Training implementation with memory management.
            """
            print(f"Starting training on {self.device}")
            
            # Apply memory optimizations
            self._apply_memory_optimizations()
            
            # Move model to device with error handling
            try:
                model = model.to(self.device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"❌ Failed to move model to {self.device}: {e}")
                    raise e
                else:
                    raise e
            
            # Create datasets and data loaders
            try:
                train_dataset, val_dataset = self.create_datasets(dataset_info)
                train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset)
            except Exception as e:
                print(f"Error loading datasets: {e}")
                return model, {"error": f"Dataset loading failed: {e}"}
            
            # Setup loss function and optimizer
            criterion = self._get_loss_function()
            
            # For SDF datasets, get latent vectors for dual learning rates
            latent_vectors = None
            if self.dataset_type in ['sdf', 'volumesdf']:
                if hasattr(train_dataset, 'latent_vectors'):
                    latent_vectors = train_dataset.latent_vectors
                    optimizer = self._get_optimizer(model, latent_vectors)
                    print(f"✓ {self.dataset_type.upper()} training setup complete with dual learning rates")
                else:
                    print(f"⚠️  No latent vectors found in {self.dataset_type.upper()} dataset, using single learning rate")
                    optimizer = self._get_optimizer(model)
            else:
                optimizer = self._get_optimizer(model)
            
            # Training history and tracking
            best_val_loss = float('inf')
            best_model_saved = False  # Track if we saved the best model
            history = []
            
            print(f"Training for {self.epochs} epochs...")
            print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
            print(f"Batch size: {self.batch_size}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
            
            # Training loop with memory management
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                
                try:
                    # Training phase
                    if self.dataset_type == 'volumesdf':
                        train_metrics = self._run_epoch_with_memory_management(
                            train_loader, model, criterion, optimizer, True, epoch)
                        train_loss = train_metrics['total_loss']
                    else:
                        train_loss = self._run_epoch_with_memory_management(
                            train_loader, model, criterion, optimizer, True, epoch)
                    
                    # Validation phase
                    if self.dataset_type == 'volumesdf':
                        val_metrics = self._run_epoch_with_memory_management(
                            val_loader, model, criterion, optimizer, False, epoch)
                        val_loss = val_metrics['total_loss']
                    else:
                        val_loss = self._run_epoch_with_memory_management(
                            val_loader, model, criterion, optimizer, False, epoch)
                    
                    epoch_time = time.time() - epoch_start_time
                    
                    # Log progress
                    if self.dataset_type == 'volumesdf':
                        print(f"Epoch {epoch:3d}/{self.epochs} | "
                            f"Train: {train_loss:.6f} (SDF: {train_metrics['sdf_loss']:.6f}, Vol: {train_metrics['volume_loss']:.6f}) | "
                            f"Val: {val_loss:.6f} (SDF: {val_metrics['sdf_loss']:.6f}, Vol: {val_metrics['volume_loss']:.6f}) | "
                            f"Time: {epoch_time:.2f}s")
                        
                        epoch_info = {
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'train_sdf_loss': train_metrics['sdf_loss'],
                            'train_volume_loss': train_metrics['volume_loss'],
                            'val_loss': val_loss,
                            'val_sdf_loss': val_metrics['sdf_loss'],
                            'val_volume_loss': val_metrics['volume_loss'],
                            'epoch_time': epoch_time
                        }
                    else:
                        print(f"Epoch {epoch:3d}/{self.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.2f}s")
                        
                        epoch_info = {
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'epoch_time': epoch_time
                        }
                    
                    history.append(epoch_info)
                    
                    # Save checkpoints
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        print(f"🎯 New best validation loss: {best_val_loss:.6f}")
                    
                    # ENHANCED: Always save best model + periodic checkpoints
                    should_save = False
                    save_reason = ""
                    
                    if is_best:
                        should_save = True
                        save_reason = "best model"
                        best_model_saved = True
                    elif epoch % self.save_frequency == 0:
                        should_save = True
                        save_reason = f"periodic (every {self.save_frequency} epochs)"
                    elif epoch == self.epochs:
                        should_save = True
                        save_reason = "final epoch"
                    
                    # Override save_best_only for the best model
                    if should_save and (is_best or not self.save_best_only):
                        checkpoint_path = self.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            is_best=is_best,
                            latent_vectors=latent_vectors,
                            dataset_info=dataset_info
                        )
                        print(f"💾 Saved checkpoint ({save_reason}): {checkpoint_path}")
                    
                    # Memory cleanup after each epoch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if "CUDA out of memory" in str(e):
                        print(f"❌ CUDA OOM at epoch {epoch}: {e}")
                        if self.config.get('allow_cpu_fallback', True) and self.device != torch.device('cpu'):
                            print("🔄 Switching to CPU mid-training...")
                            
                            # Switch to CPU
                            self.device = torch.device('cpu')
                            model = model.cpu()
                            
                            # Recreate datasets with CPU device
                            train_dataset, val_dataset = self.create_datasets(dataset_info)
                            train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset)
                            
                            # Recreate optimizer with new latent vectors
                            if latent_vectors is not None:
                                latent_vectors = train_dataset.latent_vectors
                                optimizer = self._get_optimizer(model, latent_vectors)
                            else:
                                optimizer = self._get_optimizer(model)
                            
                            print(f"✓ Successfully switched to CPU, continuing from epoch {epoch}")
                            continue
                        else:
                            raise e
                    else:
                        raise e
            
            # ENHANCED: Ensure we have a best model saved
            if not best_model_saved and history:
                print("⚠️  No best model was saved during training, saving final model as best...")
                final_checkpoint_path = self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=self.epochs,
                    train_loss=history[-1]['train_loss'],
                    val_loss=history[-1]['val_loss'],
                    is_best=True,  # Force save as best
                    latent_vectors=latent_vectors,
                    dataset_info=dataset_info
                )
                print(f"💾 Saved final model as best: {final_checkpoint_path}")
            
            # Create training report
            training_report = self._create_report(history, best_val_loss)
            
            # Add model save information to report
            training_report['best_model_saved'] = best_model_saved or bool(history)
            training_report['model_save_directory'] = self.save_dir
            
            print(f"\n✅ Training completed!")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            print(f"   Total training time: {training_report.get('total_training_time', 0):.2f}s")
            print(f"   Final device: {self.device}")
            print(f"   Model saved: {'✓' if training_report['best_model_saved'] else '✗'}")
            if training_report['best_model_saved']:
                print(f"   Save directory: {self.save_dir}")
            
            return model, training_report

    def create_datasets(self, dataset_info=None):
        """
        Create datasets based on configuration.
        
        Args:
            dataset_info: Optional dataset info from CDataProcessor (for SDF datasets)
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        
        if self.dataset_type == 'sdf':
            train_dataset = SDFDataset(dataset_info, split='train', device=self.device)
            val_dataset = SDFDataset(dataset_info, split='val', device=self.device)
        elif self.dataset_type == 'volumesdf':
            train_dataset = VolumeSDFDataset(dataset_info, split='train', device=self.device)
            val_dataset = VolumeSDFDataset(dataset_info, split='val', device=self.device)
        else:
            # Default to ShapeDataset for traditional point cloud learning
            train_dataset = ShapeDataset(self.processed_data_path, split='train', max_points=self.max_points)
            val_dataset = ShapeDataset(self.processed_data_path, split='val', max_points=self.max_points)
        
        return train_dataset, val_dataset

    def create_data_loaders(self, train_dataset, val_dataset):
        """Create data loaders from datasets."""
        
        # Get the appropriate collate function based on dataset type
        if isinstance(train_dataset, VolumeSDFDataset):
            collate_fn = VolumeSDFDataset.collate_fn
            print("✓ Using VolumeSDFDataset collate function")
        elif isinstance(train_dataset, SDFDataset):
            collate_fn = SDFDataset.collate_fn
            print("✓ Using SDFDataset collate function")
        else:
            collate_fn = None
            print("✓ Using default PyTorch collate function")
        
        # Test the collate function if it's a custom one
        if collate_fn is not None:
            try:
                # Get a few samples to test collation
                sample1 = train_dataset[0]
                sample2 = train_dataset[1] if len(train_dataset) > 1 else train_dataset[0]
                test_batch = [sample1, sample2]
                
                # Test collate function
                result = collate_fn(test_batch)
                print(f"✓ Collate function test passed. Output shapes:")
                if isinstance(result, tuple):
                    for i, tensor in enumerate(result):
                        print(f"    Output {i}: {tensor.shape}")
                
            except Exception as e:
                print(f"⚠️  Collate function test failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to default collation
                collate_fn = None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Keep at 0 for debugging
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Keep at 0 for debugging
        )
        
        return train_loader, val_loader

    def _run_epoch_with_memory_management(self, data_loader, model, loss_fn, optimizer, is_training, epoch=1):
        """
        Run epoch with memory management and gradient accumulation.
        """
        if is_training:
            model.train()
        else:
            model.eval()
        
        total_loss = 0.0
        sdf_loss_sum = 0.0
        volume_loss_sum = 0.0
        num_batches = 0
        
        # For gradient accumulation
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, batch_data in enumerate(data_loader):
                if self.dataset_type == 'volumesdf':
                    # Handle VolumeSDFDataset: (coords, latents, sdfs, volumes)
                    if len(batch_data) != 4:
                        raise ValueError(f"Expected VolumeSDFDataset with 4 elements, got {len(batch_data)} elements")
                    
                    coords, latents, sdf_targets, volume_targets = batch_data
                    coords = coords.to(self.device)
                    latents = latents.to(self.device)
                    sdf_targets = sdf_targets.to(self.device)
                    volume_targets = volume_targets.to(self.device)
                    
                    # Limit points per batch if specified
                    if self.max_points_per_batch and coords.shape[1] > self.max_points_per_batch:
                        indices = torch.randperm(coords.shape[1])[:self.max_points_per_batch]
                        coords = coords[:, indices, :]
                        sdf_targets = sdf_targets[:, indices]
                    
                    # Forward pass through VolumeDeepSDF
                    outputs = model(latents, coords)
                    
                    # Prepare targets for combined loss
                    targets = {
                        'sdf': sdf_targets,
                        'volume': volume_targets
                    }
                    
                    # Calculate combined loss
                    loss_dict = loss_fn(outputs, targets)
                    loss = loss_dict['total_loss']
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Track individual losses
                    sdf_loss_sum += loss_dict['sdf_loss'].item()
                    volume_loss_sum += loss_dict['volume_loss'].item()
                    
                elif self.dataset_type == 'sdf':
                    # Handle regular SDFDataset: (coords, latents, sdfs)
                    if len(batch_data) != 3:
                        raise ValueError(f"Expected SDFDataset with 3 elements, got {len(batch_data)} elements")
                    
                    coords, latents, sdf_targets = batch_data
                    coords = coords.to(self.device)
                    latents = latents.to(self.device)
                    sdf_targets = sdf_targets.to(self.device)
                    
                    # Limit points per batch if specified
                    if self.max_points_per_batch and coords.shape[1] > self.max_points_per_batch:
                        indices = torch.randperm(coords.shape[1])[:self.max_points_per_batch]
                        coords = coords[:, indices, :]
                        sdf_targets = sdf_targets[:, indices]
                    
                    # Forward pass through DeepSDF
                    predicted_sdfs = model(latents, coords)
                    
                    # Calculate SDF loss (with latent regularization)
                    loss = loss_fn(predicted_sdfs, sdf_targets, latents)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    
                    sdf_loss_sum += loss.item()
                    
                else:
                    # Handle regular shape dataset
                    coords, targets = batch_data
                    coords = coords.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(coords)
                    loss = loss_fn(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if is_training:
                    loss.backward()
                    
                    # Gradient accumulation step
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * self.gradient_accumulation_steps  # Unscale for logging
                num_batches += 1
                
                # Memory cleanup every few batches
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final gradient step if needed
        if is_training and num_batches % self.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        
        # Return appropriate metrics based on dataset type
        if self.dataset_type == 'volumesdf':
            return {
                'total_loss': avg_loss,
                'sdf_loss': sdf_loss_sum / num_batches,
                'volume_loss': volume_loss_sum / num_batches
            }
        else:
            return avg_loss  # Single loss value for other dataset types

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
            
        Returns:
            str: Path to saved checkpoint file
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
            'device': str(self.device),
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
            'model_architecture': type(model).__name__
        }
        
        # Add model architecture info if available
        if hasattr(model, 'get_architecture_info'):
            checkpoint['architecture_info'] = model.get_architecture_info()
        
        # Add latent vectors for SDF models
        if latent_vectors is not None:
            checkpoint['latent_vectors'] = latent_vectors.detach().cpu()
            checkpoint['latent_vectors_shape'] = list(latent_vectors.shape)
        
        # Add dataset info for reproducibility
        if dataset_info is not None:
            # Only save serializable parts of dataset_info
            serializable_dataset_info = {}
            for key, value in dataset_info.items():
                if key not in ['volume_coords']:  # Skip large arrays
                    try:
                        import json
                        json.dumps(value)  # Test if serializable
                        serializable_dataset_info[key] = value
                    except:
                        pass  # Skip non-serializable values
            checkpoint['dataset_info'] = serializable_dataset_info
        
        # Create filename with .pth extension
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if is_best:
            filename = f'best_model_{self.dataset_type}_{timestamp}.pth'
        else:
            filename = f'checkpoint_{self.dataset_type}_epoch_{epoch}_{timestamp}.pth'
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Save checkpoint
        try:
            # Ensure directory exists
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save the main checkpoint
            torch.save(checkpoint, filepath)
            print(f"💾 Saved {'best model' if is_best else 'checkpoint'}: {filename}")
            
            # Also save a 'latest' and 'best' symlink for easy access
            if is_best:
                best_link = os.path.join(self.save_dir, f'best_model_{self.dataset_type}.pth')
                try:
                    if os.path.exists(best_link):
                        os.remove(best_link)
                    # Create a copy instead of symlink for better compatibility
                    torch.save(checkpoint, best_link)
                    print(f"💾 Saved best model link: best_model_{self.dataset_type}.pth")
                except Exception as e:
                    print(f"Warning: Could not create best model link: {e}")
            
            # Always update latest checkpoint
            latest_link = os.path.join(self.save_dir, f'latest_{self.dataset_type}.pth')
            try:
                torch.save(checkpoint, latest_link)
            except Exception as e:
                print(f"Warning: Could not create latest checkpoint link: {e}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving checkpoint to {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
    
    def _get_loss_function(self):
        """Factory method to get appropriate loss function based on dataset type."""
        if self.dataset_type == 'volumesdf':
            return self._get_volume_sdf_loss()
        elif self.dataset_type == 'sdf':
            return self._get_sdf_loss()
        else:
            # Default loss for shape datasets
            return nn.MSELoss() if self.loss_function.lower() == 'mse' else nn.L1Loss()

    def _get_volume_sdf_loss(self):
        """Get combined loss function for VolumeDeepSDF training."""
        sdf_weight = self.config.get('sdf_weight', 0.5)
        volume_weight = self.config.get('volume_weight', 0.5)
        delta = self.config.get('sdf_delta', 1.0)
        
        return CombinedSDFVolumeLoss(sdf_weight, volume_weight, delta)
    
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
            # Ensure latent vectors are leaf tensors
            if not latent_vectors.is_leaf:
                raise ValueError(f"Latent vectors must be leaf tensors for optimization. "
                            f"Current: requires_grad={latent_vectors.requires_grad}, "
                            f"is_leaf={latent_vectors.is_leaf}")
            
            # Use separate learning rates for SDF training
            lr_net = self.config.get('network_learning_rate', 0.001)
            lr_latent = self.config.get('latent_learning_rate', 0.01)
            
            print(f"✓ Setting up dual learning rates: network={lr_net}, latent={lr_latent}")
            
            if self.optimizer_name.lower() == 'adam':
                optimizer = optim.Adam([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": [latent_vectors], "lr": lr_latent}  # Wrap in list
                ])
            elif self.optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": [latent_vectors], "lr": lr_latent}  # Wrap in list
                ])
            else:
                # Default to Adam for SDF training
                optimizer = optim.Adam([
                    {"params": model.parameters(), "lr": lr_net},
                    {"params": [latent_vectors], "lr": lr_latent}  # Wrap in list
                ])
            
            return optimizer
        
        else:
            # Single learning rate for traditional training
            params = list(model.parameters())
            lr = self.network_learning_rate
            
            if self.optimizer_name.lower() == 'adam':
                return optim.Adam(params, lr=lr)
            elif self.optimizer_name.lower() == 'adamw':
                return optim.AdamW(params, lr=lr)
            elif self.optimizer_name.lower() == 'sgd':
                return optim.SGD(params, lr=lr, momentum=0.9)
            else:
                return optim.Adam(params, lr=lr)  # Default

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
            "dataset_type": self.dataset_type,
            "device_used": str(self.device),
            "memory_efficient": self.memory_efficient,
            "gradient_accumulation_steps": self.gradient_accumulation_steps
        }
        
        # Add loss plot
        loss_plot = self._create_loss_plot(history)
        if loss_plot:
            report["loss_plot"] = loss_plot
        
        return report
    
    def _create_loss_plot(self, history):
        """Create comprehensive loss curve visualization for different dataset types."""
        if not history:
            return None
        
        try:
            epochs = [h['epoch'] for h in history]
            train_losses = [h['train_loss'] for h in history]
            val_losses = [h['val_loss'] for h in history]
            
            # Create figure with subplots based on dataset type
            if self.dataset_type == 'volumesdf':
                # For VolumeSDFDataset, show combined loss + individual SDF and Volume losses
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Combined loss plot
                ax1.plot(epochs, train_losses, 'o-', label='Training Loss', alpha=0.8, color='blue')
                ax1.plot(epochs, val_losses, 's-', label='Validation Loss', alpha=0.8, color='red')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Combined Loss')
                ax1.set_title('Combined SDF + Volume Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # SDF loss plot
                train_sdf_losses = [h['train_sdf_loss'] for h in history]
                val_sdf_losses = [h['val_sdf_loss'] for h in history]
                ax2.plot(epochs, train_sdf_losses, 'o-', label='Training SDF Loss', alpha=0.8, color='green')
                ax2.plot(epochs, val_sdf_losses, 's-', label='Validation SDF Loss', alpha=0.8, color='orange')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('SDF Loss')
                ax2.set_title('SDF Loss Component')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Volume loss plot
                train_vol_losses = [h['train_volume_loss'] for h in history]
                val_vol_losses = [h['val_volume_loss'] for h in history]
                ax3.plot(epochs, train_vol_losses, 'o-', label='Training Volume Loss', alpha=0.8, color='purple')
                ax3.plot(epochs, val_vol_losses, 's-', label='Validation Volume Loss', alpha=0.8, color='brown')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Volume Loss')
                ax3.set_title('Volume Loss Component')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Loss ratio plot (SDF vs Volume contribution)
                train_sdf_ratio = [sdf/(sdf+vol) for sdf, vol in zip(train_sdf_losses, train_vol_losses)]
                val_sdf_ratio = [sdf/(sdf+vol) for sdf, vol in zip(val_sdf_losses, val_vol_losses)]
                ax4.plot(epochs, train_sdf_ratio, 'o-', label='Train SDF Ratio', alpha=0.8, color='darkblue')
                ax4.plot(epochs, val_sdf_ratio, 's-', label='Val SDF Ratio', alpha=0.8, color='darkred')
                ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Equal Contribution')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('SDF Loss Ratio')
                ax4.set_title('SDF vs Volume Loss Balance')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1)
                
                plt.suptitle(f'VolumeDeepSDF Training Progress (Device: {self.device})', fontsize=16)
                
            else:
                # For regular SDF or other datasets, show standard loss plot
                plt.figure(figsize=(12, 8))
                
                # Main loss plot
                plt.subplot(2, 1, 1)
                plt.plot(epochs, train_losses, 'o-', label='Training Loss', alpha=0.8, color='blue', linewidth=2)
                plt.plot(epochs, val_losses, 's-', label='Validation Loss', alpha=0.8, color='red', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{self.dataset_type.upper()} Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Log scale plot for better visualization of convergence
                plt.subplot(2, 1, 2)
                plt.semilogy(epochs, train_losses, 'o-', label='Training Loss (log)', alpha=0.8, color='blue')
                plt.semilogy(epochs, val_losses, 's-', label='Validation Loss (log)', alpha=0.8, color='red')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (log scale)')
                plt.title('Loss Convergence (Log Scale)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'Training Progress - {self.dataset_type.upper()} (Device: {self.device})', fontsize=14)
            
            # Add training info text
            info_text = f"""Training Configuration:
• Device: {self.device}
• Memory Efficient: {self.memory_efficient}
• Gradient Accumulation: {self.gradient_accumulation_steps} steps
• Batch Size: {self.batch_size} (Effective: {self.batch_size * self.gradient_accumulation_steps})
• Best Val Loss: {min(val_losses):.6f}
• Final Val Loss: {val_losses[-1]:.6f}"""
            
            plt.figtext(0.02, 0.02, info_text, fontsize=8, verticalalignment='bottom', 
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        except Exception as e:
            print(f"Warning: Could not create loss plot: {e}")
            plt.close('all')  # Clean up any partial plots
            return None