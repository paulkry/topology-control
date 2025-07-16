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
        
        print(f"Trainer initialized - Device: {self.device}")
    
    def train_and_validate(self, model):
        """
        Main training loop with validation.
        
        Args:
            model: PyTorch model to train
            
        Returns:
            tuple: (trained_model, training_report)
        """
        print(f"Starting training on {self.device}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Create datasets and data loaders
        try:
            train_dataset = ShapeDataset(self.processed_data_path, split='train', max_points=self.max_points)
            val_dataset = ShapeDataset(self.processed_data_path, split='val', max_points=self.max_points)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return model, {"error": f"Dataset loading failed: {e}"}
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Setup loss function
        if self.loss_function.lower() == 'mse':
            criterion = nn.MSELoss()
        elif self.loss_function.lower() == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()  # Default
        
        # Setup optimizer
        if self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)  # Default
        
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
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Ensure output and target shapes match
                if outputs.shape != targets.shape:
                    # If targets need reshaping to match outputs
                    if targets.numel() == outputs.numel():
                        targets = targets.view(outputs.shape)
                    else:
                        print(f"Warning: Shape mismatch - outputs: {outputs.shape}, targets: {targets.shape}")
                        continue
                
                loss = loss_fn(outputs, targets)
                
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
            "training_history": history
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
            plt.title('Training and Validation Loss')
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
     