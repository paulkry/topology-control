import torch
import matplotlib.pyplot as plt
import base64
from io import BytesIO

torch.set_default_dtype(torch.float64)

class CModelTrainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.optimizer = config.get('optimizer', 'adamw')
        self.loss_function = config.get('loss_function', 'mse')
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 1024)
        self.learning_rate = config.get('learning_rate', 1e-3)

        # Data paths
        self.processed_data_path = config.get('processed_data_path', 'data/processed')
        

    def train_and_validate(self, model):
        # Fill up with training and validation logic
        pass


    def _run_epoch(self, data_loader, model, loss_fn, optimizer, is_training, epoch=1):
        if hasattr(loss_fn, 'set_epoch'):
            loss_fn.set_epoch(epoch)

        total_loss = 0.0
        batch_count = 0
        
        # Get task from config
        task = self.config.get('task', 'signed_distance')
        
        for batch_idx, (features, intersection_status, intersection_volume) in enumerate(data_loader):
            features = features.to(self.device)
            intersection_status = intersection_status.to(self.device)
            intersection_volume = intersection_volume.to(self.device)
            
            # Prepare targets based on task
            task_config = self.task_configs[task]
            targets = task_config['target_builder'](intersection_status, intersection_volume)
            
            # Forward pass
            predictions = model(features)
            predictions = task_config['prediction_extractor'](predictions)
            
            
            # Standard loss function
            loss = loss_fn(predictions, targets)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
                print(f"⚠️ Invalid loss at batch {batch_idx}: {loss.item()}, skipping")
                continue
            
            if is_training:
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # option
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Add debugging if needed
        
        return total_loss / batch_count if batch_count > 0 else None

    def _test_invariance(self, model, sample_features):
        """Test invariance on a small sample"""
        import GeometryUtils as gu
        
        model.eval()
        original_pred = model(sample_features)
        swapped_features = gu.swap_tetrahedrons(sample_features)
        swapped_pred = model(swapped_features)
        
        diff = torch.abs(original_pred - swapped_pred).mean().item()
        print(f"   Mean prediction difference after swap: {diff:.6f}")
        
        model.train()


    def _create_report(self, history, best_val_loss):
        """Create training report"""
        if not history:
            return {"error": "No training history"}
        
        total_time = sum(h['epoch_time'] for h in history)
        return {
            "total_epochs": len(history),
            "best_val_loss": best_val_loss,
            "final_val_loss": history[-1]['val_loss'],
            "total_training_time": total_time,
            "training_history": history
        }
    
    def _create_loss_plot(self, history):
        """Create loss curve visualization"""
        if not history:
            return None
        
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
