
import torch

# ============================================================================
# Models
# ============================================================================

# Define your neural network architectures here
# For example, a simple MLP architecture can be defined as follows:

class SimpleMLP(torch.nn.Module):
    def __init__(self, config):
        pass
        # implement simple MLP architecture
 
# ============================================================================
# ARCHITECTURE MANAGER
# ============================================================================

class CArchitectureManager:
    """Main manager for neural network architectures"""
    def __init__(self, config):
        self.config = config
        
        # Registry of available architectures
        self.architecture_registry = {
            'mlp': SimpleMLP,
        }
    
    def get_model(self):
        """
        Create and return the configured model
        
        Returns:
            nn.Module: Configured neural network model
        """
        # Get architecture configuration
        architecture_name = self.config['model_name'].lower()
        architecture_config = self.config['architecture'].get(architecture_name, {})
        
        # Validate architecture exists
        if architecture_name not in self.architecture_registry:
            available = list(self.architecture_registry.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. Available: {available}")
        
        # Create model
        model_class = self.architecture_registry[architecture_name]
        model = model_class(architecture_config)
        
        # Ensure float64 precision
        model = model.double()
        for param in model.parameters():
            if param.dtype != torch.float64:
                param.data = param.data.double()
        
        return model

    
