import torch
torch.set_default_dtype(torch.float64)

# ============================================================================
# Models
# ============================================================================

class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[128, 64, 32], output_dim=1, activation='relu', dropout=0.1):
        """
        Simple Multi-Layer Perceptron for 3D point processing.
        
        Parameters:
            input_dim (int): Input dimension (default 3 for 3D points)
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension (default 1 for signed distance)
            activation (str): Activation function ('relu', 'tanh', 'leaky_relu')
            dropout (float): Dropout probability
        """
        super(SimpleMLP, self).__init__()
        
        # Store architecture info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = "signed_distance_prediction"
        
        # Build layer dimensions
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            # Linear layer
            self.layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Dropout (except for output layer)
            if i < len(layer_dims) - 2:
                self.dropouts.append(torch.nn.Dropout(dropout))
            else:
                self.dropouts.append(torch.nn.Identity())  # No dropout on output
        
        # Activation function
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Ensure input is the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Forward pass through layers
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            
            # Apply activation and dropout (except on output layer)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = dropout(x)
        
        return x
    
    def get_architecture_info(self):
        """Get information about the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_info = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear):
                layer_info.append({
                    'layer': i,
                    'type': 'Linear',
                    'input_size': layer.in_features,
                    'output_size': layer.out_features,
                    'parameters': layer.in_features * layer.out_features + layer.out_features
                })
        
        return {
            'architecture': 'SimpleMLP',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': layer_info,
            'task': self.task
        }
 
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
        
        # Validate architecture exists
        if architecture_name not in self.architecture_registry:
            available = list(self.architecture_registry.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. Available: {available}")
        
        # Get model parameters from config
        model_params = {
            'input_dim': self.config.get('input_dim', 3),
            'hidden_dims': self.config.get('hidden_dims', [128, 64, 32]),
            'output_dim': self.config.get('output_dim', 1),
            'activation': self.config.get('activation', 'relu'),
            'dropout': self.config.get('dropout', 0.1)
        }
        
        # Create model
        model_class = self.architecture_registry[architecture_name]
        model = model_class(**model_params)
        
        # Ensure model is in float64 precision
        model = model.double()
        
        return model

    
