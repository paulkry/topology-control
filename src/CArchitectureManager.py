import torch

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

class DeepSDF(torch.nn.Module):

    def __init__(self, config=None):
        super(DeepSDF, self).__init__()
        
        # Handle configuration
        if config is None:
            config = {}
        
        # Extract parameters with defaults (no hardcoded constants)
        self.z_dim = config.get('z_dim', 128)  # Latent vector dimension
        self.layer_size = config.get('layer_size', 256)  # Hidden layer dimension
        self.dropout_p = config.get('dropout_p', 0.2)
        self.coord_dim = config.get('coord_dim', 3)  # 3D coordinates
        
        # Store architecture info
        self.input_dim = self.z_dim + self.coord_dim
        self.output_dim = 1
        self.task = "signed_distance_prediction"
        
        # Build network layers
        input_dim = self.z_dim + self.coord_dim  # latent + coordinates
        
        # Ensure layer_size is large enough for the skip connection architecture
        min_layer_size = max(self.layer_size, input_dim + 32)  # Ensure sufficient capacity
        self.layer_size = min_layer_size
        
        self.input_layer = self.create_layer_block(input_dim, self.layer_size)
        self.layer2 = self.create_layer_block(self.layer_size, self.layer_size)
        self.layer3 = self.create_layer_block(self.layer_size, self.layer_size)
        self.layer4 = self.create_layer_block(self.layer_size, self.layer_size - input_dim)
        self.layer5 = self.create_layer_block(self.layer_size, self.layer_size)
        self.layer6 = self.create_layer_block(self.layer_size, self.layer_size)
        self.layer7 = self.create_layer_block(self.layer_size, self.layer_size)
        self.layer8 = torch.nn.Linear(self.layer_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def create_layer_block(self, input_size, output_size):
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p)
        )

    def forward(self, x=None, latent_vec=None, coords=None):
        """
        Forward pass - supports both unified input and separate inputs.
        
        Args:
            x: Unified input tensor of shape [batch_size, z_dim + 3] (alternative interface)
            latent_vec: Latent vector of shape [batch_size, z_dim] (DeepSDF interface)
            coords: Coordinates of shape [batch_size, num_coords, 3] (DeepSDF interface)
            
        Returns:
            torch.Tensor: SDF values
        """
        if x is not None:
            # Unified interface (like SimpleMLP)
            # Ensure input is the right shape
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            
            # If input is 2D [batch_size, features] treat as single point
            if x.dim() == 2:
                # Add coordinate dimension: [batch_size, 1, features]
                x = x.unsqueeze(1)
            
            # Split into latent and coords
            latent_vec = x[:, :, :self.z_dim]  # [batch_size, num_coords, z_dim]
            coords = x[:, :, self.z_dim:]      # [batch_size, num_coords, 3]
            
        elif latent_vec is not None and coords is not None:
            # DeepSDF interface is latent_vec: [batch_size, z_dim] and coords: [batch_size, num_coords, 3]
            
            # Expand latent vector to match coordinate dimensions
            latent_vec = latent_vec.unsqueeze(1).repeat(1, coords.shape[1], 1)
        else:
            raise ValueError("Must provide either 'x' or both 'latent_vec' and 'coords'")
        
        # Concatenate latent and coordinates
        x = torch.cat([latent_vec, coords], dim=-1)
        skip_x = x

        x = self.input_layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(torch.cat([x, skip_x], dim=-1))  # skip connection
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        # Return SDF values
        result = x.squeeze(-1).tanh()
        
        # If input was 2D (single point), return 2D output
        if result.dim() == 3 and result.shape[1] == 1:
            result = result.squeeze(1)
            
        return result
    
    def get_architecture_info(self):
        """Get information about the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_info = []
        for i, layer in enumerate([self.input_layer, self.layer2, self.layer3, 
                                  self.layer4, self.layer5, self.layer6, 
                                  self.layer7, self.layer8]):
            if hasattr(layer, '__len__') and len(layer) > 0:
                # Sequential layer block
                linear_layer = layer[0]  # First layer is Linear
                layer_info.append({
                    'layer': i,
                    'type': 'Sequential(Linear+ReLU+Dropout)',
                    'input_size': linear_layer.in_features,
                    'output_size': linear_layer.out_features,
                    'parameters': linear_layer.in_features * linear_layer.out_features + linear_layer.out_features
                })
            elif isinstance(layer, torch.nn.Linear):
                # Final linear layer
                layer_info.append({
                    'layer': i,
                    'type': 'Linear',
                    'input_size': layer.in_features,
                    'output_size': layer.out_features,
                    'parameters': layer.in_features * layer.out_features + layer.out_features
                })
        
        return {
            'architecture': 'DeepSDF',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'z_dim': self.z_dim,
            'coord_dim': self.coord_dim,
            'layer_size': self.layer_size,
            'dropout_p': self.dropout_p,
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
            'deepsdf': DeepSDF,
        }
    
    def get_model(self, device=None):
        """
        Create and return the configured model
        
        Args:
            device: Target device for the model
        
        Returns:
            nn.Module: Configured neural network model
        """
        # Get architecture configuration
        architecture_name = self.config['model_name'].lower()
        architecture_config = self.config.get('architecture', {}).get(architecture_name, {})
        
        # For backward compatibility, also check root-level config
        if not architecture_config:
            # Use relevant config parameters from the root level
            if architecture_name == 'mlp':
                architecture_config = {
                    'input_dim': self.config.get('input_dim', 3),
                    'hidden_dims': self.config.get('hidden_dims', [128, 64, 32]),
                    'output_dim': self.config.get('output_dim', 1),
                    'activation': self.config.get('activation', 'relu'),
                    'dropout': self.config.get('dropout', 0.1)
                }
            elif architecture_name == 'deepsdf':
                architecture_config = {
                    'z_dim': self.config.get('z_dim', 256),
                    'layer_size': self.config.get('layer_size', 32),
                    'dropout_p': self.config.get('dropout_p', 0.2),
                    'coord_dim': self.config.get('coord_dim', 3)
                }
        
        # Validate architecture exists
        if architecture_name not in self.architecture_registry:
            available = list(self.architecture_registry.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. Available: {available}")
        
        # Create model with appropriate configuration
        model_class = self.architecture_registry[architecture_name]
        
        if architecture_name == 'mlp':
            # SimpleMLP expects individual parameters
            model = model_class(**architecture_config)
        elif architecture_name == 'deepsdf':
            # DeepSDF expects a config dict
            model = model_class(config=architecture_config)
        else:
            # Default: try passing config dict
            model = model_class(architecture_config)
        
        # Use float32 for better compatibility and performance
        model = model.float()
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        return model

    
