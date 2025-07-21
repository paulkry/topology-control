import torch
import torch.nn.functional as F

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

    def forward(self, latent_vec, coords):
        """
        Forward pass for DeepSDF - simplified and robust.
        
        Args:
            latent_vec: Latent vector of shape [batch_size, z_dim]
            coords: Coordinates of shape [batch_size, num_coords, 3]
            
        Returns:
            torch.Tensor: SDF values of shape [batch_size, num_coords]
        """
        # Validate input shapes
        if latent_vec.dim() != 2:
            raise ValueError(f"latent_vec must be 2D [batch_size, z_dim], got shape {latent_vec.shape}")
        if coords.dim() != 3:
            raise ValueError(f"coords must be 3D [batch_size, num_coords, 3], got shape {coords.shape}")
        
        batch_size, num_coords, coord_dim = coords.shape
        z_dim = latent_vec.shape[1]
        
        # Expand latent vector to match number of coordinates
        # [batch_size, z_dim] -> [batch_size, num_coords, z_dim]
        latent_expanded = latent_vec.unsqueeze(1).expand(batch_size, num_coords, z_dim)
        
        # Concatenate latent vector and coordinates
        # [batch_size, z_dim + 3]
        x = torch.cat([latent_expanded, coords], dim=-1)
        
        # Store for skip connection
        skip_x = x  # [batch_size, num_coords, z_dim + 3]
        
        # Reshape for processing through linear layers
        # [batch_size * num_coords, z_dim + 3]
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        skip_x_flat = skip_x.view(-1, skip_x.shape[-1])
        
        # Forward pass through network
        x = self.input_layer(x)                                    # [batch*coords, layer_size]
        x = self.layer2(x)                                         # [batch*coords, layer_size]
        x = self.layer3(x)                                         # [batch*coords, layer_size]
        x = self.layer4(x)                                         # [batch*coords, layer_size - input_dim]
        
        # Skip connection: concatenate with original input
        x = self.layer5(torch.cat([x, skip_x_flat], dim=-1))     # [batch*coords, layer_size]
        x = self.layer6(x)                                         # [batch*coords, layer_size]
        x = self.layer7(x)                                         # [batch*coords, layer_size]
        x = self.layer8(x)                                         # [batch*coords, 1]
        
        # Reshape back to original batch structure
        # [batch*coords, 1] -> [batch_size, num_coords, 1] -> [batch_size, num_coords]
        x = x.view(original_shape[0], original_shape[1], -1)
        x = x.squeeze(-1)
        
        # Apply tanh activation to bound SDF values
        return x.tanh()
    
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

class VolumeDeepSDF(torch.nn.Module):
    """
    Enhanced VolumeDeepSDF that better integrates DeepSDF and DeepSets.
    
    Key improvements:
    1. Follows DeepSDF architecture more closely for SDF prediction
    2. Uses shared features more effectively for volume prediction
    3. Better skip connections and feature reuse
    4. More principled Deep Sets implementation
    """
    
    def __init__(self, config=None):
        super(VolumeDeepSDF, self).__init__()
        
        if config is None:
            config = {}
        
        # Extract parameters with defaults - FIXED: Use 16 as default to match your dataset
        self.z_dim = config.get('z_dim', 16)  # Match your dataset configuration
        self.layer_size = config.get('layer_size', 256)  # Use 256 as base layer size
        self.dropout_p = config.get('dropout_p', 0.2)
        self.coord_dim = config.get('coord_dim', 3)
        self.volume_hidden_dim = config.get('volume_hidden_dim', 128)
        
        # Store architecture info
        self.input_dim = self.z_dim + self.coord_dim  # 16 + 3 = 19
        self.output_dim = 2  # SDF + Volume
        self.task = "sdf_and_volume_prediction"
        
        print(f"VolumeDeepSDF initialized with:")
        print(f"  z_dim: {self.z_dim}")
        print(f"  coord_dim: {self.coord_dim}")
        print(f"  input_dim: {self.input_dim}")
        print(f"  layer_size: {self.layer_size}")
        print(f"  volume_hidden_dim: {self.volume_hidden_dim}")
        
        # ================================================================
        # SHARED DEEPSDF BACKBONE (following DeepSDF paper architecture)
        # ================================================================
        
        # Initial layers before skip connection
        self.shared_layer1 = self.create_layer_block(self.input_dim, self.layer_size)           # 19 -> 256
        self.shared_layer2 = self.create_layer_block(self.layer_size, self.layer_size)          # 256 -> 256
        self.shared_layer3 = self.create_layer_block(self.layer_size, self.layer_size)          # 256 -> 256
        self.shared_layer4 = self.create_layer_block(self.layer_size, self.layer_size)          # 256 -> 256
        
        # FIXED: After skip connection, we need to handle the concatenated dimension
        # Skip connection adds input_dim to layer_size: 256 + 19 = 275
        skip_concat_dim = self.layer_size + self.input_dim  # 256 + 19 = 275
        
        # Post-skip connection layers (shared by both heads)
        self.shared_layer5 = self.create_layer_block(skip_concat_dim, self.layer_size)          # 275 -> 256
        self.shared_layer6 = self.create_layer_block(self.layer_size, self.layer_size)          # 256 -> 256
        
        # ================================================================
        # SDF PREDICTION HEAD (DeepSDF style)
        # ================================================================
        self.sdf_layer7 = self.create_layer_block(self.layer_size, self.layer_size)             # 256 -> 256
        self.sdf_layer8 = self.create_layer_block(self.layer_size, self.layer_size)             # 256 -> 256
        self.sdf_output = torch.nn.Linear(self.layer_size, 1)                                   # 256 -> 1
        
        # ================================================================
        # VOLUME PREDICTION HEAD (Deep Sets with shared features)
        # ================================================================
        
        # Phi function: per-point feature extraction for volume
        self.volume_phi = torch.nn.Sequential(
            torch.nn.Linear(self.layer_size, self.volume_hidden_dim),                           # 256 -> 128
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.volume_hidden_dim, self.volume_hidden_dim),                    # 128 -> 128
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.volume_hidden_dim, self.volume_hidden_dim),                    # 128 -> 128
            torch.nn.ReLU()
        )
        
        # Rho function: aggregated feature processing
        self.volume_rho = torch.nn.Sequential(
            torch.nn.Linear(self.volume_hidden_dim, self.volume_hidden_dim),                    # 128 -> 128
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.volume_hidden_dim, self.volume_hidden_dim // 2),               # 128 -> 64
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_p),
            torch.nn.Linear(self.volume_hidden_dim // 2, self.volume_hidden_dim // 4),          # 64 -> 32
            torch.nn.ReLU(),
            torch.nn.Linear(self.volume_hidden_dim // 4, 1)                                     # 32 -> 1
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Print architecture summary
        self._print_architecture_summary()
    
    def _print_architecture_summary(self):
        """Print architecture layer dimensions for debugging."""
        print("\nVolumeDeepSDF Architecture Summary:")
        print("=" * 50)
        print(f"Input: latent({self.z_dim}) + coords({self.coord_dim}) = {self.input_dim}")
        print(f"Shared Layer 1: {self.input_dim} -> {self.layer_size}")
        print(f"Shared Layer 2: {self.layer_size} -> {self.layer_size}")
        print(f"Shared Layer 3: {self.layer_size} -> {self.layer_size}")
        print(f"Shared Layer 4: {self.layer_size} -> {self.layer_size}")
        print(f"Skip Connection: {self.layer_size} + {self.input_dim} = {self.layer_size + self.input_dim}")
        print(f"Shared Layer 5: {self.layer_size + self.input_dim} -> {self.layer_size}")
        print(f"Shared Layer 6: {self.layer_size} -> {self.layer_size}")
        print("SDF Head:")
        print(f"  SDF Layer 7: {self.layer_size} -> {self.layer_size}")
        print(f"  SDF Layer 8: {self.layer_size} -> {self.layer_size}")
        print(f"  SDF Output: {self.layer_size} -> 1")
        print("Volume Head:")
        print(f"  Volume Phi: {self.layer_size} -> {self.volume_hidden_dim}")
        print(f"  Volume Rho: {self.volume_hidden_dim} -> 1")
        print("=" * 50)
    
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

    def forward(self, latent_vec, coords):
        """
        Enhanced forward pass with better feature sharing between SDF and volume prediction.
        
        Args:
            latent_vec: Latent vector of shape [batch_size, z_dim]
            coords: Coordinates of shape [batch_size, num_coords, 3]
            
        Returns:
            dict: {
                'sdf': SDF values of shape [batch_size, num_coords],
                'volume': Volume predictions of shape [batch_size]
            }
        """
        # Validate input shapes
        if latent_vec.dim() != 2:
            raise ValueError(f"latent_vec must be 2D [batch_size, z_dim], got shape {latent_vec.shape}")
        if coords.dim() != 3:
            raise ValueError(f"coords must be 3D [batch_size, num_coords, 3], got shape {coords.shape}")
        
        batch_size, num_coords, coord_dim = coords.shape
        z_dim = latent_vec.shape[1]
        
        # ENHANCED: More informative dimension validation
        if z_dim != self.z_dim:
            error_msg = (f"Latent vector dimension mismatch.\n"
                        f"  Model expects z_dim: {self.z_dim}\n"
                        f"  Dataset provides z_dim: {z_dim}\n"
                        f"  Latent vector shape: {latent_vec.shape}\n"
                        f"  Check your model configuration in config.yaml:\n"
                        f"    model_config.z_dim should be {z_dim}\n"
                        f"  Or check your dataset configuration.")
            raise ValueError(error_msg)
        
        if coord_dim != self.coord_dim:
            error_msg = (f"Coordinate dimension mismatch.\n"
                        f"  Model expects coord_dim: {self.coord_dim}\n"
                        f"  Dataset provides coord_dim: {coord_dim}\n"
                        f"  Coordinates shape: {coords.shape}")
            raise ValueError(error_msg)
        
        # ================================================================
        # INPUT PREPARATION (DeepSDF style)
        # ================================================================
        
        # Expand latent vector to match coordinates
        latent_expanded = latent_vec.unsqueeze(1).expand(batch_size, num_coords, z_dim)
        
        # Concatenate latent and coordinates
        x = torch.cat([latent_expanded, coords], dim=-1)  # [batch_size, num_coords, z_dim + 3]
        
        # Store original input for skip connection
        skip_input = x
        
        # Flatten for processing
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])  # [batch_size * num_coords, z_dim + 3]
        skip_input_flat = skip_input.view(-1, skip_input.shape[-1])
        
        # ================================================================
        # SHARED DEEPSDF BACKBONE
        # ================================================================
        
        # Pre-skip layers
        x = self.shared_layer1(x)      # [batch*coords, layer_size]
        x = self.shared_layer2(x)      # [batch*coords, layer_size]
        x = self.shared_layer3(x)      # [batch*coords, layer_size]
        x = self.shared_layer4(x)      # [batch*coords, layer_size]
        
        # Skip connection (DeepSDF style)
        x = torch.cat([x, skip_input_flat], dim=-1)  # [batch*coords, layer_size + input_dim]
        
        # Post-skip shared layers
        x = self.shared_layer5(x)      # [batch*coords, layer_size]
        shared_features = self.shared_layer6(x)  # [batch*coords, layer_size]
        
        # ================================================================
        # SDF PREDICTION BRANCH
        # ================================================================
        
        sdf_x = self.sdf_layer7(shared_features)   # [batch*coords, layer_size]
        sdf_x = self.sdf_layer8(sdf_x)             # [batch*coords, layer_size]
        sdf_output = self.sdf_output(sdf_x)        # [batch*coords, 1]
        
        # Reshape and apply activation
        sdf_output = sdf_output.view(original_shape[0], original_shape[1], -1).squeeze(-1)
        sdf_output = sdf_output.tanh()  # Bound SDF values
        
        # ================================================================
        # VOLUME PREDICTION BRANCH (Deep Sets)
        # ================================================================
        
        # Apply phi function to shared features (per-point processing)
        volume_features = self.volume_phi(shared_features)  # [batch*coords, volume_hidden_dim]
        
        # Reshape for aggregation
        volume_features = volume_features.view(batch_size, num_coords, -1)
        
        # Deep Sets aggregation (permutation invariant)
        aggregated_features, _ = torch.max(volume_features, dim=1)  # [batch_size, volume_hidden_dim]
        
        # Apply rho function to aggregated features
        volume_output = self.volume_rho(aggregated_features)  # [batch_size, 1]
        
        # Ensure positive volumes
        volume_output = F.softplus(volume_output).squeeze(-1)  # [batch_size]
        
        return {
            'sdf': sdf_output,      # [batch_size, num_coords]
            'volume': volume_output  # [batch_size]
        }

    def get_architecture_info(self):
        """Get detailed information about the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by component
        shared_layers = [self.shared_layer1, self.shared_layer2, self.shared_layer3, 
                        self.shared_layer4, self.shared_layer5, self.shared_layer6]
        shared_params = sum(sum(p.numel() for p in layer.parameters()) for layer in shared_layers)
        
        sdf_layers = [self.sdf_layer7, self.sdf_layer8, self.sdf_output]
        sdf_params = sum(sum(p.numel() for p in layer.parameters()) for layer in sdf_layers)
        
        volume_params = sum(p.numel() for p in self.volume_phi.parameters()) + \
                       sum(p.numel() for p in self.volume_rho.parameters())
        
        return {
            'architecture': 'VolumeDeepSDF_Enhanced',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'z_dim': self.z_dim,
            'coord_dim': self.coord_dim,
            'layer_size': self.layer_size,
            'volume_hidden_dim': self.volume_hidden_dim,
            'dropout_p': self.dropout_p,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'shared_parameters': shared_params,
            'sdf_parameters': sdf_params,
            'volume_parameters': volume_params,
            'parameter_distribution': {
                'shared_backbone': f"{shared_params/total_params*100:.1f}%",
                'sdf_head': f"{sdf_params/total_params*100:.1f}%",
                'volume_head': f"{volume_params/total_params*100:.1f}%"
            },
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
            'volumedeepsdf': VolumeDeepSDF
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
                    'z_dim': self.config.get('z_dim', 16),  # FIXED: Use 16 as default
                    'layer_size': self.config.get('layer_size', 256),  # FIXED: Use 256 as default
                    'dropout_p': self.config.get('dropout_p', 0.2),
                    'coord_dim': self.config.get('coord_dim', 3)
                }
            elif architecture_name == 'volumedeepsdf':
                architecture_config = {
                    'z_dim': self.config.get('z_dim', 16),  # FIXED: Use 16 as default
                    'layer_size': self.config.get('layer_size', 256),  # FIXED: Use 256 as default
                    'dropout_p': self.config.get('dropout_p', 0.2),
                    'coord_dim': self.config.get('coord_dim', 3),
                    'volume_hidden_dim': self.config.get('volume_hidden_dim', 128)
                }
        
        # ENHANCED: Print configuration being used for debugging
        print(f"\n🏗️  Building {architecture_name.upper()} model with config:")
        for key, value in architecture_config.items():
            print(f"  {key}: {value}")
        
        # Validate architecture exists
        if architecture_name not in self.architecture_registry:
            available = list(self.architecture_registry.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. Available: {available}")
        
        # Create model with appropriate configuration
        model_class = self.architecture_registry[architecture_name]
        
        if architecture_name == 'mlp':
            # SimpleMLP expects individual parameters
            model = model_class(**architecture_config)
        elif architecture_name in ['deepsdf', 'volumedeepsdf']:
            # These expect a config dict
            model = model_class(config=architecture_config)
        else:
            # Default: try passing config dict
            model = model_class(architecture_config)
        
        # Use float32 for better compatibility and performance
        model = model.float()
        
        # Move to device if specified
        if device is not None:
            model = model.to(device)
        
        print(f"✅ {architecture_name.upper()} model created successfully")
        return model
    
