import torch

class DeepSDF(torch.nn.Module):

    def __init__(self, config=None):
        super(DeepSDF, self).__init__()
        
        if config is None:
            config = {}
            
        self.z_dim = config.get('z_dim', 128) 
        self.layer_size = config.get('layer_size', 256)  
        self.coord_dim = config.get('coord_dim', 3)  # 3D coordinates
        
        self.input_dim = self.z_dim + self.coord_dim
        self.output_dim = 1
        self.task = "signed_distance_prediction"

        self.input_layer = self.create_layer_block(self.input_dim, self.layer_size)
        self.layer2 = self.create_layer_block(self.layer_size, self.layer_size - self.input_dim)
        self.layer3 = self.create_layer_block(self.layer_size, self.layer_size)
        self.output_layer = torch.nn.Linear(self.layer_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def create_layer_block(self, input_size, output_size):
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU(),
        )

    def forward(self, latent_vec, coords):
        """
        Forward pass for DeepSDF - 4-layer architecture.
        
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
            coords = coords.unsqueeze(1) if coords.dim() == 2 else coords  # Ensure coords is 3D
            # raise ValueError(f"coords must be 3D [batch_size, num_coords, 3], got shape {coords.shape}")
        
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
        
        # Forward pass 
        x = self.input_layer(x) # [batch*coords, layer_size]
        x = self.layer2(x) # [batch*coords, layer_size - input_dim]
        # Skip connection and concatenate with original input
        x = self.layer3(torch.cat([x, skip_x_flat], dim=-1)) # [batch*coords, layer_size]
        x = self.output_layer(x) # [batch*coords, 1]
        
        # Reshape back to original batch structure
        # [batch*coords, 1] -> [batch_size, num_coords, 1] -> [batch_size, num_coords]
        x = x.view(original_shape[0], original_shape[1], -1)
        x = x.squeeze(-1)
        
        return x.tanh()
        

        

