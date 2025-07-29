import torch 
from torch import nn
from volume.config import LAYER_SIZE

class Latent2Volume(nn.Module):
    def __init__(self, input_dim, layer_size=LAYER_SIZE, dropout_p=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.layer_size = layer_size

        def create_layer_block(input_size, output_size):
            return nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Dropout(dropout_p)
            )

        self.input_layer = create_layer_block(input_dim, layer_size)
        self.layer2 = create_layer_block(layer_size, layer_size - input_dim)
        self.layer3 = create_layer_block(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, 1)

    def forward(self, latent, coords):
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)  # [1, z_dim]
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)  # [1, 3]
        x = torch.cat([latent, coords], dim=1)  # [N, z_dim + 3]
        x1 = self.input_layer(x)
        x2 = self.layer2(x1)
        x_cat = torch.cat([x2, x], dim=-1)
        x3 = self.layer3(x_cat)
        out = self.output_layer(x3)
        return out.squeeze(-1)
    
class Latent2Genera(nn.Module):
    def __init__(self, input_dim, layer_size=LAYER_SIZE):
        super().__init__()
        self.input_dim = input_dim
        self.layer_size = layer_size

        def create_layer_block(input_size, output_size):
            return nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU()
            )

        self.input_layer = create_layer_block(input_dim, layer_size)
        self.layer2 = create_layer_block(layer_size, layer_size - input_dim)
        self.layer3 = create_layer_block(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, 1)

    def forward(self, latent, coords):
        # Ensure both are 2D
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)  # [1, z_dim]
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)  # [1, 3]
        x = torch.cat([latent, coords], dim=1)  # [N, z_dim + 3]
        x1 = self.input_layer(x)
        x2 = self.layer2(x1)
        x_cat = torch.cat([x2, x], dim=-1)
        x3 = self.layer3(x_cat)
        out = self.output_layer(x3)
        return out.squeeze(-1)