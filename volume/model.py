from torch import nn
from config import LAYER_SIZE

class Latent2Volume(nn.Module):

    def __init__(self, input_dim, layer_size = LAYER_SIZE, dropout_p = 0.2, num_layers = 4):
        super(Latent2Volume, self).__init__()
        self.dropout_p = dropout_p
        
        layers = []

        layers.append(nn.Linear(input_dim, layer_size))
        for i in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_p))
        
        layers.append(nn.Linear(layer_size, 1))

        self.net = nn.Sequential(*layers)
        

    def forward(self, latent_vec):
        """
        latent_vec has shape [batch_size, z_dim]
        """
        return self.net(latent_vec).squeeze(-1) # [batch_size]