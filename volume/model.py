import torch 
from torch import nn
from volume.config import LAYER_SIZE


class Latent2Volume(nn.Module):
    """
    A simple regressor model
    """
    def __init__(self, input_dim, layer_size = LAYER_SIZE, dropout_p = 0.2, num_layers = 2):
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
    

class Latent2Genera(nn.Module):
    """
    A simple classifier model
    """
    def __init__(self, input_dim, hidden_dim = LAYER_SIZE, num_classes = 5, num_layers=4, min_genus=0):
        super().__init__()
        self.num_classes = num_classes
        self.min_genus = min_genus

        layers = [nn.Linear(input_dim, hidden_dim)]
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def inference_forard(self, x):
        logits = self.net(x)
        return nn.functional.softmax(logits, dim=-1) + self.min_genus