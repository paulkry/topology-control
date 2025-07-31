import torch
import numpy as np
import yaml
import os

from src.CArchitectureManager import CArchitectureManager
from volume.model import Latent2Volume, Latent2Genera
from volume.compute_path import compute_path
from visualize import visualize_interpolation_path  

# Get the best model from training
with open('config/config_examples.yaml', 'r') as f:
    config = yaml.safe_load(f)

DEV = config['processor_config']['volume_processor_params']['device']
LATENT_DIM = config['model_config']['z_dim']
LAYER_SIZE = config['model_config']['layer_size']
print("LAYER_SIZE:", LAYER_SIZE)
print("Device:", DEV)
print("Latent dim:", LATENT_DIM)

best_model_path = r"C:\Users\singh\OneDrive\Documents\GitHub\topology-control\volume\trained_deepsdfs\best_model.pth"

arch_manager = CArchitectureManager(config['model_config'])
model = arch_manager.get_model().to(DEV)
ckpt = torch.load(best_model_path, map_location=DEV)
model.load_state_dict(ckpt['model_state_dict'])

# Inputs is 7 because of coord_dim = 3 (3D) and z_dim = 4 (3D coords + 1 latent dimension)
volume_regressor = Latent2Volume(input_dim=7).to(DEV)
volume_regressor.load_state_dict(ckpt['model_state_dict'])
volume_regressor.eval()

genus_classifier = Latent2Genera(input_dim=7).to(DEV)
genus_classifier.load_state_dict(ckpt['model_state_dict'])
genus_classifier.eval()

latent_vectors = ckpt['latent_vectors']  # shape [num_shapes, z_dim]
print(latent_vectors)
latent_start = torch.tensor(latent_vectors[0], dtype=torch.float32).to(DEV)  # [z_dim]
latent_end = torch.tensor(latent_vectors[1], dtype=torch.float32).to(DEV)    # [z_dim]

# Use a single coordinate for path calculation (e.g., origin)
coords_for_path = torch.zeros(1, 3, dtype=torch.float32).to(DEV)  # shape [1, 3]

# Pass coords_for_path to compute_path
path = compute_path(latent_start, latent_end, volume_regressor, 20, coords=coords_for_path).to(DEV)  # path: [steps, z_dim]

# When evaluating the model for each latent in the path:
for latent in path:
    latent_batch = latent.expand(coords_for_path.shape[0], -1).to(DEV)  # [N, z_dim]
    coords_batch = coords_for_path.to(DEV)  # [N, 3]
    sdf_values = volume_regressor(latent_batch, coords_batch)

print(path.device)
visualize_interpolation_path(model, path, volume_regressor, genus_classifier)