import torch
import numpy as np
import yaml
from src.CModelTrainer import CModelTrainer
from src.CArchitectureManager import CArchitectureManager
from src.CEvaluator import CEvaluator
from volume.model import Latent2Volume, Latent2Genera
from volume.compute_path import compute_path
from volume.compute_volume_genus import get_volume_coords
from visualize import visualize_interpolation_path  
from volume.sdfs import SDF_interpolator, sdf_sphere, sdf_torus, sdf_2_torus

# Get the best model from training
with open('config/config_examples.yaml', 'r') as f:
    config = yaml.safe_load(f)

DEV = config['processor_config']['volume_processor_params']['device']
LATENT_DIM = config['model_config']['z_dim']
LAYER_SIZE = config['model_config']['layer_size']
print("LAYER_SIZE:", LAYER_SIZE)
print("Device:", DEV)
print("Latent dim:", LATENT_DIM)

best_model_path = "/Users/marina.levay/Documents/GitHub/topology-control/artifacts/experiment_20250726_224246/training_artifacts/run_20250726_224254/models/best_model.pth"

from volume.model import Latent2Volume, Latent2Genera
arch_manager = CArchitectureManager(config['model_config'])
model = arch_manager.get_model().to(DEV)
ckpt = torch.load(best_model_path, map_location=DEV)
model.load_state_dict(ckpt['model_state_dict'])

# Inputs is 7 because of coord_dim = 3 (3D) and z_dim = 4 (3D coords + 1 latent dimension)
volume_regressor = Latent2Volume(input_dim=7, layer_size=LAYER_SIZE).to(DEV)
volume_regressor.load_state_dict(ckpt['model_state_dict'])
volume_regressor.eval()

genus_classifier = Latent2Genera(input_dim=7, layer_size=LAYER_SIZE).to(DEV)
genus_classifier.load_state_dict(ckpt['model_state_dict'])
genus_classifier.eval()

latent_vectors = ckpt['latent_vectors']  # shape [num_shapes, z_dim]
latent_start = torch.tensor(latent_vectors[0], dtype=torch.float32).to(DEV)  # [z_dim]
latent_end = torch.tensor(latent_vectors[1], dtype=torch.float32).to(DEV)    # [z_dim]
#print("latent_vectors shape:", np.array(latent_vectors).shape)
#print("latent_start shape:", latent_start.shape)
#print("latent_end shape:", latent_end.shape)

# Use a single coordinate for path calculation (e.g., origin)
coords_for_path = torch.zeros(1, 3, dtype=torch.float32).to(DEV)  # shape [1, 3]
#print("coords_for_path shape:", coords_for_path.shape)

# Pass coords_for_path to compute_path
path = compute_path(latent_start, latent_end, volume_regressor, 20, coords=coords_for_path).cpu()  # path: [steps, z_dim]
#print("path shape:", path.shape)

# When evaluating the model for each latent in the path:
for latent in path:
    latent_batch = latent.expand(coords_for_path.shape[0], -1).to(DEV)  # [N, z_dim]
    coords_batch = coords_for_path.to(DEV)  # [N, 3]
    sdf_values = volume_regressor(latent_batch, coords_batch)
    #print("latent_batch shape:", latent_batch.shape)
    #print("coords_batch shape:", coords_batch.shape)
    #print("sdf_values shape:", sdf_values.shape)

visualize_interpolation_path(model, path, volume_regressor, genus_classifier)