import torch
import numpy as np
import yaml
import os

from deepsdf.Model import DeepSDF
from volume.model import Latent2Volume, Latent2Genera
from volume.dataset import VolumeDataset, GeneraDataset
from volume.compute_path_opt import compute_path
from visualize import visualize_interpolation_path  
from volume.compute_volume_genus import generate_mesh_from_latent, compute_volume, compute_genus, get_volume_coords
from torch.utils.data import DataLoader
from volume.train import train_and_save
from volume.config import BATCH_SIZE, LR, DEV as CONFIG_DEV  # reuse existing config constants

# Get the best model from training
with open('config/config_examples.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Updated: device and resolution now live under trainer_config (see config_examples.yaml)
trainer_cfg = config.get('trainer_config', {})
# Prefer run-time override from config_examples.yaml but fall back to volume.config DEV
DEV = trainer_cfg.get('device', CONFIG_DEV)
RESOLUTION = trainer_cfg.get('resolution', 100)
LATENT_DIM = config['model_config']['z_dim']
LAYER_SIZE = config['model_config']['layer_size']
print("Layer Size:", LAYER_SIZE)
print("Device:", DEV)
print("Latent dim:", LATENT_DIM)
print("Grid resolution:", RESOLUTION)

best_model_path = r"/Users/marina.levay/Documents/GitHub/topology-control/artifacts/experiment_20250818_112302/training_artifacts/run_20250818_112334/models/best_model.pth"

model = DeepSDF(config['model_config']).to(DEV)
ckpt = torch.load(best_model_path, map_location=DEV)
model.load_state_dict(ckpt['model_state_dict'])

latent_vectors = ckpt['latent_vectors']  # shape [num_shapes, z_dim]
print("Generating new dataset from", latent_vectors.shape[0], "latents...")

volumes = []
genera = []
coords, grid_size_axis = get_volume_coords(resolution=RESOLUTION)
for i, latent in enumerate(latent_vectors):
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(DEV)
    try:
        vertices, faces = generate_mesh_from_latent(latent_tensor, coords, grid_size_axis, model)
        volume_val = compute_volume(vertices, faces)
        genus_val = compute_genus(vertices, faces)
    except Exception as e:
        print(f"Failed to process latent {i}: {e}")
        volume_val = np.nan
        genus_val = np.nan
        
    volumes.append(volume_val)
    genera.append(genus_val)

# Convert to numpy arrays
volumes = np.array(volumes)
genera = np.array(genera)
latents = np.array(latent_vectors)

# Filter out NaNs (failed mesh/genus/volume computations)
mask = ~np.isnan(volumes) & ~np.isnan(genera)
volumes = volumes[mask]
genera = genera[mask]
latents = latents[mask]

npz_path = os.path.join("volume", "data", "2d_latents_volumes.npz")
np.savez(npz_path, latents=latents, volumes=volumes, genera=genera)
print(f"Saved new dataset to {npz_path}")

# Train Latent2Volume
volume_dataset = VolumeDataset(npz_path)
volume_loader = DataLoader(volume_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=volume_dataset.collate_fn)
volume_regressor = Latent2Volume(LATENT_DIM).to(DEV)
vol_crit = torch.nn.L1Loss()  # matches train.py example
vol_opt = torch.optim.Adam(volume_regressor.parameters(), lr=LR)
train_and_save(volume_loader, volume_regressor, vol_crit, vol_opt, name="volume")

genera_dataset = GeneraDataset(npz_path)
genera_loader = DataLoader(genera_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=genera_dataset.collate_fn)
genus_classifier = Latent2Genera(LATENT_DIM, min_genus=genera_dataset.gen_min, num_classes=genera_dataset.num_classes).to(DEV)
gen_crit = torch.nn.CrossEntropyLoss()
gen_opt = torch.optim.Adam(genus_classifier.parameters(), lr=LR)
train_and_save(genera_loader, genus_classifier, gen_crit, gen_opt, name="genera")

# Reload best checkpoints to ensure we use latest saved versions
ckpt_dir = os.path.join("volume", "checkpoints")
vol_ckpt_path = os.path.join(ckpt_dir, "latent2volume_best.pt")
gen_ckpt_path = os.path.join(ckpt_dir, "latent2genera_best.pt")

if os.path.exists(vol_ckpt_path):
    ckpt_vol = torch.load(vol_ckpt_path, map_location=DEV)
    volume_regressor.load_state_dict(ckpt_vol["model_state_dict"])
    print(f"Loaded best volume regressor from {vol_ckpt_path}")
else:
    print(f"[WARN] Volume checkpoint not found at {vol_ckpt_path}")

if os.path.exists(gen_ckpt_path):
    ckpt_gen = torch.load(gen_ckpt_path, map_location=DEV)
    genus_classifier.load_state_dict(ckpt_gen["model_state_dict"])
    print(f"Loaded best genus classifier from {gen_ckpt_path}")
else:
    print(f"[WARN] Genera checkpoint not found at {gen_ckpt_path}")

# Build interpolation path between first two latent codes
if latent_vectors.shape[0] >= 2:
    latent_start = torch.tensor(latent_vectors[0], dtype=torch.float32).to(DEV)
    latent_end = torch.tensor(latent_vectors[1], dtype=torch.float32).to(DEV)
    path = compute_path(latent_start, latent_end, volume_regressor, 20, lr=0.001).to(DEV)
    _ = [volume_regressor(p.unsqueeze(0).to(DEV)) for p in path]
    visualize_interpolation_path(model, path, volume_regressor, genus_classifier)
else:
    print("Not enough latent vectors for interpolation path (need at least 2).")