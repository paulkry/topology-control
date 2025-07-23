import torch
import os
import igl
import numpy as np
import skimage

import sys
sys.path.append('../')
from src.CPipelineOrchestrator import CPipelineOrchestrator

from config import VOLUME_DIR, COORDS_FIRST
from compute_volume import get_volume_coords, compute_volume, compute_genus, predict_sdf
from tqdm import tqdm

CKPT_PATH = '../artifacts/first_working.pth'
LATENT_DIM = 16

def generate_mesh(latent_code, coords, grid_size, model, type=COORDS_FIRST, device='cuda'):
    sdf = predict_sdf(latent_code.to(device), coords[None].to(device), model.to(device), type).flatten()

    vertices, faces, e2v = igl.marching_cubes(
        sdf.cpu().numpy(), 
        coords.cpu().numpy(), 
        grid_size, grid_size, grid_size, 0.0
    )

    return vertices, faces


def generate_latent_volume_data(n, model):
    # sample from uniform(-.5, .5) as the latents
    latents = torch.rand(n, LATENT_DIM) - .5
    data_dir = os.path.join(VOLUME_DIR, "data") #"volume/data/"

    volumes = []
    genera = []

    coords, grid_size = get_volume_coords()

    for i, latent in enumerate(tqdm(latents)):
        vertices, faces = generate_mesh(latent, coords, grid_size, model, COORDS_FIRST)
        volumes.append(compute_volume(vertices, faces))
        genera.append(compute_genus(vertices, faces))

    print(genera)
    np.savez(os.path.join(data_dir, "2d_latents_volumes_deepsdf"),
                 latents=latents.numpy().astype(np.float32),
                 volumes=np.array(volumes, dtype=np.float32),
                 genera=np.array(genera, dtype=np.int8))







if __name__ == "__main__":

    # DATA GENERATION
    ckpt = torch.load(CKPT_PATH)
    orc = CPipelineOrchestrator('/homes/dnogina/code/topology-control/config/config_examples.yaml')
    model = orc.architecture_manager.get_model()
    _ = model.load_state_dict(ckpt['model_state_dict'])

    generate_latent_volume_data(2000, model)

    # 