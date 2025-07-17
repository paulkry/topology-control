import igl
import polyscope as ps
import numpy as np
import meshio as meshio
import numpy as np
import trimesh
import torch
import os

from sdfs import SDF_interpolator
from config import DEV



def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions
    on the [-1, 1]^3 cube"""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(DEV) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(DEV)

    return coords, grid_size_axis


def predict_sdf(latent, coords, model):
    # remove the line below for NN parametrized sdf

    sdf = torch.tensor([], dtype=torch.float32).view(1, 0).to(DEV)

    latent_batch = latent.unsqueeze(0)  
    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100000)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            coords = coords.unsqueeze(0)
            sdf_batch = model(latent_batch, coords)
            sdf = torch.hstack((sdf, sdf_batch.view(1, -1)))        

    return sdf

"""
This is the main function that takes a latent_code
and a potentially NN-parametrized SDF
to output the volume of the corresponding surface. 
We are hoping to use it to generate data for
the classifier (regressor) we will be training.
"""

def compute_volume(latent_code, coords, grid_size, model): 
    """
    Reconstruct the object from the latent code and visualize it.
    """
    sdf = predict_sdf(latent_code, coords, model).flatten()

    vertices, faces, _ = igl.marching_cubes(
        sdf.cpu().numpy(), 
        coords.cpu().numpy(), 
        grid_size, grid_size, grid_size, 0.0
    )

    volume, _ = triangle_mesh_to_volume(vertices, faces)
    
    return volume


def is_mesh_closed(vertices, faces):
    """
    Check if mesh is closed (watertight)
    """
    edges = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge in edges:
                edges.remove(edge)
            else:
                edges.add(edge)
    return len(edges) == 0

def triangle_mesh_to_volume(vertices, faces):
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    if not isinstance(faces, np.ndarray):
        faces = np.array(faces)

    is_closed = is_mesh_closed(vertices, faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    volume = abs(mesh.volume)
        
    return volume, is_closed

def generate_latent_volume_data(n, model, latent_dim, latent_sd, latent_mean):
    latents = torch.randn(n, latent_dim)
    latents = (latents * latent_sd) + latent_mean
    data_dir = "data/processed/train"

    volumes = []

    for latent in latents:
        volumes.append(compute_volume(latent, model))

    np.savez(os.path.join(data_dir, "volume_labeling"),
                 latents=latents.astype(np.float32),
                 volumes=volumes.astype(np.float32))
    

def generate_syn_latent_volume_data(num_samples):
    data_dir = "data/"
    interpolator = SDF_interpolator()
    # will be passing coords around to avoid recomputation
    coords, grid_size = get_volume_coords(resolution=50)

    volumes = []
    latents = []
    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"{i+1}th step; so far so good!")
        alpha = torch.rand(1).item()
        beta = torch.rand(1).item() * (1 - alpha)  # ensure alpha + beta ≤ 1
        latents.append((alpha, beta))
        assert 0.0 <= alpha + beta <= 1.0
        volume = compute_volume(torch.tensor([alpha, beta]), coords, grid_size, interpolator)
        volumes.append(volume)
    
    np.savez(os.path.join(data_dir, "2d_latents_volumes.npz"),
                latents=np.array(latents, dtype=np.float32),
                volumes=np.array(volumes, dtype=np.float32))
    


if __name__ == "__main__":

    #----------------------------------------------------------------
    # Making sure SDFs produce similar volumes
    # interpolator = SDF_interpolator()
    # volumes = []
    # coords, grid_size = get_volume_coords(resolution=50)

    # for alpha, beta in [(1, 0), (0, 1), (0, 0)]:
    #     volume = compute_volume(torch.tensor([alpha, beta]), coords, grid_size, interpolator)
    #     volumes.append(volume)

    # print(volumes)

    #----------------------------------------------------------------
    # Synthetic Data Generation

    generate_syn_latent_volume_data(10000)
    