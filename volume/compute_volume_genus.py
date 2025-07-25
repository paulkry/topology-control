import igl
import polyscope as ps
import numpy as np
import meshio as meshio
import numpy as np
import trimesh
import torch
import os
from tqdm import tqdm
from typing import List

from sdfs import SDF_interpolator, sdf_sphere, sdf_torus, sdf_2_torus
from config import DEV, VOLUME_DIR, COORDS_FIRST, LATENT_FIRST, LATENT_DIM

def generate_mesh_from_sdf(sdf, coords, grid_size):
    vertices, faces, e2v = igl.marching_cubes(
        sdf.cpu().numpy(), 
        coords.cpu().numpy(), 
        grid_size, grid_size, grid_size, 0.0
    )

    return vertices, faces


def generate_mesh_from_latent(latent_code, coords, grid_size, model, type=COORDS_FIRST):
    sdf = predict_sdf(latent_code, coords, model, type).flatten()
    vertices, faces = generate_mesh_from_sdf(sdf, coords, grid_size)

    return vertices, faces

    
def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions
    on the [-1, 1]^3 cube"""
    # Define grid
    grid_values = torch.arange(-.7, .7, float(1/resolution)).to(DEV) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(DEV)

    return coords, grid_size_axis


def predict_sdf(latent, coords, model, type=COORDS_FIRST):

    sdf_values = torch.tensor([], dtype=torch.float32).view(1, 0).to(DEV)

    # Split coords into batches because of memory limitations
    coords_batches = torch.split(coords, 100_000)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            if type==COORDS_FIRST:
                latent_batch = latent.unsqueeze(0).to(DEV)
                sdf_batch = model(latent_batch, coords)
            elif type==LATENT_FIRST:
                # expects a tiled latent vec
                latent_batch = torch.tile(latent.unsqueeze(0), (coords.shape[0], 1)).to(DEV)
                sdf_batch = model(coords, latent_batch)
            else:
                raise "Invalid type"
            # sdf_batch = model(latent_batch, coords)
            sdf_values = torch.hstack((sdf_values, sdf_batch.view(1, -1)))        

    return sdf_values



def compute_genus(vertices, faces):
    """
    Compute the genus of the object represented by the latent code.
    """
    # compute genus using: V - E + F = 2 - 2G
    V = vertices.shape[0]
    E = len(set(tuple(sorted(edge)) for face in faces for edge in [(face[i], face[(i+1) % 3]) for i in range(3)]))
    F = faces.shape[0]

    genus = (2 - (V - E + F)) // 2

    return genus


def compute_volume(vertices, faces):
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    if not isinstance(faces, np.ndarray):
        faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    volume = abs(mesh.volume)
        
    return volume

    #start regularization to preserve topology and optimize path in latent space
def L_topology(latent, coords, model, type=COORDS_FIRST):
    sdf = predict_sdf(latent, coords, model, type)
    grad_outputs = torch.ones_like(sdf, requires_grad=False)
    grads = torch.autograd.grad(
        outputs=sdf,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    laplacian = torch.sum((grads.norm(dim=1) - 1) ** 2)
    return laplacian

def L_path(latents: List[torch.Tensor], coords, model, type=COORDS_FIRST):
    losses = []
    for z_t in latents:
        losses.append(L_topology(z_t, coords, model, type))
    return torch.mean(torch.stack(losses))

def L_components(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    n_components = len(mesh.split(only_watertight=False))
    return (n_components - 1) ** 2

    
def L_path_smoothness(latents, coords, model, type=COORDS_FIRST):
    diffs = [torch.norm(latents[i+1] - latents[i])**2 for i in range(len(latents) - 1)]
    return torch.mean(torch.stack(diffs))

def L_latent_path_length(latents):
    lengths = [torch.norm(latents[i+1] - latents[i]) for i in range(len(latents) - 1)]
    return torch.sum(torch.stack(lengths))

def L_sdf_consistency(latents, coords, model, type=COORDS_FIRST):
    sdf_values = [predict_sdf(z, coords, model, type) for z in latents]
    diffs = [torch.norm(sdf_values[i+1] - sdf_values[i])**2 for i in range(len(sdf_values) - 1)]
    return torch.mean(torch.stack(diffs))

    #Regularization sum  for topology, SDF gradient smoothness on path, SDF value consistency, length reg. 
def L_path_combined(latents, coords, model, type=COORDS_FIRST, alpha=1.0, beta=1.0, gamma=1.0):
    topology_loss = L_path(latents, coords, model, type)
    smoothness_loss = L_path_smoothness(latents, coords, model, type)
    latent_length_loss = L_latent_path_length(latents)
    sdf_consistency_loss = L_sdf_consistency(latents, coords, model, type)
    return (alpha * topology_loss +
            beta * smoothness_loss +
            gamma * sdf_consistency_loss +
            0.1 * latent_length_loss)

def generate_latent_volume_data(n, model):
    # sample from uniform(0, 10)
    latents = torch.rand(n, LATENT_DIM) * 10
    data_dir = os.path.join(VOLUME_DIR, "data") #"volume/data/"

    volumes = []
    genera = []

    coords, grid_size = get_volume_coords()

    for i, latent in enumerate(tqdm(latents)):
        vertices, faces = generate_mesh_from_latent(latent, coords, grid_size, model, LATENT_FIRST)
        volumes.append(compute_volume(vertices, faces))
        genera.append(compute_genus(vertices, faces))

    print(genera)
    np.savez(os.path.join(data_dir, "2d_latents_volumes"),
                 latents=latents.numpy().astype(np.float32),
                 volumes=np.array(volumes, dtype=np.float32),
                 genera=np.array(genera, dtype=np.int8))
    

def generate_syn_latent_volume_data(num_samples):
    data_dir = os.path.join(VOLUME_DIR, "data") # volume/data
    interpolator = SDF_interpolator(sdf_sphere, sdf_torus, sdf_2_torus)
    # will be passing coords around to avoid recomputation
    coords, grid_size = get_volume_coords(resolution=50)

    volumes = []
    genera = []
    latents = []
    for i in tqdm(range(num_samples)):
        alpha = torch.rand(1).item()
        beta = torch.rand(1).item() * (1 - alpha)  # ensure alpha + beta ≤ 1
        latents.append((alpha, beta))
        assert 0.0 <= alpha + beta <= 1.0
        vertices, faces = generate_mesh_from_latent(torch.tensor([alpha, beta]), coords, grid_size, interpolator)

        volume = compute_volume(vertices, faces)
        genus = compute_genus(vertices, faces)

        volumes.append(volume)
        genera.append(genus)
    print(genera)
    np.savez(os.path.join(data_dir, "2d_latents_volumes.npz"),
                latents=np.array(latents, dtype=np.float32),
                volumes=np.array(volumes, dtype=np.float32),
                genera=np.array(genera, dtype=np.int8))
    


if __name__ == "__main__":
    #----------------------------------------------------------------
    # Data Generation
    model_path = "trained_deepsdfs/sdfnet_model.pt"
    scripted_model = torch.jit.load(model_path).to(DEV)

    generate_latent_volume_data(2000, scripted_model)
    # generate_syn_latent_volume_data(2000)
    