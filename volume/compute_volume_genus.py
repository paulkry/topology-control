from typing import List
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
import yaml

from volume.sdfs import SDF_interpolator, sdf_sphere, sdf_torus, sdf_2_torus
from volume.config import DEV, VOLUME_DIR, COORDS_FIRST, LATENT_FIRST, LATENT_DIM, LATENT_VEC_MAX
from src.CArchitectureManager import CArchitectureManager

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
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(DEV) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(DEV)

    return coords, grid_size_axis

def predict_sdf(latent, coords, model, type=COORDS_FIRST):
    # Standardized: always call model(latent, coords) with correct shapes
    model.eval()
    with torch.no_grad():
        # Detect DeepSDF by class name or flag
        is_deepsdf = hasattr(model, "deepsdf") and model.deepsdf_flag or model.__class__.__name__.lower().startswith("deepsdf")
        if is_deepsdf:
            # DeepSDF expects: latent [1, z_dim], coords [1, N, 3]
            latent_batch = latent.unsqueeze(0) if latent.dim() == 1 else latent  # [1, z_dim]
            if coords.dim() == 2:
                coords_batch = coords.unsqueeze(0)  # [1, N, 3]
            elif coords.dim() == 3:
                coords_batch = coords  # [1, N, 3]
            else:
                raise ValueError(f"coords must be [N, 3] or [1, N, 3], got {coords.shape}")
        else:
            # Latent2Volume/Latent2Genera expect: [N, z_dim], [N, 3]
            coords_batch = coords if coords.dim() == 2 else coords.squeeze(0)    # [N, 3]
            latent_batch = latent.expand(coords_batch.shape[0], -1) if latent.dim() == 1 else latent
        sdf_batch = model(latent_batch.to(DEV), coords_batch.to(DEV))
    return sdf_batch.unsqueeze(0)  # [1, N]

def compute_genus(vertices, faces):
    """
    Compute the genus of the object represented by the latent code.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    V = mesh.vertices.shape[0]
    E = mesh.edges_unique.shape[0]
    F = mesh.faces.shape[0]

    # Check for connected components (for multi-component genus computation)
    boundaries = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    B = len(trimesh.graph.connected_components(boundaries))
    C = len(mesh.split(only_watertight=False))

    euler_char = V - E + F
    genus = (2 * C - euler_char - B) // 2

    return genus


def compute_volume(vertices, faces):
    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)
    if not isinstance(faces, np.ndarray):
        faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    volume = abs(mesh.volume)
        
    return volume

def match_volume(vertices, faces, target_volume=10):
    volume = compute_volume(vertices, faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    scale = (target_volume / abs(volume)) ** (1/3)
    mesh.apply_scale(scale)

    return mesh.vertices, mesh.faces

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
    latents = torch.rand(n, LATENT_DIM) * LATENT_VEC_MAX * 2 - LATENT_VEC_MAX
    data_dir = os.path.join(VOLUME_DIR, "data") #"volume/data/"

    volumes = []
    genera = []

    coords, grid_size = get_volume_coords(resolution=50)
    ps.init()
    for i, latent in enumerate(tqdm(latents)):
        vertices, faces = generate_mesh_from_latent(latent, coords, grid_size, model, LATENT_FIRST)
        
        ps.register_surface_mesh(f"mesh_{i}", vertices, faces)
        ps.set_up_dir("z_up")
        ps.show()

        volumes.append(compute_volume(vertices, faces))
        genera.append(compute_genus(vertices, faces))

    print(genera)
    print(volumes)
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
    
    np.savez(os.path.join(data_dir, "2d_latents_volumes.npz"),
                latents=np.array(latents, dtype=np.float32),
                volumes=np.array(volumes, dtype=np.float32),
                genera=np.array(genera, dtype=np.int8))
    


if __name__ == "__main__":
    #----------------------------------------------------------------

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config_examples.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    best_model_path = r"C:\Users\singh\OneDrive\Documents\GitHub\topology-control\volume\trained_deepsdfs\best_model.pth"

    arch_manager = CArchitectureManager(config['model_config'])
    model = arch_manager.get_model().to(DEV)
    ckpt = torch.load(best_model_path, map_location=DEV)
    model.load_state_dict(ckpt['model_state_dict'])

    # # Data Generation
    # model_path = "trained_deepsdfs/sdfnet_model.pt"
    # scripted_model = torch.jit.load(model_path).to(DEV)

    generate_latent_volume_data(200, model)
    # generate_syn_latent_volume_data(2000)